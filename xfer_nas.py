#!/usr/bin/env python3
import sys
import argparse
from pathlib import Path
import getpass
import paramiko
import stat
import threading
import time
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

SFTP_HOST_1 = "129.236.163.83"
SFTP_HOST_2 = "129.236.162.218"


# -------------------- EXISTING UPLOAD HELPERS -------------------- #

def find_matching_folder(subject, date):
    data_dir = Path("/mnt/smb/locker/issa-locker/Data")
    subject_dir = data_dir / subject

    if not subject_dir.exists():
        print(f"Error: subject folder not found: {subject_dir}")
        sys.exit(1)

    matches = [
        p for p in subject_dir.iterdir()
        if p.is_dir() and subject in p.name and date in p.name
    ]

    if len(matches) == 1:
        return matches[0].resolve()
    elif len(matches) == 0:
        print(f"Error: no folders found for date {date}")
        sys.exit(1)
    else:
        print(f"Error: multiple folders found for date {date}:")
        for m in matches:
            print(m)
        sys.exit(1)


def upload_file(sftp, local_path, remote_path, host_label):
    file_size = local_path.stat().st_size
    start_time = time.time()

    with tqdm(total=file_size, unit="B", unit_scale=True,
              desc=f"{host_label}:{local_path.name}", leave=False) as pbar:

        def callback(transferred, total):
            pbar.update(transferred - pbar.n)

        sftp.put(str(local_path), remote_path, callback=callback)

    elapsed = time.time() - start_time
    speed = file_size / elapsed / 1e6 if elapsed > 0 else 0
    print(f"[{host_label}] {local_path.name} complete ({speed:.2f} MB/s)")


def upload_directory(sftp, local_path, remote_path, host_label):
    try:
        sftp.mkdir(remote_path)
    except IOError:
        pass

    for item in local_path.iterdir():
        remote_item = f"{remote_path}/{item.name}"
        if item.is_dir():
            upload_directory(sftp, item, remote_item, host_label)
        else:
            upload_file(sftp, item, remote_item, host_label)


def get_destination(host, username, password):
    transport = paramiko.Transport((host, 22))
    transport.connect(username=username, password=password)
    sftp = paramiko.SFTPClient.from_transport(transport)

    print(f"\nConnected to {host}")
    print("Top-level directories:")

    top_dirs = []
    for entry in sftp.listdir_attr("/"):
        if stat.S_ISDIR(entry.st_mode):
            print(f"[{host}] - {entry.filename}")
            top_dirs.append(entry.filename)

    destination = input(f"\n[{host}] Enter destination directory: ").strip()

    if destination not in top_dirs:
        print(f"[{host}] Error: destination '{destination}' not found.")
        sys.exit(1)

    sftp.close()
    transport.close()
    return destination


def transfer_to_server(host, username, password, folders, destination):
    try:
        transport = paramiko.Transport((host, 22))
        transport.connect(username=username, password=password)
        sftp = paramiko.SFTPClient.from_transport(transport)

        for folder in folders:
            remote_base = f"/{destination}/{folder.name}"
            print(f"[{host}] Transferring {folder.name} -> {remote_base}")
            upload_directory(sftp, folder, remote_base, host)

        print(f"[{host}] Transfer complete.")
        sftp.close()
        transport.close()

    except Exception as e:
        print(f"[{host}] Error: {e}")


# -------------------- DOWNLOAD HELPERS -------------------- #

MAX_WORKERS = 6  # adjust (4–10 is usually good)


def collect_matching_files(sftp, remote_path, subject, date, patterns, results):
    """Recursively collect matching file paths."""
    try:
        for entry in sftp.listdir_attr(remote_path):
            name = entry.filename
            full_path = f"{remote_path}/{name}"

            if "dk_g" in name:
                continue

            if stat.S_ISDIR(entry.st_mode):
                if subject in name and date in name:
                    collect_all_files(sftp, full_path, patterns, results)
                else:
                    collect_matching_files(
                        sftp, full_path, subject, date, patterns, results
                    )
    except Exception:
        pass


def collect_all_files(sftp, remote_path, patterns, results):
    """Collect files inside a matched directory."""
    try:
        for entry in sftp.listdir_attr(remote_path):
            name = entry.filename
            full_path = f"{remote_path}/{name}"

            if "dk_g" in name:
                continue

            if stat.S_ISDIR(entry.st_mode):
                collect_all_files(sftp, full_path, patterns, results)
            else:
                if any(p in name for p in patterns):
                    results.append(full_path)
    except Exception:
        pass


def download_worker(host, username, password, remote_path):
    """Each worker creates its own connection."""
    try:
        transport = paramiko.Transport((host, 22))
        transport.connect(username=username, password=password)
        sftp = paramiko.SFTPClient.from_transport(transport)

        filename = os.path.basename(remote_path)
        local_path = Path(".") / filename

        file_size = sftp.stat(remote_path).st_size

        with tqdm(
            total=file_size,
            unit="B",
            unit_scale=True,
            desc=f"{host}:{filename}",
            leave=False,
        ) as pbar:

            def callback(transferred, total):
                pbar.update(transferred - pbar.n)

            sftp.get(remote_path, str(local_path), callback=callback)

        sftp.close()
        transport.close()

        return f"[{host}] Done: {filename}"

    except Exception as e:
        return f"[{host}] Error: {remote_path} ({e})"


def download_from_server(host, username, password, subject, dates, patterns):
    try:
        transport = paramiko.Transport((host, 22))
        transport.connect(username=username, password=password)
        sftp = paramiko.SFTPClient.from_transport(transport)

        print(f"[{host}] Scanning for files...")

        results = []
        for date in dates:
            collect_matching_files(sftp, "/", subject, date, patterns, results)

        sftp.close()
        transport.close()

        if not results:
            print(f"[{host}] No matching files found.")
            return

        print(f"[{host}] Found {len(results)} files. Starting parallel download...")

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [
                executor.submit(download_worker, host, username, password, path)
                for path in results
            ]

            for future in as_completed(futures):
                print(future.result())

        print(f"[{host}] All downloads complete.")

    except Exception as e:
        print(f"[{host}] Error: {e}")


# -------------------- MAIN -------------------- #

def main():
    parser = argparse.ArgumentParser(description="SFTP transfer tool")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Upload subcommand
    upload_parser = subparsers.add_parser("upload")
    upload_parser.add_argument("subject")
    upload_parser.add_argument("dates", nargs="+")

    # Download subcommand
    download_parser = subparsers.add_parser("download")
    download_parser.add_argument("server", choices=["1", "2"])
    download_parser.add_argument("subject")
    download_parser.add_argument("dates", nargs="+")
    download_parser.add_argument("--patterns", nargs="+", required=True)

    args = parser.parse_args()

    # ---------------- UPLOAD ---------------- #
    if args.command == "upload":
        subject = args.subject
        dates = args.dates

        folders = [find_matching_folder(subject, d) for d in dates]

        print("\n--- Credentials for Server 1 ---")
        username1 = input("Username: ")
        password1 = getpass.getpass("Password: ")
        dest1 = get_destination(SFTP_HOST_1, username1, password1)

        print("\n--- Credentials for Server 2 ---")
        username2 = input("Username: ")
        password2 = getpass.getpass("Password: ")
        dest2 = get_destination(SFTP_HOST_2, username2, password2)

        t1 = threading.Thread(
            target=transfer_to_server,
            args=(SFTP_HOST_1, username1, password1, folders, dest1)
        )
        t2 = threading.Thread(
            target=transfer_to_server,
            args=(SFTP_HOST_2, username2, password2, folders, dest2)
        )

        t1.start()
        t2.start()
        t1.join()
        t2.join()

        print("\nAll uploads finished.")

    # ---------------- DOWNLOAD ---------------- #
    elif args.command == "download":
        host = SFTP_HOST_1 if args.server == "1" else SFTP_HOST_2

        username = input("Username: ")
        password = getpass.getpass("Password: ")

        download_from_server(
            host,
            username,
            password,
            args.subject,
            args.dates,
            args.patterns
        )


if __name__ == "__main__":
    main()