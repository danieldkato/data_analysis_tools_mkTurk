#!/usr/bin/env python3
import sys
from pathlib import Path
import getpass
import paramiko
import stat
import threading
import time
from tqdm import tqdm

SFTP_HOST_1 = "129.236.163.83"
SFTP_HOST_2 = "129.236.162.218"


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

    with tqdm(
        total=file_size,
        unit="B",
        unit_scale=True,
        desc=f"{host_label}:{local_path.name}",
        leave=False,
    ) as pbar:

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
    """Connect to server, list top-level dirs, and prompt for destination"""
    try:
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
            sftp.close()
            transport.close()
            sys.exit(1)

        sftp.close()
        transport.close()
        return destination

    except Exception as e:
        print(f"[{host}] Error connecting: {e}")
        sys.exit(1)


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


def main():
    if len(sys.argv) < 3:
        print("Usage: script.py <subject> <date1> [date2 date3 ...]")
        sys.exit(1)

    subject = sys.argv[1]
    dates = sys.argv[2:]

    print(f"Subject: {subject}")
    print(f"Dates: {', '.join(dates)}")

    folders = []
    for date in dates:
        folder = find_matching_folder(subject, date)
        print(f"Matched folder for {date}: {folder}")
        folders.append(folder)

    # Get credentials and destinations first
    print("\n--- Credentials for Server 1 ---")
    username1 = input("Username: ") 
    password1 = getpass.getpass("Password: ")
    destination1 = get_destination(SFTP_HOST_1, username1, password1)

    print("\n--- Credentials for Server 2 ---")
    username2 = input("Username: ")
    password2 = getpass.getpass("Password: ")
    destination2 = get_destination(SFTP_HOST_2, username2, password2)

    # Start transfers in parallel
    t1 = threading.Thread(
        target=transfer_to_server,
        args=(SFTP_HOST_1, username1, password1, folders, destination1)
    )

    t2 = threading.Thread(
        target=transfer_to_server,
        args=(SFTP_HOST_2, username2, password2, folders, destination2)
    )

    t1.start()
    t2.start()

    t1.join()
    t2.join()

    print("\nAll transfers finished.")


if __name__ == "__main__":
    main()