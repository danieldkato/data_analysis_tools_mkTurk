#!/bin/bash
#SBATCH --job-name=dartsort
#SBATCH --account=issa
#SBATCH --chdir=/home/yy3658/NeuralWaveform
#SBATCH --cpus-per-task=32
#SBATCH --mem=128gb
#SBATCH --partition=issa
#SBATCH --gres=gpu:8
#SBATCH --nodelist=ax09
#SBATCH --output=/home/dk2643/dartsort_%A.out
#
# Runs DARTsort for one session ($MONKEY/$DATE), then chains the Kilosort sbatch.
#   1. Mount the NAS (the raw .ap.bin may live there, not on engram).
#   2. Run dartsort with --keep-stage: the preprocessed copy is staged but NOT
#      removed, so the Kilosort run that follows reuses it (same shared /local
#      lock + session-named staged dir; see staging.py) instead of re-staging.
#   3. After dartsort succeeds, submit run_kilosort.sh with
#      --dependency=afterok so Kilosort starts only if dartsort succeeded.

set -u

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /share/issa/users/yy3658/shared_env/data_processing

echo "=== Resource Validation ==="
echo "Node: $(hostname)"
echo "Monkey: $MONKEY, Date: $DATE"
echo "CPUs allocated: $SLURM_CPUS_PER_TASK"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# Validate GPUs
N_GPUS=$(python3 -c "import torch; print(torch.cuda.device_count())")
echo "PyTorch visible GPUs: $N_GPUS"
if [ "$N_GPUS" -eq 0 ]; then
    echo "ERROR: No GPUs detected. Exiting."
    exit 1
fi
nvidia-smi --query-gpu=index,name,memory.free --format=csv,noheader
echo "=========================="

monkey="$MONKEY"
date="$DATE"

# This script lives in <pkg>/spike_sorting/; resolve the package parent so the
# modules (which use intra-package relative imports) can be run via -m, and the
# sorting dir so we can submit the sibling kilosort sbatch.
THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_PARENT="$(dirname "$(dirname "$THIS_DIR")")"   # dir containing data_analysis_tools_mkTurk/

# Manifest of staged dirs created by run_dartsort, swept on SIGKILL/timeout (the
# only case the Python finally misses). With --keep-stage we deliberately do NOT
# sweep on a clean exit — the staged copy is left for the kilosort run — so the
# trap only fires on an abnormal exit and only touches our own staged dirs.
STAGING_MANIFEST="${NEURALWF_STAGING_MANIFEST:-/tmp/dartsort_staging_${SLURM_JOB_ID:-0}.manifest}"
export NEURALWF_STAGING_MANIFEST="$STAGING_MANIFEST"
rm -f "$STAGING_MANIFEST" 2>/dev/null || true
DARTSORT_OK=0
cleanup() {
    # On a clean exit with --keep-stage we KEEP the staged copy for kilosort.
    if (( DARTSORT_OK == 1 )); then
        echo "dartsort succeeded; keeping staged copy for kilosort (--keep-stage)"
        return 0
    fi
    [ -f "$STAGING_MANIFEST" ] || return 0
    while IFS= read -r d; do
        case "$d" in
            /local/neuralwf_staging*|*/rec_ppx.staging.*)   # only our own staged dirs
                echo "Sweeping leftover staged dir: $d"; rm -rf "$d" ;;
        esac
    done < "$STAGING_MANIFEST"
    rm -f "$STAGING_MANIFEST"
}
trap cleanup EXIT
trap 'trap - EXIT; cleanup; exit 143' TERM INT

# --- Mount the NAS for this monkey (raw .ap.bin may live there) ---
# Key-based auth (public key in issalab's authorized_keys on each server). Mount
# layout: $HOME/mounted_nas/<monkey>/<monkey><tag>_<x>  (tag: .83->2, .218->3;
# <x> = digit in share name, none -> 1). A session may be duplicated across
# mounts; run_dartsort picks the first copy with the .ap.bin.
SSHFS_OPTS="IdentityFile=$HOME/.ssh/id_ed25519,IdentitiesOnly=yes,reconnect,ServerAliveInterval=15,ServerAliveCountMax=6,TCPKeepAlive=yes,ConnectTimeout=10,Ciphers=aes128-gcm@openssh.com,Compression=no,big_writes,max_read=131072,cache=yes,kernel_cache,cache_timeout=115200"

NAS_SHARES=(
  "Bourgeois  129.236.163.83   2"
  "West       129.236.163.83   2"
  "West2      129.236.163.83   2"
  "West3      129.236.163.83   2"
  "Bourgeois  129.236.162.218  3"
  "Bourgeois2 129.236.162.218  3"
  "West       129.236.162.218  3"
  "West2      129.236.162.218  3"
)

mount_share() {
  local host="$1" share="$2" mnt="$3"
  mkdir -p "$mnt"
  echo "Mounting ${share} from ${host} at $mnt ..."
  sshfs "issalab@${host}:/${share}" "$mnt" -o "$SSHFS_OPTS"
  if ! mountpoint -q "$mnt" || ! ls "$mnt" >/dev/null 2>&1; then
    echo "ERROR: ${share} failed to mount at $mnt"
    return 1
  fi
}

# Per-job mount root so concurrent jobs on the same node never unmount each
# other's shares mid-read.
NAS_ROOT="$HOME/mounted_nas/dartsort_${SLURM_JOB_ID:-0}"
unmount_nas() {
  for m in "${NAS_MNTS[@]:-}"; do
    [ -n "$m" ] && fusermount -u "$m" 2>/dev/null || true
  done
}
# Chain NAS unmount onto the existing EXIT/TERM/INT cleanup.
trap 'unmount_nas; cleanup' EXIT
trap 'trap - EXIT; unmount_nas; cleanup; exit 143' TERM INT

NAS_MNTS=()
for entry in "${NAS_SHARES[@]}"; do
  read -r share host tag <<<"$entry"
  smonkey="${share%[0-9]}"
  [[ "$smonkey" == "$monkey" ]] || continue   # only this job's monkey
  x="${share##*[!0-9]}"; [[ -z "$x" ]] && x=1
  mnt="$NAS_ROOT/${smonkey}/${smonkey}${tag}_${x}"
  if mount_share "$host" "$share" "$mnt"; then
    NAS_MNTS+=("$mnt")
  else
    echo "WARNING: skipping $mnt (mount failed); other copies may still serve this session"
  fi
done
if (( ${#NAS_MNTS[@]} == 0 )); then
  echo "WARNING: no NAS share mounted for monkey '$monkey' — relying on engram copy"
fi

export NEURALWF_NAS_ROOT="$NAS_ROOT"
export PYTHONUNBUFFERED=1

# --- Run dartsort with --keep-stage (stage, but keep the copy for kilosort) ---
echo "--- Running DARTsort (--keep-stage) for $monkey $date ---"
cd "$PKG_PARENT"
python -m data_analysis_tools_mkTurk.spike_sorting.run_dartsort \
    --monkey "$monkey" --date "$date" --keep-stage
DARTSORT_OK=1
echo "DARTsort completed for $monkey $date"

# --- Chain the Kilosort sbatch, only if dartsort succeeded (afterok) ---
echo "--- Submitting Kilosort (depends on this dartsort job) ---"
sbatch --dependency=afterok:"${SLURM_JOB_ID}" \
    --export=ALL,MONKEY="$monkey",DATE="$date" \
    "$THIS_DIR/run_kilosort.sh"
