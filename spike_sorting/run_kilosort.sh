#!/bin/bash
#SBATCH --job-name=kilosort
#SBATCH --account=yy3658
#SBATCH --chdir=/home/yy3658/NeuralWaveform
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 16
#SBATCH --gres=gpu:2
#SBATCH --time=15:00:00
#SBATCH --mem=60gb
#SBATCH --partition=issa
#SBATCH --nodelist=ax09
#SBATCH --output=/home/dk2643/kilosort_%A.out

### Single-session Kilosort4, pinned to ax09. Normally submitted by run_dartsort.sh
### (--dependency=afterok, --export=MONKEY,DATE) so it reuses dartsort's kept staged
### copy instead of re-staging. run_kilosort skips already-complete sessions and picks
### the staging destination itself (choose_stage_mode: /local float16, else engram
### float32; see staging.py). The node's real /local is bound in so the shared lock works.
###
### Standalone fallback (MONKEY/DATE unset): array + SESSION_LIST, one session per task:
###   sbatch --array=0-$((N-1))%8 --export=ALL,SESSION_LIST=my_list.txt run_kilosort.sh
### (--nodelist=ax09 pins all tasks to ax09; remove it for a wide array.)

set -u
mkdir -p /home/yy3658/NeuralWaveform/jobs/logs/

# Single-session mode: MONKEY/DATE come from the dartsort sbatch via --export.
# If unset, fall back to SESSION_LIST / array mode (one session per array task).
if [[ -n "${MONKEY:-}" && -n "${DATE:-}" ]]; then
  monkey="$MONKEY"
  date="$DATE"
  echo "=== [single session] $(hostname)  $(date '+%F %T') ==="
  echo "monkey=$monkey date=$date"
else
  SESSION_LIST="${SESSION_LIST:-all_sessions_runnable.txt}"
  if [[ ! -f "$SESSION_LIST" ]]; then
    echo "ERROR: MONKEY/DATE not set and session list not found: $SESSION_LIST"; exit 1
  fi
  mapfile -t SESSIONS < <(grep -v '^[[:space:]]*$' "$SESSION_LIST")
  TOTAL=${#SESSIONS[@]}
  IDX=${SLURM_ARRAY_TASK_ID:-0}
  if (( IDX >= TOTAL )); then
    echo "task $IDX >= session count $TOTAL; nothing to do"; exit 0
  fi
  SESS="${SESSIONS[$IDX]}"
  monkey="${SESS%%_*}"
  rest="${SESS#*_}"
  date="${rest%%_*}"
  if [[ ! "$date" =~ ^[0-9]{8}$ ]]; then
    echo "ERROR: cannot parse monkey/date from '$SESS'"; exit 1
  fi
  echo "=== [task $IDX/$((TOTAL-1))] $(hostname)  $(date '+%F %T') ==="
  echo "session: $SESS  ->  monkey=$monkey date=$date"
fi

# Manifest of staged dirs this task creates: written by run_kilosort, swept by the
# cleanup trap on SIGKILL/timeout (the only case the Python finally misses). Path
# shared via NEURALWF_STAGING_MANIFEST so Python and this wrapper agree.
STAGING_MANIFEST="/home/yy3658/NeuralWaveform/jobs/logs/staging_${SLURM_JOB_ID:-0}.manifest"
export NEURALWF_STAGING_MANIFEST="$STAGING_MANIFEST"
rm -f "$STAGING_MANIFEST" 2>/dev/null || true   # start clean (e.g. on requeue)

# --- Mount the NAS on this compute node before launching the container ---
# Key-based auth (issalab pubkey in authorized_keys on each server).
# NOTE: do NOT add StrictHostKeyChecking=accept-new — it triggers a connection reset
# on these servers; rely on the cached host key in ~/.ssh/known_hosts.
#
# Two Synology NAS servers, each exporting several shares. Mount layout:
#   $NAS_ROOT/<monkey>/<monkey><tag>_<x>   (<tag>: .83->2, .218->3; <x>: digit in
# the share name, none->1). A session may be duplicated across mounts; run_kilosort
# picks the first copy with the .ap.bin. We mount only THIS task's monkey, since the
# session can only live under its own monkey.
SSHFS_OPTS="IdentityFile=$HOME/.ssh/id_ed25519,IdentitiesOnly=yes,reconnect,ServerAliveInterval=15,ServerAliveCountMax=6,TCPKeepAlive=yes,ConnectTimeout=10,Ciphers=aes128-gcm@openssh.com,Compression=no,big_writes,max_read=131072,cache=yes,kernel_cache,cache_timeout=115200"

# "share host tag" per line; mounted only if the share's monkey == $monkey.
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

# Mount one share. The mountpoint is per-task (see NAS_ROOT below), so no sibling
# ever shares it — nothing stale to clear, just mount fresh and verify. Returns
# non-zero on failure.
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

# Per-task mount root. Concurrent array tasks often land on the SAME node (node
# packing + %N throttle); the old shared mountpoint meant one task's pre-mount
# `fusermount -u` killed a share a sibling was mid-read on -> invalid mmap -> Bus
# error / FileNotFoundError mid-sort. A per-task root removes that collision.
NAS_ROOT="$HOME/mounted_nas/task_${SLURM_JOB_ID:-0}_${IDX}"

NAS_MNTS=()
for entry in "${NAS_SHARES[@]}"; do
  read -r share host tag <<<"$entry"
  smonkey="${share%[0-9]}"
  [[ "$smonkey" == "$monkey" ]] || continue   # only this task's monkey
  x="${share##*[!0-9]}"; [[ -z "$x" ]] && x=1
  mnt="$NAS_ROOT/${smonkey}/${smonkey}${tag}_${x}"
  if mount_share "$host" "$share" "$mnt"; then
    NAS_MNTS+=("$mnt")
  else
    echo "WARNING: skipping $mnt (mount failed); other copies may still serve this session"
  fi
done

if (( ${#NAS_MNTS[@]} == 0 )); then
  echo "ERROR: no NAS share mounted for monkey '$monkey' — aborting task"
  exit 1
fi

# --- Shared node-local scratch (/local) for staging ---
# A node stages at most ONE recording on /local at a time, under the fixed dir
# /local/neuralwf_staging (shared with run_dartsort). Binding the node's REAL /local
# lets run_kilosort atomically claim it (mkdir): the winner stages there (float16),
# everyone else stages float32 to engram. A dir orphaned by SIGKILL is reclaimed by
# run_kilosort once stale (NEURALWF_LOCAL_STALE_S, default 12 h) — no bash pre-sweep,
# and we must NOT delete it here since it may belong to a LIVE sibling.
echo "--- host /local check ($(hostname)) ---"
if [[ ! -d /local ]]; then
  echo "ERROR: host /local missing — aborting"; exit 1
fi
echo "--- /local free space ---"
df -h /local || true

# Always unmount the NAS and drop this task's staging copies when it ends — normal
# exit, failure, AND SLURM kills (scancel / walltime / OOM send SIGTERM, which a bare
# EXIT trap misses). SIGKILL can't be trapped, but run_kilosort's stale-reclaim
# recovers an abandoned /local slot, and an orphaned per-task mount under
# $HOME/mounted_nas/task_* is unique; sweep those periodically if KILLs are frequent.
cleanup() {
  for m in "${NAS_MNTS[@]}"; do
    if mountpoint -q "$m" 2>/dev/null || stat "$m" >/dev/null 2>&1; then
      echo "--- unmounting NAS ($m) ---"
      fusermount -u "$m" 2>/dev/null || fusermount -uz "$m" 2>/dev/null || true
    fi
  done
  # Remove this task's now-empty mount root so $HOME/mounted_nas doesn't fill up.
  [[ -n "${NAS_ROOT:-}" ]] && rm -rf "$NAS_ROOT" 2>/dev/null || true
  # Release this task's staging copies on SIGKILL/timeout (run_kilosort already does
  # this and clears the manifest on clean exit). We trust the manifest's exact dirs:
  # the /local slot and/or the engram float32 dir; the case filter below allows only
  # those two known names, never arbitrary paths. The engram dir is job-id-stamped so
  # removing it only hits OUR copy. The /local slot name is shared, so we delete it
  # only when STALE (same check as run_kilosort's reclaim) to never drop a live
  # sibling's stage.
  local stale_min="${NEURALWF_LOCAL_STALE_MIN:-720}"   # 12 h, matches Python
  if [[ -f "$STAGING_MANIFEST" ]]; then
    while IFS= read -r staged; do
      case "$staged" in
        /local/neuralwf_staging)
          [[ -e "$staged" ]] || continue
          if [[ -n "$(find "$staged" -type f -mmin "-${stale_min}" -print -quit 2>/dev/null)" ]]; then
            echo "keeping /local slot (live, likely a sibling): $staged"; continue
          fi
          ;;
        */rec_ppx.staging.${SLURM_JOB_ID:-0}) ;;
        *) echo "skip non-staging path in manifest: $staged"; continue ;;
      esac
      [[ -e "$staged" ]] && { echo "releasing staged copy: $staged"; rm -rf "$staged" 2>/dev/null || true; }
    done < "$STAGING_MANIFEST"
    rm -f "$STAGING_MANIFEST" 2>/dev/null || true
  fi
}
trap cleanup EXIT
trap 'trap - EXIT; cleanup; exit 143' TERM INT

for m in "${NAS_MNTS[@]}"; do
  echo "--- NAS mount check ($m) ---"
  mountpoint "$m" || echo "WARNING: $m is not a mountpoint"
done

SIF=~/vscode.sif
PYTHON_BIN=/home/yy3658/shared_env/data_processing/bin/python
# Bind the per-task mount root ($NAS_ROOT) so every mounted share is visible in the
# container (run_kilosort searches it via NEURALWF_NAS_ROOT) — do NOT reassign
# NAS_ROOT to the shared tree, or sibling tasks collide. Bind the node's REAL /local
# so run_kilosort can stage there AND see siblings' stage dirs for the contention
# check — do NOT bind a per-task /local subdir, or every task claims it and the
# engram/NAS float32 overflow never triggers.
PYTHON="apptainer exec \
  -B /run:/run \
  --mount type=bind,src=/mnt/smb/locker/issa-locker,dst=/mnt/smb/locker/issa-locker \
  --mount type=bind,src=/share/issa,dst=/share/issa \
  --mount type=bind,src=$NAS_ROOT,dst=$NAS_ROOT \
  --mount type=bind,src=/local,dst=/local \
  --nv $SIF $PYTHON_BIN"
echo "Using Apptainer: $SIF"

export NEURALWF_NAS_ROOT="$NAS_ROOT"
export PYTHONUNBUFFERED=1

$PYTHON -V
echo "GPUs visible: ${CUDA_VISIBLE_DEVICES:-unset}"

# run_kilosort and analyze_bystim use intra-package relative imports, so they run via
# -m from the dir CONTAINING data_analysis_tools_mkTurk/. This script lives in
# <pkg>/spike_sorting/, so that parent is two dirs up (same derivation as
# run_dartsort.sh; override NEURALWF_PKG_PARENT to relocate). cd there and `python -m`
# directly: cwd becomes sys.path[0] and the container inherits it via the bind mounts.
# (No --no-stage: every session is staged + preprocessed; run_kilosort just picks the
# destination — /local float16 if free, else engram float32; see choose_stage_mode.)
THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_PARENT="${NEURALWF_PKG_PARENT:-$(dirname "$(dirname "$THIS_DIR")")}"
cd "$PKG_PARENT"
echo "--- Running Kilosort4 (auto staging) for $monkey $date ---"
$PYTHON -u -m data_analysis_tools_mkTurk.spike_sorting.run_kilosort --monkey "$monkey" --date "$date"
RC=$?
echo "--- [task $IDX] $SESS kilosort exit code: $RC ($(date '+%F %T')) ---"

# Only on a clean sort: build per-stimulus PSTHs for every sorted unit inside this
# job (rather than a manual notebook step). analyze_bystim derives the unit list from
# the KSLabel.npy the sort just wrote, writes one clu{nnn}_psth_stim per unit, and
# skips when already complete (cheap requeues). A by_stim failure is logged but does
# NOT change the job's exit status — the sort itself succeeded. (Still cd'd to
# $PKG_PARENT from above.)
if (( RC == 0 )); then
  echo "--- Running analyze_bystim (kilosort) for $monkey $date ---"
  $PYTHON -u -m data_analysis_tools_mkTurk.analyze_bystim --monkey "$monkey" --date "$date" --source kilosort
  BYSTIM_RC=$?
  echo "--- [task $IDX] $SESS by_stim exit code: $BYSTIM_RC ($(date '+%F %T')) ---"
fi

# Only on a clean by_stim: combine the per-unit PSTHs (plus quality / template
# metrics already written by run_kilosort) into the single-unit session HDF5 via
# process_ks_data. It is idempotent — by_stim and metrics skip when already
# complete, so this effectively just runs the IO/H5 step here. A failure is logged
# but does NOT change the job's exit status. (Still cd'd to $PKG_PARENT.)
if (( RC == 0 && BYSTIM_RC == 0 )); then
  echo "--- Building single-unit HDF5 (process_ks_data) for $monkey $date ---"
  $PYTHON -u -m data_analysis_tools_mkTurk.spike_sorting.process_ks_data --monkey "$monkey" --date "$date"
  H5_RC=$?
  echo "--- [task $IDX] $SESS process_ks_data exit code: $H5_RC ($(date '+%F %T')) ---"
fi

echo "--- [task $IDX] $SESS done ($(date '+%F %T')) ---"
exit $RC
