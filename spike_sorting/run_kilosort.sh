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
### (--dependency=afterok, --export=MONKEY,DATE), so it reuses dartsort's kept staged
### copy (shared /local lock + session-named dir; see staging.py) instead of
### re-staging. run_kilosort skips sessions already complete; staging destination is
### chosen automatically by choose_stage_mode (engram float32 / /local float16). The
### node's real /local is bound into the container so the shared lock works.
###
### Standalone batch fallback (when MONKEY/DATE are unset): add --array and a
### SESSION_LIST to run many sessions, one per task, e.g.
###   sbatch --array=0-$((N-1))%8 --export=ALL,SESSION_LIST=my_list.txt run_kilosort.sh
### (--nodelist=ax09 above pins all tasks to ax09; remove it for a wide array.)

set -u
mkdir -p /home/yy3658/NeuralWaveform/jobs/logs/

# Single-session mode (the workflow): MONKEY/DATE come from the dartsort sbatch via
# --export. If they are not set, fall back to session-list / array mode for
# standalone batch use (sbatch --array=0-N%K with a SESSION_LIST).
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

# Manifest of staged dirs this task creates, written by run_kilosort and swept by
# the cleanup trap on SIGKILL/timeout (the only case the Python finally misses).
# Shared via NEURALWF_STAGING_MANIFEST so Python and this wrapper agree on path.
STAGING_MANIFEST="/home/yy3658/NeuralWaveform/jobs/logs/staging_${SLURM_JOB_ID:-0}.manifest"
export NEURALWF_STAGING_MANIFEST="$STAGING_MANIFEST"
rm -f "$STAGING_MANIFEST" 2>/dev/null || true   # start clean (e.g. on requeue)

# --- Mount the NAS on this compute node before launching the container ---
# Key-based auth (public key in issalab's authorized_keys on each server).
# NOTE: do NOT add StrictHostKeyChecking=accept-new — it triggers a connection
# reset on these servers; rely on the cached host key in ~/.ssh/known_hosts.
#
# Two Synology NAS servers, each exporting several shares. Mount layout:
#   $HOME/mounted_nas/<monkey>/<monkey><tag>_<x>
# where <tag> identifies the server (.83 -> 2, .218 -> 3) and <x> is the digit
# in the share name (no digit -> 1). A session may be duplicated across these
# mounts; run_kilosort picks the first copy with the .ap.bin. We only mount THIS
# task's monkey, since the session can only live under its own monkey.
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

# Mount one share, recovering from any stale/dead FUSE endpoint first. A crashed
# Mount one share. The mountpoint is per-task (see NAS_ROOT below), so no sibling
# task ever shares it — there is no stale endpoint to clear, we just mount fresh
# and verify. Returns non-zero on failure.
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

# Per-task mount root. Concurrent array tasks frequently land on the SAME node
# (node packing + %N throttle), and the old shared mountpoint
# ($HOME/mounted_nas/<monkey>/<share>) meant every task's pre-mount cleanup
# `fusermount -u`'d the share a sibling task was mid-read on -> the running
# task's mmap went invalid -> Bus error / FileNotFoundError mid-sort. Giving
# each task its own mount root removes that cross-task collision entirely.
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
# A node stages at most ONE recording on /local at a time, under the single fixed
# dir /local/neuralwf_staging (shared with run_dartsort). We bind the node's REAL
# /local into the container
# so run_kilosort can atomically claim that dir (mkdir): whoever wins stages there
# (float16), everyone else stages float32 to engram instead. A
# dir orphaned by a SIGKILL is reclaimed by run_kilosort once it goes stale
# (NEURALWF_LOCAL_STALE_S, default 12 h) — no bash pre-sweep needed, and we must
# NOT delete it here since it may belong to a LIVE sibling.
echo "--- host /local check ($(hostname)) ---"
if [[ ! -d /local ]]; then
  echo "ERROR: host /local missing — aborting"; exit 1
fi
echo "--- /local free space ---"
df -h /local || true

# Always unmount the NAS and drop this task's staging copies when it ends — on
# normal exit, on failure, AND on SLURM kills (scancel / walltime / OOM send
# SIGTERM, which EXIT alone does not catch). SIGKILL can't be trapped, but the
# stale-reclaim in run_kilosort recovers an abandoned /local slot, and an orphaned
# mount under $HOME/mounted_nas/task_* is unique-per-task; sweep those periodically
# if KILLs are frequent.
cleanup() {
  for m in "${NAS_MNTS[@]}"; do
    if mountpoint -q "$m" 2>/dev/null || stat "$m" >/dev/null 2>&1; then
      echo "--- unmounting NAS ($m) ---"
      fusermount -u "$m" 2>/dev/null || fusermount -uz "$m" 2>/dev/null || true
    fi
  done
  # Remove this task's now-empty mount root so $HOME/mounted_nas doesn't fill up.
  [[ -n "${NAS_ROOT:-}" ]] && rm -rf "$NAS_ROOT" 2>/dev/null || true
  # Release this task's staging copies on SIGKILL/timeout (run_kilosort already
  # does this and clears the manifest on a clean exit). We trust the manifest,
  # which run_kilosort wrote with the exact dirs it created: the /local slot
  # (/local/neuralwf_staging) and/or its engram float32 dir. The safety filter
  # only allows those two known staging names, never arbitrary paths.
  #
  # The engram dir is job-id-stamped, so removing it can only ever hit OUR copy.
  # The /local slot name is shared, so a stale manifest could in theory name a
  # slot a SIBLING has since re-claimed; we therefore only delete /local when it
  # is STALE (no file written recently), exactly as run_kilosort's reclaim does,
  # so we never delete a live sibling's stage.
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
# Bind this task's per-task mount root so every mounted share is visible in the
# container; run_kilosort searches this monkey's mounts under NEURALWF_NAS_ROOT.
# NAS_ROOT was set above to a per-task path (do NOT reassign it to the shared
# tree here, or sibling tasks collide again).
# Bind the node's REAL /local so run_kilosort can stage there AND see sibling
# tasks' stage dirs for the contention check (do NOT bind a per-task subdir, or
# every task claims /local and the engram/NAS float32 overflow never triggers).
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

# No --no-stage: every session is staged + preprocessed; run_kilosort only picks
# the temp destination (/local float16 if free, else engram float32; see
# choose_stage_mode in staging.py).
#
# run_kilosort is now a package module (data_analysis_tools_mkTurk.spike_sorting.
# run_kilosort) using intra-package relative imports, so it must be launched with
# -m from the dir that CONTAINS data_analysis_tools_mkTurk/. Set PKG_PARENT to that
# dir (override NEURALWF_PKG_PARENT to relocate the checkout).
PKG_PARENT="${NEURALWF_PKG_PARENT:-/home/yy3658/helpers}"
echo "--- Running Kilosort4 (auto staging) for $monkey $date ---"
$PYTHON -u -c "import sys; sys.path.insert(0, '$PKG_PARENT'); from runpy import run_module; sys.argv=['run_kilosort','--monkey','$monkey','--date','$date']; run_module('data_analysis_tools_mkTurk.spike_sorting.run_kilosort', run_name='__main__')"
RC=$?
echo "--- [task $IDX] $SESS exit code: $RC ($(date '+%F %T')) ---"
exit $RC
