"""Shared staging strategy for the spike-sorting pipelines (run_dartsort, run_kilosort).

Both pipelines preprocess a SpikeGLX recording and stage a single preprocessed copy
to fast disk before sorting. The destination and lock are SHARED across pipelines so
a node never holds more than one staged recording at a time, regardless of which
pipeline staged it.

Strategy:
  * The raw *ap.bin is the same file whether it lives on engram (the locker) or a
    mounted NAS. Engram is checked first; on a miss the session subpath is mapped
    onto each NAS mount and one copy is picked at random (load balancing).
  * Every session is staged + preprocessed. choose_stage_mode picks the destination:
      - raw .bin on engram             -> engram (float32)
      - raw .bin NAS-only, /local free -> /local (float16)  [claims the slot]
      - raw .bin NAS-only, /local busy -> engram (float32)
  * /local holds at most ONE staged recording at a time. The fixed LOCAL_STAGING_DIR
    is the lock: a task claims it with mkdir(exist_ok=False); a stale orphan (no
    writes within LOCAL_STALE_S) is reclaimed.
  * Staged copies are session-named, so a complete copy left by a prior/parallel run
    of the same session is reused instead of re-staged (staged_copy_is_valid).
"""

import os
import time
import random
import shutil
import logging
from pathlib import Path

import spikeinterface.full as si

from ..make_engram_path import BASE_DATA_PATH

logger = logging.getLogger(__name__)

LOCAL_STAGING_ROOT = Path("/local")

# A node stages at most ONE recording on /local at a time, under this single fixed
# dir SHARED by every pipeline (NOT job-id- or pipeline-namespaced). The dir itself
# is the lock: a task claims /local by atomically creating it (mkdir exist_ok=False);
# if it already exists, /local is busy and the task falls back to engram. A dir left
# behind by a SIGKILL'd job is reclaimed once it goes stale (no writes within
# LOCAL_STALE_S).
LOCAL_STAGING_DIR = LOCAL_STAGING_ROOT / "neuralwf_staging"
LOCAL_STALE_S = float(os.environ.get("NEURALWF_LOCAL_STALE_S", str(12 * 3600)))  # 12 h

# Staging chunk size for rec.save() (seconds). Override via NEURALWF_STAGE_CHUNK_S.
STAGE_CHUNK_S = float(os.environ.get("NEURALWF_STAGE_CHUNK_S", "5"))

# Engram is authoritative for the session tree (BASE_DATA_PATH/<monkey>/<session>/...).
# The raw .bin may live on engram OR a mounted NAS (same file). NAS layout has NO
# monkey level: <mount>/<session>/<run>_g0/<run>_g0_imec0/. Set NEURALWF_NAS_ROOT to
# the mounted_nas root; a session may live under any per-share mount, so search all.
NAS_ROOT = (
    Path(os.environ["NEURALWF_NAS_ROOT"]).expanduser()
    if os.environ.get("NEURALWF_NAS_ROOT")
    else Path("~/mounted_nas").expanduser()
)

STAGE_LOCAL = "local"
STAGE_ENGRAM = "engram"


def detect_compute_resources() -> tuple[int, int]:
    """Detect available CPUs and GPUs, respecting SLURM allocations."""
    import torch
    n_cpus = int(os.environ.get('SLURM_CPUS_PER_TASK', os.cpu_count() or 1))
    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    logger.info(f"Detected {n_cpus} CPUs, {n_gpus} GPUs")
    return n_cpus, n_gpus


def nas_mounts(monkey: str | None = None) -> list[Path]:
    """Per-share NAS mount dirs (<mount>/<session>/... lives directly under each)."""
    pattern = f"{monkey}/*" if monkey else "*/*"
    mounts = sorted(p for p in NAS_ROOT.glob(pattern) if p.is_dir())
    return mounts if mounts else [NAS_ROOT]


def find_recording_dir(session_dir: Path, include_dk: bool = False) -> Path:
    """Descend into the SpikeGLX run folder for this session.

    SpikeGLX nests data as <session>/<run>_g0/<run>_g0_imec0/. We locate the run by
    its *.ap.meta sidecar (always present on engram). "_dk" runs are excluded by
    default. Raises if the choice is ambiguous after filtering.
    """
    if list(session_dir.glob("*ap.meta")):
        return session_dir

    candidates = []
    for sub in sorted(session_dir.glob("*")):
        if sub.is_dir() and list(sub.glob("*ap.meta")):
            candidates.append(sub)
    for sub in sorted(session_dir.glob("*/*")):
        if sub.is_dir() and list(sub.glob("*ap.meta")):
            candidates.append(sub)

    if not candidates:
        raise FileNotFoundError(f"no *ap.meta found under {session_dir}")

    logger.info(f"Recording-dir candidates ({len(candidates)}): {[c.name for c in candidates]}")

    if not include_dk:
        non_dk = [c for c in candidates
                  if "_dk_" not in c.name and not c.name.endswith("_dk")
                  and "_dk_g" not in c.parent.name]
        if non_dk:
            candidates = non_dk
        else:
            logger.warning("Only _dk runs found; proceeding with them")

    if len(candidates) > 1:
        raise ValueError(
            "Multiple candidate runs after filtering — disambiguate the session:\n  "
            + "\n  ".join(str(c) for c in candidates)
        )
    logger.info(f"Selected recording dir: {candidates[0]}")
    return candidates[0]


def locate_bin(engram_rec_dir: Path, monkey: str) -> tuple[bool, list[Path]]:
    """Find where the raw *ap.bin lives, WITHOUT choosing among NAS copies.

    Returns (on_engram, nas_copies). Engram is checked first; on a miss the session-
    relative subpath is mapped onto each NAS mount (which drops the monkey level).
    Raises if neither engram nor any NAS mount has it.
    """
    if list(engram_rec_dir.glob("*ap.bin")):
        return True, []

    try:
        rel = engram_rec_dir.relative_to(BASE_DATA_PATH / monkey)
    except ValueError:
        rel = engram_rec_dir.relative_to(BASE_DATA_PATH)

    tried, found = [], []
    for mount in nas_mounts(monkey):
        nas_rec_dir = mount / rel
        tried.append(nas_rec_dir)
        if list(nas_rec_dir.glob("*ap.bin")):
            found.append(nas_rec_dir)

    if found:
        return False, found

    tried_str = "\n  ".join(str(p) for p in tried) or "(no NAS mounts found)"
    raise FileNotFoundError(
        f"no *ap.bin found on engram ({engram_rec_dir}) or any NAS mount under "
        f"{NAS_ROOT}. Tried:\n  {tried_str}"
    )


def pick_nas_copy(nas_copies: list[Path]) -> Path:
    """Pick one NAS copy at RANDOM among identical duplicates (load balancing)."""
    nas_rec_dir = random.choice(nas_copies)
    logger.info(f"Raw .bin found on NAS ({len(nas_copies)} cop"
                f"{'ies' if len(nas_copies) > 1 else 'y'}, using {nas_rec_dir})")
    if not list(nas_rec_dir.glob("*ap.meta")):
        logger.warning(
            f"NAS dir has .bin but no .ap.meta ({nas_rec_dir}); "
            "read_spikeglx needs the meta co-located with the bin"
        )
    return nas_rec_dir


def staged_copy_is_valid(stage_dir: Path) -> bool:
    """True if stage_dir holds a complete, loadable SpikeInterface binary cache.

    Reuse a staged dir only if read_binary_folder can open it (binary.json + the
    .bin memmaps), so a half-written cache from a SIGKILL'd job is not mistaken for
    a finished one. Cheap metadata read; does not read the samples.
    """
    if not stage_dir.exists():
        return False
    try:
        si.read_binary_folder(stage_dir)
        return True
    except Exception as e:
        logger.warning(f"staged copy at {stage_dir} is not loadable ({e}); will re-stage")
        return False


def _local_staging_stale() -> bool:
    """True if the fixed /local staging dir exists but is abandoned (orphaned by a
    SIGKILL): nothing under it has been written within LOCAL_STALE_S."""
    try:
        newest = max((p.stat().st_mtime for p in LOCAL_STAGING_DIR.rglob("*") if p.is_file()),
                     default=LOCAL_STAGING_DIR.stat().st_mtime)
    except OSError:
        return False
    return (time.time() - newest) > LOCAL_STALE_S


def _staging_manifest_path() -> Path:
    """File listing this job's staged dir(s), read by a SLURM kill-trap so a
    SIGKILL'd task still gets its staged copy swept. Override with
    NEURALWF_STAGING_MANIFEST."""
    job_id = os.environ.get('SLURM_JOB_ID', os.getpid())
    # In practice the SLURM scripts always set NEURALWF_STAGING_MANIFEST; this
    # fallback is only used when running outside SLURM. __file__ is now
    # <repo>/spike_sorting/staging.py, so three parents up reaches <repo>'s parent.
    default = Path(__file__).resolve().parents[2] / "jobs" / "logs" / f"staging_{job_id}.manifest"
    return Path(os.environ.get("NEURALWF_STAGING_MANIFEST", str(default)))


def _record_staging_path(stage_dir: Path) -> None:
    """Append a staged dir to this job's manifest (best-effort)."""
    try:
        mf = _staging_manifest_path()
        mf.parent.mkdir(parents=True, exist_ok=True)
        with open(mf, "a") as f:
            f.write(str(stage_dir) + "\n")
    except OSError as e:
        logger.warning(f"could not record staging path for kill-cleanup: {e}")


def claim_local_staging() -> bool:
    """Atomically claim the node's single /local staging slot.

    The fixed dir IS the lock: mkdir(exist_ok=False) succeeds for exactly one task.
    Returns True if we now own /local staging. A stale orphan (no writes in
    LOCAL_STALE_S) is reclaimed and re-claimed.
    """
    try:
        LOCAL_STAGING_DIR.mkdir(parents=True, exist_ok=False)
        _record_staging_path(LOCAL_STAGING_DIR)
        return True
    except FileExistsError:
        if _local_staging_stale():
            logger.warning(f"reclaiming stale /local staging dir {LOCAL_STAGING_DIR} "
                           f"(no writes in {LOCAL_STALE_S/3600:.1f} h)")
            shutil.rmtree(LOCAL_STAGING_DIR, ignore_errors=True)
            try:
                LOCAL_STAGING_DIR.mkdir(parents=True, exist_ok=False)
                _record_staging_path(LOCAL_STAGING_DIR)
                return True
            except FileExistsError:
                return False
        return False
    except OSError as e:
        logger.warning(f"could not claim /local staging ({e}); treating as busy")
        return False


def choose_stage_mode(on_engram: bool) -> str:
    """Decide where to stage the preprocessed copy for this session.

      * raw .bin on engram             -> STAGE_ENGRAM (float32)
      * raw .bin NAS-only, /local free -> STAGE_LOCAL (float16)  [claims the slot]
      * raw .bin NAS-only, /local busy -> STAGE_ENGRAM (float32)

    STAGE_LOCAL is returned only when this task atomically claims the single /local
    slot, so the slot is already held on return.
    """
    if on_engram:
        logger.info("stage decision: raw .bin on engram -> stage to engram (float32)")
        return STAGE_ENGRAM
    if claim_local_staging():
        logger.info("stage decision: NAS-only, claimed /local -> stage to /local (float16)")
        return STAGE_LOCAL
    logger.info("stage decision: NAS-only, /local busy -> stage to engram (float32)")
    return STAGE_ENGRAM


def stage_recording(rec: si.BaseRecording, mode: str, engram_rec_dir: Path,
                    session: str) -> si.BaseRecording:
    """Save the preprocessed recording for fast I/O during sorting.

    mode=STAGE_LOCAL  -> /local, float16 (size-limited node scratch)
    mode=STAGE_ENGRAM -> a temp dir in the session's engram folder, float32

    The staged dir is named per session so a leftover copy can be matched to the
    session it belongs to (and reused on a re-run of the same session, even one
    staged by the other pipeline).
    """
    n_cpus, _ = detect_compute_resources()
    job_id = os.environ.get('SLURM_JOB_ID', os.getpid())
    if mode == STAGE_ENGRAM:
        # Session/job-id-stamped temp dir in the session's engram folder, float32.
        stage_dir = engram_rec_dir / f"rec_ppx.staging.{session}.{job_id}"
        dtype = "float32"
        root_hint = engram_rec_dir
        _record_staging_path(stage_dir)
    else:
        # The single /local slot was already claimed (and recorded) by
        # choose_stage_mode/claim_local_staging; the binary goes in a
        # <session>/rec_ppx/ subdir so the copy is tied to a session.
        stage_dir = LOCAL_STAGING_DIR / session / "rec_ppx"
        dtype = "float16"
        root_hint = LOCAL_STAGING_ROOT

    # Reuse an existing, fully-staged copy for this session instead of re-staging.
    if staged_copy_is_valid(stage_dir):
        logger.info(f"Loading existing staged cache: {stage_dir}")
        return si.read_binary_folder(stage_dir)
    if stage_dir.exists():
        # Present but incomplete (e.g. SIGKILL mid-write) — clear before re-staging.
        logger.warning(f"clearing incomplete staged dir before re-staging: {stage_dir}")
        shutil.rmtree(stage_dir, ignore_errors=True)

    try:
        stage_dir.parent.mkdir(parents=True, exist_ok=True)
        logger.info(
            f"Staging preprocessed recording to {stage_dir} "
            f"(mode={mode}, n_jobs={n_cpus}, chunk={STAGE_CHUNK_S}s, dtype={dtype})"
        )
        # /local stages float16: the zscored recording is float-valued, and float32
        # doubles on-disk size (a ~4h recording is ~604 GiB float32 vs ~302 GiB
        # float16), overflowing /local. float16 is plenty (Kilosort whitens
        # internally; dartsort detects on the same scale). engram uses float32.
        rec = rec.save(folder=stage_dir, n_jobs=n_cpus, chunk_duration=f"{STAGE_CHUNK_S}s", dtype=dtype)
        logger.info(f"Staging complete: {stage_dir}")
    except OSError as e:
        shutil.rmtree(stage_dir, ignore_errors=True)
        raise RuntimeError(
            f"Staging failed ({e}) on {root_hint}; free up space "
            f"(a ~4h recording needs ~302 GiB float16 / ~604 GiB float32)."
        ) from e
    return rec


def cleanup_staging(mode: str, engram_rec_dir: Path, session: str | None = None) -> None:
    """Remove this job's staged copy and release the /local slot.

    STAGE_LOCAL removes the fixed LOCAL_STAGING_DIR (releasing the node's single
    staging slot). STAGE_ENGRAM removes the session/job-id-stamped temp dir in the
    engram session folder. Never touches pre-existing data. Clears the kill-cleanup
    manifest so a wrapper trap has nothing left to sweep.
    """
    job_id = os.environ.get('SLURM_JOB_ID', os.getpid())
    if mode == STAGE_ENGRAM:
        sess = session if session is not None else engram_rec_dir.name
        stage_dir = engram_rec_dir / f"rec_ppx.staging.{sess}.{job_id}"
    elif mode == STAGE_LOCAL:
        stage_dir = LOCAL_STAGING_DIR
    else:
        return
    if stage_dir.exists():
        shutil.rmtree(stage_dir, ignore_errors=True)
        logger.info(f"Cleaned up staging (released {mode} slot): {stage_dir}")
    try:
        _staging_manifest_path().unlink(missing_ok=True)
    except OSError:
        pass
