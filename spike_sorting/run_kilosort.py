"""Kilosort4 spike sorting pipeline for Neuropixels recordings.

Mirrors the structure of run_dartsort.py: a standalone run_kilosort(monkey, date)
that resolves paths, preprocesses, sorts with Kilosort4, and is safe to re-run
(skips already-complete sessions unless override=True).

Pipeline steps:
    1. Resolve the session and descend into the SpikeGLX run folder (skips _dk runs)
    2. Load SpikeGLX recording and preprocess (highpass, phase shift, bad channel
       removal, spatial filter, zscore) — IBL-style
    3. Stage the preprocessed recording to fast local disk for I/O
    4. Run Kilosort4 (optionally with a DREDge motion estimate from dartsort_output/)
    5. Clean up local staging

Saved files (in <session>/kilosort4/), written by Kilosort4:
    - spike_times.npy            : spike times in samples
    - spike_clusters.npy         : final cluster id per spike
    - spike_templates.npy        : template id per spike (Phy convention)
    - amplitudes.npy             : per-spike template scaling factor
    - templates.npy              : (n_templates, n_time, n_chan) waveform templates
    - similar_templates.npy      : template similarity matrix
    - channel_map.npy            : channel indices
    - channel_positions.npy      : channel geometry
    - whitening_mat.npy / _inv.npy : whitening matrices
    - pc_features.npy / pc_feature_ind.npy : PC features
    - cluster_KSLabel.tsv / cluster_ContamPct.tsv / cluster_Amplitude.tsv / cluster_group.tsv
    - params.py                  : Phy config
    - ops.npy                    : full Kilosort ops dict
    - kilosort4.log              : run log
"""

import time
import pickle
import logging

import numpy as np
import torch
import spikeinterface.full as si
import kilosort

from pathlib import Path

from ..utils_meta import init_dirs
from ..make_engram_path import BASE_DATA_PATH, BASE_SAVE_OUT_PATH
from .staging import (
    find_recording_dir,
    locate_bin,
    pick_nas_copy,
    choose_stage_mode,
    stage_recording,
    cleanup_staging,
    STAGE_LOCAL,
)
from .quality_metrics import run_quality_metrics, save_template_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

EXPECTED_OUTPUT_FILES = [
    "spike_times.npy",
    "spike_clusters.npy",
    "templates.npy",
    "amplitudes.npy",
    "channel_positions.npy",
    "whitening_mat_inv.npy",
    "params.py",
    "ops.npy",
]


def find_bin_dir(engram_rec_dir: Path, monkey: str) -> Path:
    """Return the recording dir holding the raw *ap.bin (engram, or a NAS copy).

    Back-compat: engram wins; otherwise pick a NAS copy at random immediately.
    New callers should use locate_bin (+ pick_nas_copy) so the random NAS pick is
    deferred until after the staging decision.
    """
    on_engram, nas_copies = locate_bin(engram_rec_dir, monkey)
    if on_engram:
        logger.info(f"Raw .bin found on engram: {engram_rec_dir}")
        return engram_rec_dir
    return pick_nas_copy(nas_copies)


def _resolve_session_paths(monkey: str, date: str) -> tuple[Path, Path, Path, Path]:
    """Resolve engram recording dir + save/plot/kilosort paths (NO bin location).

    Bin location is deliberately left out so callers can decide the staging mode
    (which depends only on engram-vs-NAS) BEFORE committing to a specific NAS copy.
    """
    data_path_list, save_out_path, plot_save_out_path = init_dirs(BASE_DATA_PATH, monkey, date, BASE_SAVE_OUT_PATH)

    if len(data_path_list) == 1:
        session_dir = Path(data_path_list[0])
        save_out_path = save_out_path[0]
        plot_save_out_path = plot_save_out_path[0]
    else:
        raise ValueError('Multiple or no data paths found for given monkey and date')

    engram_rec_dir = find_recording_dir(session_dir)

    # Kilosort results MUST be written to engram (the locker), never the NAS.
    ks_output_path = engram_rec_dir / "kilosort4"
    if not ks_output_path.resolve().is_relative_to(BASE_DATA_PATH.resolve()):
        raise RuntimeError(
            f"refusing to write kilosort output outside engram: {ks_output_path} "
            f"(BASE_DATA_PATH={BASE_DATA_PATH})"
        )
    ks_output_path.mkdir(exist_ok=True)
    return engram_rec_dir, save_out_path, plot_save_out_path, ks_output_path


def resolve_kilosort_path(monkey: str, date: str) -> tuple[Path, Path, Path, Path, Path]:
    """Resolve engram recording dir, raw-bin dir, save, plot, and kilosort paths.

    Returns (engram_rec_dir, bin_dir, save_out_path, plot_save_out_path,
    ks_output_path). engram_rec_dir anchors dartsort_output/ and the kilosort4/
    output; bin_dir is where the raw .ap.bin lives (engram or a randomly chosen
    NAS copy). run_kilosort itself does NOT use this — it defers the NAS pick
    until after the staging decision (see _resolve_session_paths + locate_bin).
    """
    engram_rec_dir, save_out_path, plot_save_out_path, ks_output_path = _resolve_session_paths(monkey, date)
    bin_dir = find_bin_dir(engram_rec_dir, monkey)
    return engram_rec_dir, bin_dir, save_out_path, plot_save_out_path, ks_output_path


def prep_kilosort(rec: si.BaseRecording) -> si.BaseRecording:
    """Apply IBL-style preprocessing (highpass, phase shift, bad-channel removal,
    spatial filter, zscore)."""
    rec = si.highpass_filter(rec)
    rec = si.phase_shift(rec)
    bad_channel_ids, channel_labels = si.detect_bad_channels(rec)
    logger.info(f"Removing {len(bad_channel_ids)} bad channels: {list(bad_channel_ids)}")
    rec = rec.remove_channels(bad_channel_ids)
    rec = si.highpass_spatial_filter(rec)
    rec = si.zscore(rec, num_chunks_per_segment=50, mode="mean+std")
    return rec


STAGE_NONE = "none"


def stage_recording_locally(rec: si.BaseRecording, rec_dir: Path) -> si.BaseRecording:
    """Back-compat wrapper: stage to /local (float16)."""
    return stage_recording(rec, STAGE_LOCAL, engram_rec_dir=rec_dir, session=rec_dir.name)


def cleanup_local_staging() -> None:
    """Back-compat: remove this job's /local staging dir."""
    cleanup_staging(STAGE_LOCAL, engram_rec_dir=Path("/nonexistent"))


def is_session_complete(output_path: Path) -> bool:
    """Check if all expected Kilosort output files exist for this session."""
    missing = [f for f in EXPECTED_OUTPUT_FILES if not (output_path / f).exists()]
    if set(missing) == set(EXPECTED_OUTPUT_FILES):
        logger.info("No output files found, session is not complete")
        return False
    elif missing:
        logger.info(f"Missing outputs: {', '.join(missing)}")
        return False
    return True


def load_motion_est(engram_rec_dir: Path):
    """Load the DREDge motion estimate produced by the dartsort pipeline.

    motion_est.pkl always lives on engram in <engram_rec_dir>/dartsort_output,
    even when the raw recording is staged from the NAS.
    """
    motion_path = engram_rec_dir / 'dartsort_output' / "motion_est.pkl"
    if not motion_path.exists():
        raise FileNotFoundError(
            f"dredge=True but motion estimate not found: {motion_path}. "
            "Run the dartsort pipeline first, or call with dredge=False."
        )
    with open(motion_path, "rb") as jar:
        motion_est = pickle.load(jar)
    logger.info(f"Loaded DREDge motion estimate: {motion_path}")
    return motion_est


def _nsavedchans_from_meta(bin_path) -> int:
    """Total channels saved on disk (incl. sync) from the SpikeGLX .ap.meta.

    The raw .ap.bin's byte layout has nSavedChans columns; read_spikeglx drops
    the sync channel so it can't supply this. The .meta sits beside the .bin.
    """
    bin_path = Path(bin_path)
    meta = bin_path.with_suffix(".meta")
    if not meta.exists():
        raise FileNotFoundError(f"no .ap.meta beside raw .bin: {meta}")
    for line in meta.read_text().splitlines():
        if line.startswith("nSavedChans="):
            return int(line.split("=", 1)[1].strip())
    raise ValueError(f"nSavedChans not found in {meta}")


def export_spike_times_per_unit(monkey: str, date: str, samp_rate: int = 30000,
                                overwrite: bool = False) -> None:
    """Prepare a session's Kilosort4 output for analyze_bystim(source='kilosort').

    From the raw KS4 output, builds the two inputs the kilosort PSTH analysis reads:
      - <engram_rec_dir>/kilosort4/spike_times_perunit/clu_{nnn}_st.npy : per-unit spike
        times (seconds), split from the flat spike_times.npy / spike_clusters.npy.
      - <save_out_path>/kilosort4/KSLabel.npy : per-template label array, used to
        enumerate units.

    overwrite=False skips sessions that already have spike_times_perunit populated.
    Run after run_kilosort has produced the kilosort4/ output for the session.
    """
    import pandas as pd

    # One session per (monkey, date). _resolve_session_paths raises on multiple/none
    # and descends into the SpikeGLX run folder (so kilosort4/ sits beside the .ap.meta).
    engram_rec_dir, save_out_path, _, _ = _resolve_session_paths(monkey, date)
    ks_data_dir = engram_rec_dir / 'kilosort4'
    ks_out_dir = Path(save_out_path) / 'kilosort4'
    perunit_dir = ks_data_dir / 'spike_times_perunit'

    if not ks_data_dir.exists():
        raise FileNotFoundError(f"no raw kilosort4 output at {ks_data_dir}; run run_kilosort first")

    already = sorted(perunit_dir.glob('clu_*_st.npy')) if perunit_dir.exists() else []
    if already and not overwrite:
        logger.info(f"{perunit_dir} already has {len(already)} unit files, "
                    f"skipping (overwrite=False)")
        return

    spike_times = np.load(ks_data_dir / 'spike_times.npy')
    spike_clusters = np.squeeze(np.load(ks_data_dir / 'spike_clusters.npy'))
    cluster_ids = np.unique(spike_clusters)

    # Build KSLabel: one entry per template index, filled from cluster_KSLabel.tsv.
    cluster_KSLabel = pd.read_csv(ks_data_dir / 'cluster_KSLabel.tsv', sep='\t')
    n_templates = int(cluster_ids.max()) + 1
    KSLabel = np.full(n_templates, '', dtype=object)
    KSLabel[cluster_ids] = np.squeeze(np.array(
        [cluster_KSLabel[cluster_KSLabel['cluster_id'] == c]['KSLabel'].values for c in cluster_ids]
    ))
    ks_out_dir.mkdir(parents=True, exist_ok=True)
    np.save(ks_out_dir / 'KSLabel.npy', KSLabel)
    logger.info(f"wrote {ks_out_dir / 'KSLabel.npy'} ({len(KSLabel)} units)")

    # Split spike times per unit (seconds).
    perunit_dir.mkdir(parents=True, exist_ok=True)
    for i in range(len(KSLabel)):
        np.save(perunit_dir / 'clu_{:0>3d}_st.npy'.format(i),
                spike_times[spike_clusters == i] / samp_rate)
    logger.info(f"wrote per-unit spike times to {perunit_dir}")


def run_kilosort(monkey: str, date: str, dredge: bool = True, override: bool = False,
                 stage_local: bool = True) -> None:
    """Run the full Kilosort4 pipeline: preprocess, stage, sort, clean up.

    Standalone — call with just monkey and date. Skips sessions that are already
    complete unless override=True. With dredge=True (default) it feeds the DREDge
    motion estimate from <session>/dartsort_output/motion_est.pkl.

    stage_local (default True): EVERY session is staged + preprocessed (SI
    cascade + DREDge); the destination is chosen per session by choose_stage_mode:
      * raw .bin on engram          -> stage to engram (float32)
      * raw .bin NAS-only, /local free -> stage to /local (float16)
      * raw .bin NAS-only, /local busy -> stage to engram (float32)
    The staged temp copy is deleted after sorting; results write to kilosort4/.
    Set False (--no-stage) for a BENCHMARK-only mode: Kilosort reads the raw *ap.bin
    in place with NO SI preprocessing and DREDge disabled (not quality-equivalent),
    overwriting the same kilosort4/ dir.
    """
    logger.info(f"Starting kilosort pipeline for {monkey} {date} "
                f"(stage_local={stage_local}, dredge={dredge})")
    t_start = time.perf_counter()

    if not torch.cuda.is_available():
        logger.warning("CUDA not available — Kilosort4 will be very slow or fail")

    # Resolve engram/output paths first, then locate the bin WITHOUT committing to
    # a NAS copy — the staging decision only needs engram-vs-NAS.
    engram_rec_dir, save_out_path, plot_save_out_path, ks_output_path = _resolve_session_paths(monkey, date)
    out_dir = ks_output_path

    # Completeness check FIRST, before any staging decision. A done session must
    # not claim the /local slot (or read the NAS) just to bail — both staged and
    # no-stage runs write to this same kilosort4/ dir, so the check is mode-agnostic.
    if is_session_complete(out_dir) and not override:
        logger.info(f"Session already complete, skipping {monkey} {date}")
        return

    on_engram, nas_copies = locate_bin(engram_rec_dir, monkey)
    logger.info(f"Resolved paths: engram_rec={engram_rec_dir}, "
                f"bin_on_engram={on_engram}, nas_copies={len(nas_copies)}, "
                f"ks_output={ks_output_path}")

    # Decide staging destination. Every session is staged + preprocessed (SI +
    # DREDge); only the destination varies by bin location and /local contention
    # (see choose_stage_mode). --no-stage (stage_local=False) is a benchmark-only
    # hard override that instead reads the raw .bin in place (no prep, no DREDge).
    # STAGE_LOCAL means the /local slot is now CLAIMED, so everything after this
    # must release it via the finally below.
    if not stage_local:
        stage_mode = STAGE_NONE
        logger.info("--no-stage: forcing STAGE_NONE (read raw .bin in place, no prep/DREDge)")
    else:
        stage_mode = choose_stage_mode(on_engram)
    staged = stage_mode != STAGE_NONE

    # From here on a claimed /local slot (or any staged dir) must be released, so
    # EVERYTHING below runs under the try/finally that calls cleanup_staging —
    # including bin selection, read, prep and motion load, any of which can raise.
    bin_dir = None  # so the finally can clean up even if pick_nas_copy never ran
    try:
        # Commit to a concrete bin dir. The random NAS-copy pick (load balancing
        # across servers) happens HERE, after the staging decision (deferred pick).
        if on_engram:
            bin_dir = engram_rec_dir
        else:
            bin_dir = pick_nas_copy(nas_copies)

        # DREDge only applies to a staged, preprocessed recording; the --no-stage
        # raw read has no matching motion estimate, so disable it there.
        if not staged and dredge:
            logger.info("--no-stage: forcing dredge=False (motion est does not match raw .bin)")
            dredge = False

        # Load + preprocess (raw .bin read from bin_dir, which may be engram or NAS)
        t0 = time.perf_counter()
        rec = si.read_spikeglx(bin_dir, stream_id="imec0.ap")
        logger.info(str(rec))
        logger.info(f"[timing] read_spikeglx done @ {time.strftime('%F %T')} "
                    f"(+{time.perf_counter() - t0:.1f}s)")

        if staged:
            # Preprocessing is baked into the staged copy; skip it when not staging
            # (Kilosort applies its own internal filtering off the raw .bin instead).
            t0 = time.perf_counter()
            rec = prep_kilosort(rec)
            logger.info(f"[timing] prep_kilosort done @ {time.strftime('%F %T')} "
                        f"(+{time.perf_counter() - t0:.1f}s)")
        else:
            logger.info("no staging: skipping prep_kilosort; Kilosort reads RAW .bin")

        # Optional DREDge motion estimate from engram (load before staging so we fail fast)
        motion_est = load_motion_est(engram_rec_dir) if dredge else None

        if not staged:
            out_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"no staging: results -> {out_dir}")

        if staged:
            t0 = time.perf_counter()
            rec = stage_recording(rec, stage_mode, engram_rec_dir, engram_rec_dir.name)
            logger.info(f"[timing] staging ({stage_mode}) done @ {time.strftime('%F %T')} "
                        f"(+{time.perf_counter() - t0:.1f}s)")
            filename = rec._recording_segments[0].file_path
            data_dtype = rec.dtype
            n_chan_bin = rec.get_num_channels()
        else:
            # Point Kilosort straight at the raw *ap.bin where it lives.
            raw_bins = sorted(Path(bin_dir).glob("*ap.bin"))
            if not raw_bins:
                raise FileNotFoundError(f"no *ap.bin in {bin_dir}")
            filename = str(raw_bins[0])
            data_dtype = rec.dtype
            # The RAW file holds ALL saved channels (384 AP + sync), whereas
            # read_spikeglx drops the sync channel -> get_num_channels() is 384.
            # Kilosort byte-counts the file against n_chan_bin, so it must be the
            # on-disk total (nSavedChans from the .ap.meta), not 384, or it errors
            # "Bytes in binary file did not divide evenly". chanMap below still
            # selects only the 384 real channels, leaving sync out of the sort.
            n_chan_bin = _nsavedchans_from_meta(filename)
            logger.info(f"no staging ({stage_mode}): feeding Kilosort raw file {filename} "
                        f"(dtype={data_dtype}, n_chan_bin={n_chan_bin} incl. sync, "
                        f"no staged copy)")

        geom = rec.get_channel_locations()
        probe = dict(
            chanMap=np.arange(len(geom)),
            xc=geom[:, 0],
            yc=geom[:, 1],
            n_chan=len(geom),
            kcoords=np.zeros(len(geom), dtype=int),
        )
        settings = {
            'fs': rec.sampling_frequency,
            'n_chan_bin': n_chan_bin,
            'nblocks': 5,
        }

        logger.info(f"Running Kilosort4 (dredge={'on' if dredge else 'off'}, "
                    f"stage_mode={stage_mode}) -> {out_dir}")
        t0 = time.perf_counter()
        kilosort.run_kilosort(
            settings=settings,
            probe=probe,
            filename=filename,
            data_dtype=data_dtype,
            results_dir=out_dir,
            dredge_motion_est=motion_est,
        )
        logger.info(f"[timing] Kilosort4 sorting done @ {time.strftime('%F %T')} "
                    f"(+{time.perf_counter() - t0:.1f}s)")
    finally:
        if staged:
            cleanup_staging(stage_mode, engram_rec_dir, engram_rec_dir.name)
            logger.info("Staging cleanup completed")

    logger.info(f"[timing] TOTAL pipeline @ {time.strftime('%F %T')} "
                f"(+{time.perf_counter() - t_start:.1f}s = "
                f"{(time.perf_counter() - t_start) / 60:.2f} min)")

    if is_session_complete(out_dir):
        logger.info(f"Session successfully completed: {monkey} {date}")
        # Export per-unit spike times + KSLabel for analyze_bystim(source='kilosort').
        # Done after staging cleanup, only on a complete sort.
        export_spike_times_per_unit(monkey, date, overwrite=override)
        # Save unit quality metrics (presence_ratios / amplitude_cutoffs / viol_rates)
        # for good-single-unit selection (see quality_metrics.is_good_unit).
        run_quality_metrics(monkey, date, overwrite=override)
        # Persist fr / contamPct / template_depths / temp_chan_amps (standard Phy/KS
        # post-processing arrays KS4 doesn't emit) into the save-out kilosort4/ folder.
        save_template_metrics(monkey, date, overwrite=override)
    else:
        logger.warning(f"Session completed but some output files are missing: {monkey} {date}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run Kilosort4 spike sorting pipeline')
    parser.add_argument('--monkey', type=str, required=True, help='Monkey name')
    parser.add_argument('--date', type=str, required=True, help='Recording date (YYYYMMDD)')
    parser.add_argument('--override', action='store_true', help='Override existing outputs and rerun')
    parser.add_argument('--no-stage', dest='stage_local', action='store_false',
                        help='Sort directly off the raw .ap.bin with no /local staging '
                             '(benchmark; writes to kilosort4/, skips SI preprocessing)')
    args = parser.parse_args()

    run_kilosort(args.monkey, args.date, override=args.override, stage_local=args.stage_local)
