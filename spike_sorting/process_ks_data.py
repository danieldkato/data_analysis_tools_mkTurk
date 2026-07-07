import argparse
import logging
import os
import time
from pathlib import Path

try:
    from ..analyze_bystim import analyze_bystim_all, kilosort_psth_complete
    from ..utils_meta import init_dirs
    from ..make_engram_path import ENGRAM_PATH, BASE_DATA_PATH, BASE_SAVE_OUT_PATH
    from ..IO import ch_dicts_2_h5
    from .quality_metrics import run_quality_metrics, save_template_metrics
except ImportError:
    from data_analysis_tools_mkTurk.analyze_bystim import analyze_bystim_all, kilosort_psth_complete
    from data_analysis_tools_mkTurk.utils_meta import init_dirs
    from data_analysis_tools_mkTurk.make_engram_path import ENGRAM_PATH, BASE_DATA_PATH, BASE_SAVE_OUT_PATH
    from data_analysis_tools_mkTurk.IO import ch_dicts_2_h5
    from data_analysis_tools_mkTurk.spike_sorting.quality_metrics import run_quality_metrics, save_template_metrics


# Default root for the single-unit session HDF5s. Explicit, shared top-level
# location on the locker (NOT under any user's personal tree). Each session lands
# at <DEFAULT_OUTPUT_BASE>/<monkey>/<recording_dir>/ks/<date>.h5. Override per-call
# with the output_base argument / --output-base.
DEFAULT_OUTPUT_BASE = str(ENGRAM_PATH / 'processed_h5')


def process_ks_data(monkey: str, date: str, n_jobs: int = -1, force: bool = False,
                    output_base: str = DEFAULT_OUTPUT_BASE):
    """
    Process a Kilosort4-sorted session into a single-unit session HDF5.

    Single-unit analogue of process_session_data (process_session_pipeline.py):
    assumes Kilosort4 has already produced its raw output for the session and
    orchestrates the per-unit analysis from there through the combined HDF5.

    Args:
        monkey (str): Identifier for the monkey subject.
        date (str): Date of the experimental session in string format.
        n_jobs (int, optional): Parallel workers for the per-unit by_stim step
            (-1 = all cores). Defaults to -1.
        force (bool, optional): Recompute the by_stim PSTHs / metrics even if they
            already exist. Defaults to False.
        output_base (str, optional): Root directory for the session HDF5. The file
            is written to <output_base>/<monkey>/<recording_dir>/ks/<date>.h5.
            Defaults to DEFAULT_OUTPUT_BASE (<ENGRAM_PATH>/processed_h5, e.g.
            /mnt/smb/locker/issa-locker/processed_h5).

    Returns:
        str: Path to the written session HDF5.

    Notes:
        - Requires the raw Kilosort4 output (spike_times/clusters, templates,
          KSLabel.npy, the cluster_*.tsv files) to already exist for the session.
        - The pipeline consists of three stages:
            1. Per-unit stimulus-based PSTHs across all sorted units
               (analyze_bystim_all with source='kilosort'), skipped when already
               complete unless force=True.
            2. Per-unit quality and template metrics (run_quality_metrics,
               save_template_metrics), which populate the unit_quality /
               unit_spatial tables written into the HDF5.
            3. Combine the per-unit PSTHs into a single session HDF5
               (ch_dicts_2_h5 with source='ks').

    Example:
        >>> process_ks_data("Bourgeois", "20241025", n_jobs=-1)
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logger = logging.getLogger(__name__)

    logger.info(f"Starting kilosort session processing for monkey={monkey}, date={date}, n_jobs={n_jobs}")
    total_start = time.time()

    # Resolve session paths (one recording per monkey/date).
    data_path_list, save_out_path_list, _ = init_dirs(BASE_DATA_PATH, monkey, date, BASE_SAVE_OUT_PATH)
    if len(save_out_path_list) != 1:
        raise ValueError(f"expected exactly one recording for {monkey} {date}, found {len(save_out_path_list)}")
    preprocessed_data_path = str(save_out_path_list[0])
    recording_dir = Path(preprocessed_data_path).name

    output_directory = os.path.join(output_base, monkey, recording_dir, 'ks')

    # Stage 1: Per-unit stimulus-based PSTHs (analyze_bystim, source='kilosort')
    logger.info("Stage 1/3: Starting per-unit by_stim PSTHs...")
    stage1_start = time.time()
    if not force and kilosort_psth_complete(monkey, date):
        logger.info("  by_stim already complete; skipping (use force=True to recompute)")
    else:
        analyze_bystim_all(monkey, date, source='kilosort', n_jobs=n_jobs)
    logger.info(f"Stage 1/3: by_stim PSTHs completed in {time.time() - stage1_start:.1f}s")

    # Stage 2: Per-unit quality and template metrics
    logger.info("Stage 2/3: Starting per-unit quality and template metrics...")
    stage2_start = time.time()
    run_quality_metrics(monkey, date, overwrite=force)
    save_template_metrics(monkey, date, overwrite=force)
    logger.info(f"Stage 2/3: quality and template metrics completed in {time.time() - stage2_start:.1f}s")

    # Stage 3: Combine per-unit PSTHs into a single session HDF5 (source='ks')
    logger.info("Stage 3/3: Writing combined single-unit HDF5...")
    stage3_start = time.time()
    ch_dicts_2_h5(
        BASE_DATA_PATH, monkey, date,
        preprocessed_data_path=preprocessed_data_path,
        channels=None,
        save_output=True,
        fname=date,
        output_directory=output_directory,
        source='ks',
    )
    h5_path = os.path.join(output_directory, date + '.h5')
    logger.info(f"Stage 3/3: HDF5 written to {h5_path} in {time.time() - stage3_start:.1f}s")

    total_elapsed = time.time() - total_start
    logger.info(f"Kilosort processing for monkey {monkey} on date {date} completed in {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")

    return h5_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process a Kilosort4-sorted session into a single-unit session HDF5.')
    parser.add_argument('--monkey', type=str, required=True, help='Identifier for the monkey subject')
    parser.add_argument('--date', type=str, required=True, help='Date of the experimental session')
    parser.add_argument('--n-jobs', type=int, default=-1, help='Parallel workers for the by_stim step (-1 = all cores)')
    parser.add_argument('--force', action='store_true', help='Recompute by_stim PSTHs / metrics even if already present')
    parser.add_argument('--output-base', type=str, default=DEFAULT_OUTPUT_BASE,
                        help=f'Root dir for the session HDF5 (default: {DEFAULT_OUTPUT_BASE})')
    args = parser.parse_args()
    process_ks_data(args.monkey, args.date, n_jobs=args.n_jobs, force=args.force, output_base=args.output_base)
