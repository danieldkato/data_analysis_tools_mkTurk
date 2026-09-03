import argparse
import logging
import os
import time
from pathlib import Path
import h5py
import numpy as np

try:
    from ..analyze_bystim import analyze_bystim_all, kilosort_psth_complete
    from ..utils_meta import init_dirs, resolve_ks_h5_path
    from ..make_engram_path import ENGRAM_PATH, BASE_DATA_PATH, BASE_SAVE_OUT_PATH
    from ..IO import ch_dicts_2_h5
    from .quality_metrics import run_quality_metrics, save_template_metrics
except ImportError:
    from data_analysis_tools_mkTurk.analyze_bystim import analyze_bystim_all, kilosort_psth_complete
    from data_analysis_tools_mkTurk.utils_meta import init_dirs, resolve_ks_h5_path
    from data_analysis_tools_mkTurk.make_engram_path import ENGRAM_PATH, BASE_DATA_PATH, BASE_SAVE_OUT_PATH
    from data_analysis_tools_mkTurk.IO import ch_dicts_2_h5
    from data_analysis_tools_mkTurk.spike_sorting.quality_metrics import run_quality_metrics, save_template_metrics


# Default root for the single-unit session HDF5s. Explicit, shared top-level
# location on the locker (NOT under any user's personal tree). Each session lands
# at <DEFAULT_OUTPUT_BASE>/<monkey>/<recording_dir>/ks/<date>.h5. Override per-call
# with the output_base argument / --output-base.
DEFAULT_OUTPUT_BASE = str(ENGRAM_PATH / 'processed_h5')


# On-disk storage settings for the session HDF5. Not exposed as parameters: these
# are properties of the file format we commit to, not per-call choices.
#   - int32: spike counts are small integers, so a wider type doubles the file for
#     nothing. NaN pads become a large negative sentinel, which the read path
#     (h5_2_dat_array_rsvp, h5_2_df) converts back to NaN.
#   - lzf: the slab is mostly NaN pad and small counts, so it compresses heavily.
#   - BIN_CHUNK_SIZE: chunking the time axis as well as the trial axis is what lets
#     a time_window restricted read touch fewer bytes on disk.
#   - 'fixed': reads back far faster than 'table' over a network filesystem, and is
#     safe because nothing queries trial_params partially.
SPIKE_DTYPE = np.int32
COMPRESS_DATA = True
CHUNK_SIZE = 20
BIN_CHUNK_SIZE = 50
TRIAL_PARAMS_FORMAT = 'fixed'


# Keys every session HDF5 must hold. ch_dicts_2_h5() truncates the output file
# (mode 'w') before writing, then appends the pandas tables only after closing the
# h5py handle, so a run that dies partway leaves a file holding `data` but missing
# the later tables -- present on disk, but unusable in h5_2_trial_df()/h5_2_ch_meta().
REQUIRED_H5_KEYS = ['data', 'stim_indices', 'trial_params', 'trial_params_short',
                    'zero_coordinates', 'imro_table']

# Per-unit metrics tables, written only for source='ks'. Absent when
# build_unit_info_dfs() failed: ch_dicts_2_h5 catches that, warns, and substitutes an
# empty DataFrame, which to_hdf writes as no key at all. The HDF5 is otherwise
# complete and usable, so these are reported but not treated as incomplete.
KS_METRICS_H5_KEYS = ['unit_quality', 'unit_spatial']


def h5_is_complete(h5_path):
    """
    Check whether a session HDF5 holds every dataset/table a reader expects.

    Args:
        h5_path (str): Path to the session HDF5.

    Returns:
        (bool, list): Whether every required key is present, and the names of any
            missing keys -- required ones first, then absent ks metrics tables,
            which are reported but do not affect the bool.
    """
    if not os.path.exists(h5_path):
        return False, ['<file does not exist>']
    try:
        with h5py.File(h5_path, 'r') as f:
            keys = set(f.keys())
    except Exception as e:
        return False, ['<unreadable: {}>'.format(e)]
    missing_required = [k for k in REQUIRED_H5_KEYS if k not in keys]
    missing_metrics = [k for k in KS_METRICS_H5_KEYS if k not in keys]
    return not missing_required, missing_required + missing_metrics


def process_ks_data(monkey: str, date: str, n_jobs: int = -1, force: bool = False,
                    output_base: str = DEFAULT_OUTPUT_BASE, suffix: str = ''):
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
        suffix (str, optional): Appended to the HDF5 filename stem, so the file
            lands at <date><suffix>.h5. Lets a regenerated session sit alongside
            the existing one instead of overwriting it, e.g. suffix='_int32' to
            compare old and new before committing to a bulk rewrite. Defaults to
            '', which writes <date>.h5 and overwrites any file already there.

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
    _, save_out_path_list, _ = init_dirs(BASE_DATA_PATH, monkey, date, BASE_SAVE_OUT_PATH)
    if len(save_out_path_list) != 1:
        raise ValueError(f"expected exactly one recording for {monkey} {date}, found {len(save_out_path_list)}")
    preprocessed_data_path = str(save_out_path_list[0])

    output_directory = str(resolve_ks_h5_path(monkey, date, output_base=output_base).parent)
    fname = date + suffix

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
        fname=fname,
        output_directory=output_directory,
        source='ks',
        chunk_size=CHUNK_SIZE,
        bin_chunk_size=BIN_CHUNK_SIZE,
        dtype=SPIKE_DTYPE,
        compress_data=COMPRESS_DATA,
        trial_params_format=TRIAL_PARAMS_FORMAT,
    )
    h5_path = os.path.join(output_directory, fname + '.h5')
    complete, missing = h5_is_complete(h5_path)
    if not complete:
        raise RuntimeError(f"HDF5 at {h5_path} is missing {', '.join(missing)}; the write did not finish cleanly")
    if missing:
        logger.warning(f"HDF5 written without optional tables: {', '.join(missing)}")
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
    parser.add_argument('--suffix', type=str, default='',
                        help='Appended to the HDF5 filename stem (<date><suffix>.h5), to regenerate without overwriting')
    args = parser.parse_args()
    process_ks_data(args.monkey, args.date, n_jobs=args.n_jobs, force=args.force, output_base=args.output_base,
                    suffix=args.suffix)
