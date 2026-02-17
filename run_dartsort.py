"""DARTsort spike sorting pipeline for Neuropixels recordings.

Pipeline steps:
    1. Load SpikeGLX recording and preprocess (highpass, phase shift, bad channel removal, spatial filter, zscore)
    2. Plot raw traces heatmap
    3. Run DARTsort subtraction (detect, denoise, localize spikes) — GPU-parallelized across chunks
    4. Run DREDge motion registration (rigid, bin_s=5.0 for long recordings)
    5. Compute motion-corrected channel assignments
    6. Save results and generate registration plots

Saved files (in <data_path>/dartsort_output/):
    - subtraction.h5                    : DARTsort subtraction results (spike waveforms, features)
    - geom.npy                          : Channel geometry after bad channel removal (n_channels, 2)
    - bad_channels.npy                  : Detected bad channel IDs
    - times_samples.npy                 : Spike times in samples (n_spikes,)
    - channels.npy                      : Max channel per spike (n_spikes,)
    - channel_index.npy                 : Channel neighborhood index
    - denoised_ptp_amplitudes.npy       : Peak-to-trough amplitudes (n_spikes,)
    - denoised_ptp_amplitude_vectors.npy: Per-channel amplitude vectors (n_spikes, n_neighbors)
    - point_source_localizations.npy    : Spike localizations [x, y, z, alpha] (n_spikes, 4)
    - denoised_logpeaktotrough.npy      : Log peak-to-trough ratio (n_spikes,)
    - max_channels_registered.npy       : Motion-corrected channel assignments (n_spikes,)
    - motion_est.pkl                    : DREDge motion estimate object
"""

import os, sys, logging
import warnings
import numpy as np
import numpy.typing as npt
import spikeinterface.full as si
from dartsort.main import subtract, ComputationConfig
from dartsort.util.data_util import DARTsortSorting
import dartsort.vis as dartvis
from pathlib import Path
from dredge import dredge_ap
import dredge.motion_util as mu
import matplotlib.pyplot as plt
import torch
import pickle
import shutil

# Suppress PyTorch tensor resize warning from dartsort library
warnings.filterwarnings("ignore", message="An output with one or more elements was resized")
try:
    from .utils_meta import init_dirs
    from .make_engram_path import BASE_DATA_PATH, BASE_SAVE_OUT_PATH
except ImportError:
    from data_analysis_tools_mkTurk.utils_meta import init_dirs
    from data_analysis_tools_mkTurk.make_engram_path import BASE_DATA_PATH, BASE_SAVE_OUT_PATH

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def resolve_dartsort_path(monkey: str, date: str) -> tuple[Path, Path, Path, Path]:
    """Resolve data, save, and plot paths for a given monkey/date session."""
    data_path_list, save_out_path, plot_save_out_path = init_dirs(BASE_DATA_PATH, monkey, date, BASE_SAVE_OUT_PATH)

    if len(data_path_list) == 1:
        data_path = data_path_list[0]
        save_out_path = save_out_path[0]
        plot_save_out_path = plot_save_out_path[0]
    else:
        raise ValueError('Multiple or no data paths found for given monkey and date')
    
    output_path = data_path / "dartsort_output"
    output_path.mkdir(exist_ok=True)

    return data_path, save_out_path, plot_save_out_path, output_path

def prep_dartsort(data_path: Path, output_path: Path, save_preprocessed_data: bool = False, remove_bad_channels: bool = True) -> si.BaseRecording:
    """Load SpikeGLX recording and apply IBL-style preprocessing (highpass, phase shift, zscore).
    If save_preprocessed_data=True, caches the binary to data_path/rec_ppx for reuse by Kilosort.
    Saves geom.npy and bad_channels.npy to output_path for downstream use."""

    rec = si.read_spikeglx(data_path, stream_id="imec0.ap")

    if save_preprocessed_data and (data_path / "rec_ppx").exists():
        rec = si.read_binary_folder(data_path / "rec_ppx")
    else:
        rec = si.highpass_filter(rec)
        rec = si.phase_shift(rec)
        bad_channel_ids, channel_labels = si.detect_bad_channels(rec)
        if remove_bad_channels:
            rec = rec.remove_channels(bad_channel_ids)
        rec = si.highpass_spatial_filter(rec)
        rec = si.zscore(rec, num_chunks_per_segment=50, mode="mean+std")

        # Save bad channels and geometry for downstream use
        np.save(output_path / 'bad_channels.npy', bad_channel_ids)

        if save_preprocessed_data:
            n_cpus, _ = detect_compute_resources()
            rec = rec.save(folder=data_path / "rec_ppx", n_jobs=n_cpus, chunk_duration="1s")

    np.save(output_path / 'geom.npy', rec.get_channel_locations())

    return rec

def plot_traces(rec: si.BaseRecording, output_path: Path) -> None:
    """Plot first 1000 samples of traces as a heatmap."""
    traces = rec.get_traces(0, 0, 1000)
    vm = np.percentile(np.abs(traces), 99.9)
    plt.imshow(traces.T, aspect="auto", vmin=-vm, vmax=vm, cmap=plt.cm.seismic) # type: ignore
    plt.colorbar(label="amplitude (su)")
    plt.xlabel("time (samples)")
    plt.ylabel("channel");
    plt.savefig(output_path / 'traces.png')
    plt.close()

def plot_unregistered_positions(initial_detections: DARTsortSorting, output_path: Path) -> None:
    """Plot spike positions before motion registration."""
    fig, ax = plt.subplots()
    dartvis.scatter_spike_features(
        sorting=initial_detections,
        amplitudes_dataset_name="denoised_ptp_amplitudes",
    );

    plt.savefig(output_path / "unregistered_positions.png")
    plt.close()

def save_results(initial_detections: DARTsortSorting, output_path: Path) -> None:
    """Save spike times, amplitudes, channels, localizations, and waveform features as .npy files."""
    np.save(output_path/ 'times_samples.npy', initial_detections.times_samples)
    np.save(output_path / 'denoised_ptp_amplitudes.npy', initial_detections.denoised_ptp_amplitudes)
    np.save(output_path / 'denoised_ptp_amplitude_vectors.npy', initial_detections.denoised_ptp_amplitude_vectors)
    np.save(output_path / 'channels.npy', initial_detections.channels)
    np.save(output_path / 'channel_index.npy', initial_detections.channel_index)
    np.save(output_path / 'point_source_localizations.npy', initial_detections.point_source_localizations)
    np.save(output_path / 'denoised_logpeaktotrough.npy', initial_detections.denoised_logpeaktotrough)

def run_registration(initial_detections: DARTsortSorting, rec: si.BaseRecording):
    """Run DREDge motion registration, filtering spikes within ±50um of probe bounds.
    Uses bin_s=5.0 for long recordings (>1hr) to reduce memory usage."""
    z = initial_detections.point_source_localizations[:, 2]
    geom = rec.get_channel_locations()
    valid = z == np.clip(z, geom[:, 1].min() - 50, geom[:, 1].max() + 50)
    z = z[valid]
    t = initial_detections.times_seconds[valid]
    a = initial_detections.denoised_ptp_amplitudes[valid]

    motion_est, extra_info = dredge_ap.register(a, z, t, bin_s=5.0)
    return motion_est

def plot_registration(initial_detections: DARTsortSorting, motion_est, output_path: Path) -> None:
    """Plot unregistered spikes with motion traces, and motion-corrected spike positions."""
    # Unregistered with motion estimate overlay
    fig, ax = plt.subplots()
    dartvis.scatter_time_vs_depth(
        sorting=initial_detections,
        amplitudes_dataset_name="denoised_ptp_amplitudes",
    )
    ls = mu.plot_me_traces(motion_est, plt.gca(), color="r")
    plt.legend([ls[0]], ["motion estimate"], loc="upper left")
    plt.xlabel("time (s)")
    plt.ylabel("depth (um)")
    plt.savefig(output_path / "registration.png")
    plt.close()

    # Registered (motion-corrected) positions
    fig, ax = plt.subplots()
    dartvis.scatter_time_vs_depth(
        sorting=initial_detections,
        amplitudes_dataset_name="denoised_ptp_amplitudes",
        registered=True,
        motion_est=motion_est,
        limits=None,
    )
    ax.set_xlabel("time (s)")
    ax.set_ylabel("motion-corrected depth (um)")
    plt.savefig(output_path / "registered_positions.png")
    plt.close()

def compute_registered_channels(
    initial_detections: DARTsortSorting, rec: si.BaseRecording, motion_est
) -> npt.NDArray[np.int_]:
    """Compute nearest channel for each spike after motion correction.
    Uses Euclidean distance between spike (x, corrected_depth) and channel geometry."""
    geom = rec.get_channel_locations()
    times_samples = initial_detections.times_samples
    point_source_localizations = initial_detections.point_source_localizations
    depths_um = point_source_localizations[:, 2]
    corrected_s = motion_est.correct_s(times_samples, depths_um)
    max_channels_registered = np.array([
        np.argmin(np.square(geom[:, 0] - point_source_localizations[j, 0]) + np.square(geom[:, 1] - corrected_s[j]))
        for j in range(point_source_localizations.shape[0])
    ], dtype=int)
    return max_channels_registered

def detect_compute_resources() -> tuple[int, int]:
    """Detect available CPUs and GPUs, respecting SLURM allocations."""
    n_cpus = int(os.environ.get('SLURM_CPUS_PER_TASK', os.cpu_count() or 1))
    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    logger.info(f"Detected {n_cpus} CPUs, {n_gpus} GPUs")
    return n_cpus, n_gpus

def get_initial_detections(output_path: Path, rec: si.BaseRecording) -> DARTsortSorting:
    """Load existing subtraction.h5 if available, otherwise run dartsort.subtract."""
    n_cpus, n_gpus = detect_compute_resources()
    h5_path = output_path / "subtraction.h5"
    
    if h5_path.exists():
        try:
            initial_detections = DARTsortSorting.from_peeling_hdf5(h5_path, load_simple_features=True)
            logger.info("Loaded existing subtraction results")
        except OSError as e:
            if "already open for write" in str(e):
                logger.warning(f"HDF5 file locked or corrupted. Deleting and rerunning subtraction.")
                h5_path.unlink()
                initial_detections = subtract(output_path, rec, computation_cfg=ComputationConfig(n_jobs_cpu=n_cpus, n_jobs_gpu=n_gpus))
                if initial_detections is None:
                    raise RuntimeError("DARTsort subtraction failed to produce results.")
            else:
                raise
    else:
        initial_detections = subtract(output_path, rec, computation_cfg=ComputationConfig(n_jobs_cpu=n_cpus, n_jobs_gpu=n_gpus))
        if initial_detections is None:
            raise RuntimeError("DARTsort subtraction failed to produce results.")
    return initial_detections

EXPECTED_OUTPUT_FILES = [
    'subtraction.h5', 'geom.npy', 'bad_channels.npy', 'times_samples.npy',
    'channels.npy', 'channel_index.npy', 'denoised_ptp_amplitudes.npy',
    'denoised_ptp_amplitude_vectors.npy', 'point_source_localizations.npy',
    'denoised_logpeaktotrough.npy', 'max_channels_registered.npy', 'motion_est.pkl',
]

def is_session_complete(output_path: Path) -> bool:
    """Check if all expected output files exist for this session."""
    missing = [f for f in EXPECTED_OUTPUT_FILES if not (output_path / f).exists()]
    if set(missing) == set(EXPECTED_OUTPUT_FILES):
        logger.info("No output files found, session is not complete")
        return False
    elif missing:
        logger.info(f"Missing outputs: {', '.join(missing)}")
        return False
    return True

def stage_recording_locally(rec: si.BaseRecording, data_path: Path) -> si.BaseRecording:
    """Save preprocessed recording to local /tmp for fast I/O during subtraction."""
    n_cpus, _ = detect_compute_resources()
    job_id = os.environ.get('SLURM_JOB_ID', os.getpid())
    local_dir = Path(f"/tmp/dartsort_{job_id}") / data_path.name / "rec_ppx"
    if local_dir.exists():
        logger.info(f"Loading existing local cache: {local_dir}")
        return si.read_binary_folder(local_dir)
    local_dir.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Staging preprocessed recording to {local_dir}")
    rec = rec.save(folder=local_dir, n_jobs=n_cpus, chunk_duration="1s", dtype="float16")
    logger.info(f"Staging complete: {local_dir}")
    return rec

def cleanup_local_staging() -> None:
    """Remove local /tmp staging directory for this job."""
    job_id = os.environ.get('SLURM_JOB_ID', os.getpid())
    local_dir = Path(f"/tmp/dartsort_{job_id}")
    if local_dir.exists():
        shutil.rmtree(local_dir)
        logger.info(f"Cleaned up local staging: {local_dir}")

def run_dartsort(monkey: str, date: str) -> None:
    """Run full dartsort pipeline: preprocess, subtract, save, register, and plot."""
    logger.info(f"Starting dartsort pipeline for {monkey} {date}")

    data_path, save_out_path, plot_save_out_path, output_path = resolve_dartsort_path(monkey, date)
    logger.info(f"Resolved paths: data={data_path}, save_out={save_out_path}, plot_save_out={plot_save_out_path}")

    if is_session_complete(output_path):
        logger.info(f"Session already complete, skipping {monkey} {date}")
        return

    rec = prep_dartsort(data_path, output_path, save_preprocessed_data=False, remove_bad_channels=True)
    logger.info("Preprocessing completed")

    plot_traces(rec, plot_save_out_path)
    logger.info("Trace plot completed")

    try:
        rec = stage_recording_locally(rec, data_path)
        logger.info("Local staging completed")
        initial_detections = get_initial_detections(output_path, rec)
        logger.info("Spike subtraction completed")
    finally:
        cleanup_local_staging()
        logger.info("Local staging cleanup completed")

    motion_est = run_registration(initial_detections, rec)
    logger.info("Motion registration completed")

    max_channels_registered = compute_registered_channels(initial_detections, rec, motion_est)
    logger.info("Registered channels computed")

    # Save all results
    save_results(initial_detections, output_path)
    np.save(output_path / 'max_channels_registered.npy', max_channels_registered)
    pickle.dump(motion_est, open(output_path / "motion_est.pkl", "wb"))
    logger.info("Results saved")

    # Plots
    plot_unregistered_positions(initial_detections, output_path)
    plot_registration(initial_detections, motion_est, output_path)
    logger.info("Plots completed")

    logger.info("Pipeline finished")



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run DARTsort spike sorting pipeline')
    parser.add_argument('--monkey', type=str, required=True, help='Monkey name')
    parser.add_argument('--date', type=str, required=True, help='Recording date (YYYYMMDD)')
    args = parser.parse_args()

    run_dartsort(args.monkey, args.date)
