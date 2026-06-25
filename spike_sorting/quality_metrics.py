"""Per-unit quality metrics for a Kilosort4 sort, used to identify good single units.

Computes and saves three arrays (one entry per template/cluster index) to the
session's save-out kilosort4/ folder:
    - presence_ratios.npy  : fraction of the recording the unit fires in (0-1)
    - amplitude_cutoffs.npy : estimated fraction of spikes missing below threshold (0-0.5)
    - viol_rates.npy       : ISI-violation false-positive rate (0 = clean, >1 = contaminated)

A "good single unit" is typically selected by combining these (e.g. high presence
ratio, low amplitude cutoff, low ISI violations) together with KSLabel == 'good'.

The metric functions follow the AllenSDK / cortex-lab sortingQuality formulation
(Hill et al. 2011), matching the Kilosort.ipynb implementation.
"""

import logging
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d

from ..utils_meta import init_dirs
from ..make_engram_path import BASE_DATA_PATH, BASE_SAVE_OUT_PATH
from .staging import find_recording_dir

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

SAMP_RATE = 30000

# Good-unit selection thresholds (match Kilosort.ipynb's strict/less-strict sets).
VIOL_RATE_MAX = 0.5        # ISI-violation fpRate below this
FIRING_RATE_MIN = 0.5      # Hz, above this
PRESENCE_RATIO_MIN = 0.9   # fires in >90% of time bins
CONTAM_PCT_MAX = 50        # Kilosort ContamPct below this
AMP_CUTOFF_MAX = 0.5       # estimated missing-spike fraction below this (strict only)
# ISI-violation parameters. isi_threshold is the refractory window: any inter-spike
# interval shorter than this counts as a violation. min_isi is the tiny floor below
# which a second spike is treated as a duplicate and dropped.
# NOTE: Kilosort.ipynb had these two SWAPPED (it passed min_isi into the
# isi_threshold slot), which made isi_threshold effectively 0 -> sum(isis < 0) == 0
# -> viol_rates all zero for every unit. Fixed here so viol_rates is meaningful;
# values therefore differ from the old notebook output.
ISI_THRESHOLD = 0.35 * 1e-3  # 0.35 ms refractory window
MIN_ISI = 0                  # duplicate-spike floor


# ---- single-spike-train metrics --------------------------------------------------

def firing_rate(spike_train, min_time=None, max_time=None):
    """Firing rate (Hz). Uses first/last spike if no bounds given."""
    if min_time is not None and max_time is not None:
        duration = max_time - min_time
    else:
        duration = np.max(spike_train) - np.min(spike_train)
    return spike_train.size / duration


def isi_violations(spike_train, min_time, max_time, isi_threshold, min_isi=0):
    """ISI-violation false-positive rate for one spike train (Hill et al. 2011).

    fpRate: 0 = perfect, <0.5 some contamination, >1 heavy contamination.
    """
    duplicate_spikes = np.where(np.diff(spike_train) <= min_isi)[0]
    spike_train = np.delete(spike_train, duplicate_spikes + 1)
    isis = np.diff(spike_train)

    num_spikes = len(spike_train)
    num_violations = sum(isis < isi_threshold)
    violation_time = 2 * num_spikes * (isi_threshold - min_isi)
    total_rate = firing_rate(spike_train, min_time, max_time)
    c = num_violations / (violation_time * total_rate)
    if c < 0.25:  # valid solution to the quadratic for fpRate
        fpRate = (1 - np.sqrt(1 - 4 * c)) / 2
    else:  # no valid solution
        fpRate = 1.0
    return fpRate, num_violations


def presence_ratio(spike_train, min_time, max_time, num_bins=100):
    """Fraction of time bins in which the unit spikes (0-1)."""
    h, _ = np.histogram(spike_train, np.linspace(min_time, max_time, num_bins))
    return np.sum(h > 0) / num_bins


def amplitude_cutoff(amplitudes, num_histogram_bins=500, histogram_smoothing_value=3):
    """Approximate fraction of spikes missing from an amplitude distribution (0-0.5).

    Assumes a symmetric amplitude histogram (not valid under drift).
    """
    amplitudes = amplitudes[~np.isnan(amplitudes)]
    # np.histogram can't subdivide a zero-width (or empty/non-finite) range into
    # num_histogram_bins finite bins -> ValueError. Such a cluster has no amplitude
    # spread, so no spikes are cut off: report 0.
    if amplitudes.size == 0 or not np.isfinite(amplitudes).all() \
            or np.ptp(amplitudes) == 0:
        return 0.0
    h, b = np.histogram(amplitudes, num_histogram_bins, density=True)
    pdf = gaussian_filter1d(h, histogram_smoothing_value)
    support = b[:-1]

    peak_index = np.argmax(pdf)
    G = np.argmin(np.abs(pdf[peak_index:] - pdf[0])) + peak_index

    bin_size = np.mean(np.diff(support))
    fraction_missing = np.sum(pdf[G:]) * bin_size
    return np.min([fraction_missing, 0.5])


# ---- per-cluster aggregation -----------------------------------------------------

def calculate_presence_ratio(spike_times, spike_clusters, total_units):
    cluster_ids = np.unique(spike_clusters)
    ratios = np.zeros((total_units,))
    for cluster_id in cluster_ids:
        m = spike_clusters == cluster_id
        ratios[cluster_id] = presence_ratio(
            spike_times[m], min_time=np.min(spike_times), max_time=np.max(spike_times)
        )
    return ratios


def calculate_isi_violations(spike_times, spike_clusters, total_units, isi_threshold, min_isi):
    cluster_ids = np.unique(spike_clusters)
    viol_rates = np.zeros((total_units,))
    num_viol = np.zeros((total_units,))
    for cluster_id in cluster_ids:
        m = spike_clusters == cluster_id
        viol_rates[cluster_id], num_viol[cluster_id] = isi_violations(
            spike_times[m], min_time=np.min(spike_times), max_time=np.max(spike_times),
            isi_threshold=isi_threshold, min_isi=min_isi,
        )
    return viol_rates, num_viol


def calculate_amplitude_cutoff(spike_clusters, amplitudes, total_units):
    cluster_ids = np.unique(spike_clusters)
    amplitude_cutoffs = np.zeros((total_units,))
    for cluster_id in cluster_ids:
        m = spike_clusters == cluster_id
        amplitude_cutoffs[cluster_id] = amplitude_cutoff(amplitudes[m])
    return amplitude_cutoffs


# ---- good-unit selection ---------------------------------------------------------

def is_good_unit(viol_rates: np.ndarray, fr: np.ndarray, presence_ratios: np.ndarray,
                 contam_pct: np.ndarray,
                 amplitude_cutoffs: np.ndarray | None = None,
                 KSLabel: np.ndarray | None = None,
                 strict: bool = True, require_single_unit: bool = True) -> np.ndarray:
    """Boolean mask of which units pass the quality criteria (Kilosort.ipynb's sets).

    Vectorized: pass the per-template metric arrays (all indexed by template id) and
    get back a boolean array; or pass scalars to test a single unit. NaN metrics
    (templates with no spikes) fail every comparison and so are excluded.

    Always applied:
        viol_rates < VIOL_RATE_MAX, fr > FIRING_RATE_MIN,
        presence_ratios > PRESENCE_RATIO_MIN, contam_pct < CONTAM_PCT_MAX
    strict=True               also requires amplitude_cutoffs < AMP_CUTOFF_MAX
                              (amplitude_cutoffs must be provided).
    require_single_unit=True  also requires KSLabel == 'good' (KSLabel must be
                              provided); set False for the "single_mua" sets that
                              keep MUA-labelled units too.

    The four Kilosort.ipynb sets map to:
        strict_single_unit       : strict=True,  require_single_unit=True
        less_strict_single_unit  : strict=False, require_single_unit=True
        strict_single_mua_unit   : strict=True,  require_single_unit=False
        less_strict_single_mua   : strict=False, require_single_unit=False
    """
    viol_rates = np.asarray(viol_rates, dtype=float)
    good = (
        (viol_rates < VIOL_RATE_MAX)
        & (np.asarray(fr, dtype=float) > FIRING_RATE_MIN)
        & (np.asarray(presence_ratios, dtype=float) > PRESENCE_RATIO_MIN)
        & (np.asarray(contam_pct, dtype=float) < CONTAM_PCT_MAX)
    )
    if strict:
        if amplitude_cutoffs is None:
            raise ValueError("strict=True requires amplitude_cutoffs")
        good = good & (np.asarray(amplitude_cutoffs, dtype=float) < AMP_CUTOFF_MAX)
    if require_single_unit:
        if KSLabel is None:
            raise ValueError("require_single_unit=True requires KSLabel")
        good = good & (np.asarray(KSLabel) == 'good')
    return good


# ---- session entry point ---------------------------------------------------------

def _unwhiten(x, wmi):
    return np.dot(np.ascontiguousarray(x), np.ascontiguousarray(wmi))


def _resolve_paths(monkey: str, date: str) -> tuple[Path, Path]:
    """Resolve (engram kilosort4 dir, save-out kilosort4 dir) for one session.

    One session per (monkey, date); raises on multiple/none. The raw KS4 output is
    read from the engram run folder; metrics are written to the save-out side
    (alongside KSLabel.npy), matching Kilosort.ipynb's kilosort_path.
    """
    data_path_list, save_out_path_list, _ = init_dirs(BASE_DATA_PATH, monkey, date, BASE_SAVE_OUT_PATH)
    if len(data_path_list) != 1:
        raise ValueError('Multiple or no data paths found for given monkey and date')
    engram_rec_dir = find_recording_dir(Path(data_path_list[0]))
    return engram_rec_dir / 'kilosort4', Path(save_out_path_list[0]) / 'kilosort4'


def run_quality_metrics(monkey: str, date: str, overwrite: bool = False) -> None:
    """Compute and save presence_ratios / amplitude_cutoffs / viol_rates for a session.

    Reads the raw KS4 output (spike_times, spike_clusters, templates, amplitudes,
    whitening_mat_inv) and writes the three .npy arrays into the save-out kilosort4/
    folder. Indexed by template id, so they align with KSLabel.npy.
    """
    ks_data_dir, ks_out_dir = _resolve_paths(monkey, date)
    if not ks_data_dir.exists():
        raise FileNotFoundError(f"no raw kilosort4 output at {ks_data_dir}; run run_kilosort first")

    targets = ['presence_ratios.npy', 'amplitude_cutoffs.npy', 'viol_rates.npy']
    if not overwrite and all((ks_out_dir / t).exists() for t in targets):
        logger.info(f"quality metrics already present in {ks_out_dir}, skipping (overwrite=False)")
        return

    # Load raw KS4 output.
    spike_times = np.squeeze(np.load(ks_data_dir / 'spike_times.npy'))
    spike_clusters = np.squeeze(np.load(ks_data_dir / 'spike_clusters.npy'))
    templates = np.load(ks_data_dir / 'templates.npy')         # (n_templates, n_time, n_chan)
    amp_scale = np.squeeze(np.load(ks_data_dir / 'amplitudes.npy'))  # per-spike template scaling
    wmi = np.load(ks_data_dir / 'whitening_mat_inv.npy')
    n_templates = templates.shape[0]

    # Per-spike amplitude = unwhitened template peak-to-trough (best channel) * scale.
    template_uw = np.stack([_unwhiten(np.squeeze(t), wmi) for t in templates])
    temp_chan_amps = template_uw.max(axis=1) - template_uw.min(axis=1)
    temp_amps_unscaled = temp_chan_amps.max(axis=1)            # max over channels
    spike_amps = temp_amps_unscaled[spike_clusters] * amp_scale

    logger.info(f"computing quality metrics for {n_templates} templates "
                f"({len(spike_times)} spikes)")
    amplitude_cutoffs = calculate_amplitude_cutoff(spike_clusters, spike_amps, n_templates)
    presence_ratios = calculate_presence_ratio(spike_times, spike_clusters, n_templates)
    viol_rates, _ = calculate_isi_violations(
        spike_times / SAMP_RATE, spike_clusters, n_templates,
        isi_threshold=ISI_THRESHOLD, min_isi=MIN_ISI,
    )

    ks_out_dir.mkdir(parents=True, exist_ok=True)
    np.save(ks_out_dir / 'presence_ratios.npy', presence_ratios)
    np.save(ks_out_dir / 'amplitude_cutoffs.npy', amplitude_cutoffs)
    np.save(ks_out_dir / 'viol_rates.npy', viol_rates)
    logger.info(f"wrote presence_ratios / amplitude_cutoffs / viol_rates to {ks_out_dir}")


def save_template_metrics(monkey: str, date: str, overwrite: bool = False) -> None:
    """Save the Kilosort.ipynb post-processing template metrics for a session.

    These arrays are part of the standard Phy/KS post-processing output but are not
    emitted by Kilosort4 itself; quality_metrics derives them on the fly (via
    _load_fr / _load_contam_pct / unwhitened templates). This persists them as .npy
    into the save-out kilosort4/ folder so downstream code can read them directly
    instead of re-deriving. All are indexed by template id (aligned to KSLabel).

        - fr.npy              : per-template firing rate (Hz)
        - contamPct.npy       : Kilosort ContamPct (NaN where absent)
        - temp_chan_amps.npy  : (n_templates, n_chan) peak-to-trough of the unwhitened
                                template on each channel
        - template_depths.npy : amplitude-weighted depth (um) along the probe (ycoords)
        - template_xcoords.npy : amplitude-weighted x-position (um) across the probe
        - template_wf.npy     : (n_templates, n_time) scaled best-channel waveform
        - template_amp.npy    : best-channel peak-to-trough * mean template scale factor
        - template_dur.npy    : trough-to-peak duration of the best-channel waveform (s)
        - template_ptratio.npy : peak/trough amplitude ratio of the best-channel waveform
        - vertical_spread.npy : vertical extent (um) where channel amp > 15% of the peak

    Run after run_kilosort has produced the kilosort4/ output for the session.
    """
    ks_data_dir, ks_out_dir = _resolve_paths(monkey, date)
    if not ks_data_dir.exists():
        raise FileNotFoundError(f"no raw kilosort4 output at {ks_data_dir}; run run_kilosort first")

    targets = ['fr.npy', 'contamPct.npy', 'temp_chan_amps.npy', 'template_depths.npy',
               'template_xcoords.npy', 'template_wf.npy', 'template_amp.npy',
               'template_dur.npy', 'template_ptratio.npy', 'vertical_spread.npy']
    if not overwrite and all((ks_out_dir / t).exists() for t in targets):
        logger.info(f"template metrics already present in {ks_out_dir}, skipping (overwrite=False)")
        return

    templates = np.load(ks_data_dir / 'templates.npy')         # (n_templates, n_time, n_chan)
    wmi = np.load(ks_data_dir / 'whitening_mat_inv.npy')
    chan_pos = np.load(ks_data_dir / 'channel_positions.npy')
    xcoords, ycoords = chan_pos[:, 0], chan_pos[:, 1]
    spike_clusters = np.squeeze(np.load(ks_data_dir / 'spike_clusters.npy'))
    amp_scale = np.squeeze(np.load(ks_data_dir / 'amplitudes.npy'))  # per-spike template scaling
    n_templates, n_time, n_chan = templates.shape

    fr = _load_fr(ks_data_dir, n_templates)
    contam_pct = _load_contam_pct(ks_data_dir, n_templates)

    # Unwhitened template peak-to-trough per channel; best channel and weighted position.
    template_uw = np.stack([_unwhiten(np.squeeze(t), wmi) for t in templates])
    temp_chan_amps = template_uw.max(axis=1) - template_uw.min(axis=1)  # (n_templates, n_chan)
    temp_amps_unscaled = temp_chan_amps.max(axis=1)             # max over channels
    best_channel = np.argmax(temp_chan_amps, axis=1)
    template_depths = (temp_chan_amps * ycoords).sum(axis=1) / temp_chan_amps.sum(axis=1)
    template_xcoords = (temp_chan_amps * xcoords).sum(axis=1) / temp_chan_amps.sum(axis=1)

    # Per-template scale = mean of the per-spike amplitude scaling that used that template.
    counts = np.bincount(spike_clusters, minlength=n_templates)
    sums = np.bincount(spike_clusters, weights=amp_scale, minlength=n_templates)
    template_scale_factor = np.divide(sums, counts, out=np.full(n_templates, np.nan),
                                      where=counts > 0)
    template_scaled = template_uw * template_scale_factor[:, None, None]

    # Best-channel scaled waveform and its shape metrics.
    template_wf = template_scaled[np.arange(n_templates), :, best_channel]  # (n_templates, n_time)
    template_amp = temp_amps_unscaled * template_scale_factor
    trough_t = np.argmin(template_wf, axis=1)
    peak_t = np.argmax(template_wf, axis=1)
    template_dur = (peak_t - trough_t) / SAMP_RATE
    template_ptratio = np.max(template_wf, axis=1) / np.min(template_wf, axis=1)

    # Vertical spread: distance to the furthest channel still above 15% of the best
    # channel's peak-to-trough amplitude.
    vertical_spread = np.full(n_templates, np.nan)
    for n, (ch, t_amp) in enumerate(zip(best_channel, temp_chan_amps)):
        spread_ind = np.where(t_amp[ch] * 0.15 > t_amp)[0]
        if len(spread_ind):
            furthest = spread_ind[np.argsort(t_amp[spread_ind])[::-1][0]]
            vertical_spread[n] = np.abs(ycoords[ch] - ycoords[furthest])

    ks_out_dir.mkdir(parents=True, exist_ok=True)
    np.save(ks_out_dir / 'fr.npy', fr)
    np.save(ks_out_dir / 'contamPct.npy', contam_pct)
    np.save(ks_out_dir / 'temp_chan_amps.npy', temp_chan_amps)
    np.save(ks_out_dir / 'template_depths.npy', template_depths)
    np.save(ks_out_dir / 'template_xcoords.npy', template_xcoords)
    np.save(ks_out_dir / 'template_wf.npy', template_wf)
    np.save(ks_out_dir / 'template_amp.npy', template_amp)
    np.save(ks_out_dir / 'template_dur.npy', template_dur)
    np.save(ks_out_dir / 'template_ptratio.npy', template_ptratio)
    np.save(ks_out_dir / 'vertical_spread.npy', vertical_spread)
    logger.info(f"wrote {len(targets)} template metrics to {ks_out_dir}")


def _load_fr(ks_data_dir: Path, n_templates: int) -> np.ndarray:
    """Per-template firing rate (Hz): spike count / recording duration."""
    spike_times = np.squeeze(np.load(ks_data_dir / 'spike_times.npy'))
    spike_clusters = np.squeeze(np.load(ks_data_dir / 'spike_clusters.npy'))
    counts = np.bincount(spike_clusters, minlength=n_templates)
    n_secs = spike_times.max() / SAMP_RATE
    return counts / n_secs


def _load_contam_pct(ks_data_dir: Path, n_templates: int) -> np.ndarray:
    """Per-template Kilosort ContamPct from cluster_ContamPct.tsv (NaN where absent)."""
    import pandas as pd
    df = pd.read_csv(ks_data_dir / 'cluster_ContamPct.tsv', sep='\t')
    contam = np.full(n_templates, np.nan)
    cid = df['cluster_id'].to_numpy()
    contam[cid] = df['ContamPct'].to_numpy()
    return contam


def _load_kslabel(ks_data_dir: Path, n_templates: int) -> np.ndarray:
    """Per-template KSLabel ('good'/'mua') from cluster_KSLabel.tsv ('' where absent)."""
    import pandas as pd
    df = pd.read_csv(ks_data_dir / 'cluster_KSLabel.tsv', sep='\t')
    labels = np.full(n_templates, '', dtype=object)
    cid = df['cluster_id'].to_numpy()
    labels[cid] = df['KSLabel'].to_numpy()
    return labels


def load_good_unit_mask(monkey: str, date: str, strict: bool = True,
                        require_single_unit: bool = True) -> np.ndarray:
    """Load a session's saved metrics and return the is_good_unit boolean mask.

    Convenience wrapper: reads presence_ratios / amplitude_cutoffs / viol_rates / fr /
    contamPct from the save-out kilosort4/ folder, derives KSLabel from the raw KS4
    output, and applies is_good_unit. Run run_quality_metrics and save_template_metrics
    first.
    """
    ks_data_dir, ks_out_dir = _resolve_paths(monkey, date)
    presence_ratios = np.load(ks_out_dir / 'presence_ratios.npy')
    amplitude_cutoffs = np.load(ks_out_dir / 'amplitude_cutoffs.npy')
    viol_rates = np.load(ks_out_dir / 'viol_rates.npy')
    fr = np.load(ks_out_dir / 'fr.npy')
    contam_pct = np.load(ks_out_dir / 'contamPct.npy')
    n_templates = len(viol_rates)
    KSLabel = _load_kslabel(ks_data_dir, n_templates)
    return is_good_unit(viol_rates, fr, presence_ratios, contam_pct,
                        amplitude_cutoffs=amplitude_cutoffs, KSLabel=KSLabel,
                        strict=strict, require_single_unit=require_single_unit)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Compute Kilosort4 unit quality metrics')
    parser.add_argument('--monkey', type=str, required=True, help='Monkey name')
    parser.add_argument('--date', type=str, required=True, help='Recording date (YYYYMMDD)')
    parser.add_argument('--overwrite', action='store_true', help='Recompute even if metrics exist')
    args = parser.parse_args()

    run_quality_metrics(args.monkey, args.date, overwrite=args.overwrite)
    save_template_metrics(args.monkey, args.date, overwrite=args.overwrite)
