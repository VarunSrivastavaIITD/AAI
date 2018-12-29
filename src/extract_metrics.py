import numpy as np
import pandas as pd
from scipy.signal import (
    butter,
    filtfilt,
    find_peaks,
    find_peaks_cwt,
    medfilt,
    savgol_filter,
)

from utils import detrend, smooth, positions2onehot, lowpass


def geneggfilter(data):
    data = data / np.max(np.abs(data))
    # data[data > 0] = 0
    # data = medfilt(data, 9)
    # data = savgol_filter(data, 21, 2)
    data = smooth(data, 49)
    return data


def groundeggfilter(data):
    data = data / np.max(np.abs(data))
    # data[data > 0] = 0
    # data = medfilt(data, 7)
    # data = savgol_filter(data, 7, 2)
    return data


def genegg_process(data):
    data = geneggfilter(data)

    degg = np.gradient(data, edge_order=2)
    degg[degg > 0] = 0
    degg = medfilt(degg, 3)

    # degg = savgol_filter(degg, 51, 2)
    return degg


def groundegg_process(data):
    data = groundeggfilter(data)

    degg = np.gradient(data, edge_order=2)
    degg[degg > 0] = 0
    # degg = medfilt(degg, 7)
    return degg


def detectgenwaveletgci(data, peak_range=(7, 25)):
    degg = genegg_process(data)
    out = find_peaks_cwt(-degg, np.arange(*peak_range))
    return out


def detectgroundwaveletgci(data, peak_range=(7, 15)):
    degg = groundegg_process(data)
    out = find_peaks_cwt(-degg, np.arange(*peak_range))
    return out


def detectscipygci(data):
    degg = np.gradient(data)
    out, _ = find_peaks(-degg, distance=55, prominence=(None, None))
    return out


def detect_voiced_region(true_egg, reconstructed_egg, power_threshold=0.01):
    def _get_signal_power(x, window):
        power = np.convolve(x ** 2, window / window.sum(), mode="same")
        return power

    def _get_window(window_len=10, window="flat"):
        if window == "flat":  # average
            w = np.ones(window_len, "d")
        else:
            w = eval("np." + window + "(window_len)")

        return w

    true_scaler = pd.Series(np.abs(true_egg)).nlargest(100).median()
    reconstructed_scaler = pd.Series(np.abs(reconstructed_egg)).nlargest(100).median()

    true_egg = true_egg / true_scaler
    reconstructed_egg = reconstructed_egg / reconstructed_scaler

    window = _get_window(window_len=501, window="hanning")
    power = _get_signal_power(true_egg, window)

    regions = power >= power_threshold
    true_egg_voiced = true_egg[regions]
    reconstructed_egg_voiced = reconstructed_egg[regions]

    return (
        true_egg_voiced * true_scaler,
        reconstructed_egg_voiced * reconstructed_scaler,
    )


def corrected_naylor_metrics(ref_signal, est_signal):
    # Settings
    # TODO: precise values to be decided later

    assert np.squeeze(ref_signal).ndim == 1
    assert np.squeeze(est_signal).ndim == 1

    ref_signal = np.squeeze(ref_signal)
    est_signal = np.squeeze(est_signal)

    min_f0 = 20
    max_f0 = 500
    min_glottal_cycle = 1 / max_f0
    max_glottal_cycle = 1 / min_f0

    nHit = 0
    nMiss = 0
    nFalse = 0
    nCycles = 0
    highNumCycles = 100000
    estimation_distance = np.full(highNumCycles, np.nan)

    ref_fwdiffs = np.zeros_like(ref_signal)
    ref_bwdiffs = np.zeros_like(ref_signal)

    ref_fwdiffs[:-1] = np.diff(ref_signal)
    ref_fwdiffs[-1] = max_glottal_cycle
    ref_bwdiffs[1:] = np.diff(ref_signal)
    ref_bwdiffs[0] = max_glottal_cycle

    hits = []
    misses = []
    falsealarms = []

    for i in range(len(ref_fwdiffs)):

        # m in original file
        ref_cur_sample = ref_signal[i]
        ref_dist_fw = ref_fwdiffs[i]
        ref_dist_bw = ref_bwdiffs[i]

        # Condition to check for valid larynx cycle
        # TODO: Check parity of differences, neg peak <-> gci, pos peak <-> goi
        # TODO: Check applicability of strict inequality
        dist_in_allowed_range = (
            min_glottal_cycle <= ref_dist_fw <= max_glottal_cycle
            and min_glottal_cycle <= ref_dist_bw <= max_glottal_cycle
        )
        if dist_in_allowed_range:

            cycle_start = ref_cur_sample - ref_dist_bw / 2
            cycle_stop = ref_cur_sample + ref_dist_fw / 2

            est_GCIs_in_cycle = est_signal[
                np.logical_and(est_signal > cycle_start, est_signal < cycle_stop)
            ]
            n_est_in_cycle = np.count_nonzero(est_GCIs_in_cycle)

            nCycles += 1

            if n_est_in_cycle == 1:
                nHit += 1
                estimation_distance[nHit] = est_GCIs_in_cycle[0] - ref_cur_sample
                hits.append((est_GCIs_in_cycle[0], estimation_distance))
            elif n_est_in_cycle < 1:
                nMiss += 1
                misses.append(ref_cur_sample)
            else:
                nFalse += 1
                falsealarms.extend(est_GCIs_in_cycle)

    estimation_distance = estimation_distance[np.invert(np.isnan(estimation_distance))]

    identification_rate = nHit / nCycles
    miss_rate = nMiss / nCycles
    false_alarm_rate = nFalse / nCycles
    identification_accuracy = (
        0 if np.size(estimation_distance) == 0 else np.std(estimation_distance)
    )

    return {
        "identification_rate": identification_rate,
        "miss_rate": miss_rate,
        "false_alarm_rate": false_alarm_rate,
        "identification_accuracy": identification_accuracy,
        "hits": hits,
        "misses": misses,
        "fars": falsealarms,
        "nhits": nHit,
        "nmisses": nMiss,
        "nfars": nFalse,
        "ncycles": nCycles,
    }


def extract_gci_metrics(true_egg, estimated_egg, detrend_egg=False, fs=16e3):
    fnames = {}
    if type(true_egg) is str:
        fnames = {"true_file": true_egg}
        true_egg = np.load(true_egg)
        if detrend_egg:
            _, true_egg = detrend(None, true_egg)
    if type(estimated_egg) is str:
        estimated_egg = np.load(estimated_egg)

    if len(true_egg) > len(estimated_egg):
        true_egg = true_egg[: len(estimated_egg)]

    true_egg, estimated_egg = detect_voiced_region(true_egg, estimated_egg)
    true_gci = detectgroundwaveletgci(true_egg) / fs
    estimated_gci = detectgenwaveletgci(estimated_egg) / fs

    metrics = corrected_naylor_metrics(true_gci, estimated_gci)
    metrics.update(fnames)
    return metrics


def extract_speed_quotient(smoothed_degg: np.ndarray, peaks: np.ndarray):
    assert len(peaks) % 2 == 1

    zero_cross_regions = np.stack((peaks[1::2], peaks[2::2]), axis=-1)

    zero_indices = [
        zero_crossings((smoothed_degg[r[0] + 1 : r[1] + 1])) for r in zero_cross_regions
    ]

    zero_positions = np.fromiter(
        (np.median(np.nonzero(zi)[0]) for zi in zero_indices), np.int
    )

    goi = peaks[1::2]
    ngci = peaks[2::2]

    sq = (ngci - zero_positions - goi - 1) / (zero_positions + 1)
    return np.sum(sq), sq.shape[0]


def extract_quotient_metrics(true_egg, estimated_egg, detrend_egg=False, fs=16e3):
    def _detect_voiced_region_as_regions(
        true_egg, reconstructed_egg, power_threshold=0.05
    ):
        def _get_signal_power(x, window):
            power = np.convolve(x ** 2, window / window.sum(), mode="same")
            return power

        def _get_window(window_len=10, window="flat"):
            if window == "flat":  # average
                w = np.ones(window_len, "d")
            else:
                w = eval("np." + window + "(window_len)")

            return w

        true_scaler = pd.Series(np.abs(true_egg)).nlargest(100).median()
        reconstructed_scaler = (
            pd.Series(np.abs(reconstructed_egg)).nlargest(100).median()
        )

        true_egg = true_egg / true_scaler
        reconstructed_egg = reconstructed_egg / reconstructed_scaler

        window = _get_window(window_len=501, window="hanning")
        power = _get_signal_power(true_egg, window)

        regions = power >= power_threshold
        # true_egg_voiced = true_egg[regions]
        # reconstructed_egg_voiced = reconstructed_egg[regions]

        return regions

    fnames = {}
    if type(true_egg) is str:
        fnames = {"true_file": true_egg}
        true_egg = np.load(true_egg)
        if detrend_egg:
            _, true_egg = detrend(None, true_egg)
    if type(estimated_egg) is str:
        estimated_egg = np.load(estimated_egg)

    true_egg = lowpass(true_egg)
    estimated_egg = lowpass(estimated_egg)

    if len(true_egg) > len(estimated_egg):
        true_egg = true_egg[: len(estimated_egg)]

    regions = _detect_voiced_region_as_regions(true_egg, estimated_egg)

    true_gci = detectgroundwaveletgci(true_egg)
    estimated_gci = detectgenwaveletgci(estimated_egg)

    true_egg, estimated_egg = groundeggfilter(true_egg), geneggfilter(estimated_egg)
    true_degg = np.gradient(true_egg, edge_order=2)
    estimated_degg = np.gradient(estimated_egg, edge_order=2)

    true_gci = positions2onehot(true_gci, regions.shape) * regions
    estimated_gci = positions2onehot(estimated_gci, regions.shape) * regions
    true_gci = np.nonzero(true_gci)[0]
    estimated_gci = np.nonzero(estimated_gci)[0]

    true_goi = []
    estimated_goi = []
    for i in range(true_gci.shape[0] - 1):
        true_goi.append(
            true_gci[i] + np.argmax(true_degg[true_gci[i] + 1 : true_gci[i + 1]]) + 1
        )
    for i in range(estimated_gci.shape[0] - 1):
        estimated_goi.append(
            estimated_gci[i]
            + np.argmax(estimated_degg[estimated_gci[i] + 1 : estimated_gci[i + 1]])
            + 1
        )

    true_goi = np.array(true_goi)
    estimated_goi = np.array(estimated_goi)

    labelregions = label(regions)[0]

    true_goi = positions2onehot(true_goi, regions.shape)
    estimated_goi = positions2onehot(estimated_goi, regions.shape)
    true_gci = positions2onehot(true_gci, regions.shape)
    estimated_gci = positions2onehot(estimated_gci, regions.shape)

    true_peaks = -true_gci + true_goi
    estimated_peaks = -estimated_gci + estimated_goi

    tpeaks = []
    true_degg_list = []
    for r in find_objects(labelregions):
        tregion = true_peaks[r]
        true_degg_region = true_degg[r]
        tpos = np.nonzero(tregion)[0]

        if len(tpos) < 2:
            continue

        if tregion[tpos[0]] > 0:
            tpos = tpos[1:]
        if tregion[tpos[-1]] > 0:
            tpos = tpos[:-1]
        assert len(tpos) % 2 == 1
        tregion = tregion[tpos[0] : tpos[-1] + 1]
        tpeaks.append(tregion)
        true_degg_list.append(true_degg_region[tpos[0] : tpos[-1] + 1])

    epeaks = []
    estimated_degg_list = []
    for r in find_objects(labelregions):
        eregion = estimated_peaks[r]
        estimated_degg_region = estimated_degg[r]
        epos = np.nonzero(eregion)[0]

        if len(epos) < 2:
            continue

        if eregion[epos[0]] > 0:
            epos = epos[1:]
        if eregion[epos[-1]] > 0:
            epos = epos[:-1]
        assert len(epos) % 2 == 1
        eregion = eregion[epos[0] : epos[-1] + 1]
        epeaks.append(eregion)
        estimated_degg_list.append(estimated_degg_region[epos[0] : epos[-1] + 1])

    true_peaks_list = [np.nonzero(t)[0] for t in tpeaks]
    estimated_peaks_list = [np.nonzero(e)[0] for e in epeaks]

    metrics = {
        "CQ_true": 0,
        "OQ_true": 0,
        "SQ_true": 0,
        "CQ_estimated": 0,
        "OQ_estimated": 0,
        "SQ_estimated": 0,
    }

    count = 0
    sq_count = 0
    for tr, tr_degg in zip(true_peaks_list, true_degg_list):
        for i in range(1, tr.shape[0], 2):
            count += 1
            metrics["CQ_true"] += (tr[i + 1] - tr[i]) / (tr[i + 1] - tr[i - 1])
            metrics["OQ_true"] += (tr[i] - tr[i - 1]) / (tr[i + 1] - tr[i - 1])

        temp = extract_speed_quotient(tr_degg, tr)
        metrics["SQ_true"] += temp[0]
        sq_count += temp[1]

    metrics["SQ_true"] /= sq_count
    metrics["CQ_true"] /= count
    metrics["OQ_true"] /= count

    count = 0
    sq_count = 0
    for er, er_degg in zip(estimated_peaks_list, estimated_degg_list):
        for i in range(1, er.shape[0], 2):
            count += 1
            metrics["CQ_estimated"] += (er[i + 1] - er[i]) / (er[i + 1] - er[i - 1])
            metrics["OQ_estimated"] += (er[i] - er[i - 1]) / (er[i + 1] - er[i - 1])

        temp = extract_speed_quotient(er_degg, er)
        metrics["SQ_estimated"] += temp[0]
        sq_count += temp[1]
    metrics["SQ_estimated"] /= sq_count
    metrics["CQ_estimated"] /= count
    metrics["OQ_estimated"] /= count

    metrics.update(fnames)
    return metrics
