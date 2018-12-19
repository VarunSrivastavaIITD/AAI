import numpy as np
from scipy.signal import (
    find_peaks,
    find_peaks_cwt,
    medfilt,
    savgol_filter,
    butter,
    filtfilt,
)
from librosa.util import peak_pick
import pandas as pd
from utils import detrend, smooth


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


def detectrosagci(data):
    degg = np.gradient(data)
    # normalization = np.median(np.sort(degg)[-100:])
    # degg = degg / normalization
    maxdiff = np.max(np.abs(degg))
    threshold = maxdiff / 5
    # threshold = 0
    window = 20
    wait = 50
    out = peak_pick(-degg, window, window, window, window, threshold, wait)
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
            elif n_est_in_cycle < 1:
                nMiss += 1
            else:
                nFalse += 1

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
    }


def extract_metrics(true_egg, estimated_egg, detrend_egg=False, fs=16e3):
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
