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
from scipy.ndimage import find_objects, label
from librosa import zero_crossings
from utils import (
    detrend,
    smooth,
    positions2onehot,
    lowpass,
    onehot2positions,
    minmaxnormalize,
)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

plt.switch_backend("qt5agg")


def geneggfilter(data):
    data = minmaxnormalize(data)
    # data[data > 0] = 0
    # data = medfilt(data, 9)
    # data = savgol_filter(data, 21, 2)
    data = smooth(data, 49)
    return data


def groundeggfilter(data):
    data = minmaxnormalize(data)
    # data[data > 0] = 0
    # data = medfilt(data, 7)
    # data = savgol_filter(data, 7, 2)
    data = smooth(data, 49)
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


def detect_voiced_region(
    true_egg, reconstructed_egg, power_threshold=0.01, return_regions=False
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
    reconstructed_scaler = pd.Series(np.abs(reconstructed_egg)).nlargest(100).median()

    true_egg = true_egg / true_scaler
    reconstructed_egg = reconstructed_egg / reconstructed_scaler

    window = _get_window(window_len=501, window="hanning")
    power = _get_signal_power(true_egg, window)

    regions = power >= power_threshold

    if return_regions:
        return regions

    true_egg_voiced = true_egg[regions]
    reconstructed_egg_voiced = reconstructed_egg[regions]

    return (
        true_egg_voiced * true_scaler,
        reconstructed_egg_voiced * reconstructed_scaler,
    )


def apply_region_to_positions(positions: np.ndarray, regions: np.ndarray):
    onehot = positions2onehot(positions, regions.shape) * regions
    return onehot2positions(onehot)


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


def extract_speed_quotient_egg(egg: np.ndarray, peaks: np.ndarray, region):
    assert len(peaks) % 2 == 1

    zero_cross_regions = np.stack((peaks[1::2], peaks[2::2]), axis=-1)

    # zero_indices = [
    #     zero_crossings((smoothed_degg[r[0] + 1 : r[1] + 1])) for r in zero_cross_regions
    # ]

    zero_indices = [
        np.argmax(egg[region.start + rz[0] + 1 : region.start + rz[1] + 1])
        for rz in zero_cross_regions
    ]

    zero_positions = np.fromiter((np.median(zi) for zi in zero_indices), np.int)

    goi = peaks[1::2]
    ngci = peaks[2::2]

    sq = (ngci - zero_positions - goi - 1) / (zero_positions + 1)
    return np.sum(sq), sq.shape[0]


def extract_goi(
    gci: np.ndarray, degg: np.ndarray, regions: np.ndarray = None
) -> np.ndarray:
    """Find goi positions in the degg signal

    Args:
        gci: (integer) positions of gci in the degg array
        degg: the degg signal

    Returns:
        positions of goi in the degg array

    """

    if regions is None:
        regions = np.ones_like(degg)

    voiced_slices = [
        obj[0] for obj in find_objects(label(regions)[0])
    ]  # List of slices
    onehot_gci = positions2onehot(gci, regions.shape)

    voiced_goi: list = []
    for s in voiced_slices:
        onehot_gci_voiced = onehot_gci[s].astype(np.int)
        positions_gci_voiced = s.start + onehot2positions(onehot_gci_voiced)
        positions_gci_voiced = positions_gci_voiced.astype(np.int)

        voiced_goi.extend(
            cur + np.argmax(degg[cur + 1 : next]) + 1
            for cur, next in zip(positions_gci_voiced, positions_gci_voiced[1:])
        )

    return np.array(voiced_goi).astype(np.int)


def extract_quotient_metrics(true_egg, estimated_egg, detrend_egg=False, fs=16e3):
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

    regions = detect_voiced_region(
        true_egg, estimated_egg, power_threshold=0.01, return_regions=True
    )

    true_gci = detectgroundwaveletgci(true_egg)
    estimated_gci = detectgenwaveletgci(estimated_egg)

    true_egg, estimated_egg = groundeggfilter(true_egg), geneggfilter(estimated_egg)
    true_degg = np.gradient(true_egg, edge_order=2)
    estimated_degg = np.gradient(estimated_egg, edge_order=2)

    true_gci = positions2onehot(true_gci, regions.shape) * regions
    estimated_gci = positions2onehot(estimated_gci, regions.shape) * regions
    true_gci = np.nonzero(true_gci)[0]
    estimated_gci = np.nonzero(estimated_gci)[0]

    # fig, axs = plt.subplots(2, 1, sharex=True)

    # fig.suptitle(
    #     "Visualization of GOI and GCI for file {}".format(
    #         fnames.get("true_file", "unknown file")
    #     )
    # )
    # axs[0].set_title("Ground Truth")
    # axs[1].set_title("Generated Lies")

    # axs[0].vlines(true_gci, 0, -1, color="b", label="GCI")
    # axs[1].vlines(estimated_gci, 0, -1, color="b", label="GCI")

    # axs[0].plot(true_egg, color="r", label="EGG")
    # axs[0].plot(true_degg, color="g", label="DEGG")
    # axs[1].plot(estimated_egg, color="r", label="EGG")
    # axs[1].plot(estimated_degg, color="g", label="DEGG")

    # for ax in axs:
    #     ax.plot(1 * regions, color="g")
    #     ax.axhline(0, *ax.get_xlim(), color="k")
    #     ax.set_ylabel("Amplitude")
    #     ax.set_xlabel("Sample")
    #     ax.legend(loc=1)
    # plt.subplots_adjust(
    #     top=0.92, bottom=0.09, left=0.08, right=0.95, hspace=0.1, wspace=0.2
    # )
    # mng = plt.get_current_fig_manager()
    # mng.window.showMaximized()
    # plt.show()

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
    true_regions = []
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
        true_regions.append(slice(r[0].start + tpos[0], r[0].start + tpos[-1] + 1))
        tpeaks.append(tregion)
        true_degg_list.append(true_degg_region[tpos[0] : tpos[-1] + 1])

    epeaks = []
    estimated_degg_list = []
    estimated_regions = []
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
        estimated_regions.append(slice(r[0].start + epos[0], r[0].start + epos[-1] + 1))
        epeaks.append(eregion)
        estimated_degg_list.append(estimated_degg_region[epos[0] : epos[-1] + 1])

    true_peaks_list = [np.nonzero(t)[0] for t in tpeaks]
    estimated_peaks_list = [np.nonzero(e)[0] for e in epeaks]

    # visualizaion begin

    # fig, axs = plt.subplots(2, 1, sharex=True)

    # fig.suptitle(
    #     "Visualization of GOI and GCI for file {}".format(
    #         fnames.get("true_file", "unknown file")
    #     )
    # )
    # axs[0].set_title("Ground Truth")
    # axs[1].set_title("Generated Lies")
    # gci_patch = mpatches.Patch(color="b", label="GCI")
    # goi_patch = mpatches.Patch(color="m", label="GOI")
    # sqpeak_patch = mpatches.Patch(color="y", label="Peak")
    # egg_patch = mpatches.Patch(color="r", label="EGG")
    # degg_patch = mpatches.Patch(color="g", label="DEGG")
    # for r, t in zip(true_regions, true_peaks_list):
    #     axs[0].vlines(r.start + np.array(t[::2]), 0, -1, color="b")
    #     axs[0].vlines(r.start + np.array(t[1::2]), 0, 1, color="m")

    #     degg_region = true_degg[r]

    #     zero_cross_regions = np.stack((t[1::2], t[2::2]), axis=-1)

    # zero_indices = [
    #     r.start
    #     + rz[0]
    #     + 1
    #     + onehot2positions(zero_crossings((degg_region[rz[0] + 1 : rz[1] + 1])))
    #     for rz in zero_cross_regions
    # ]
    #     zero_indices = [
    #         r.start
    #         + rz[0]
    #         + 1
    #         + np.argmax(true_egg[r.start + rz[0] + 1 : r.start + rz[1] + 1])
    #         for rz in zero_cross_regions
    #     ]

    #     zero_positions = np.fromiter((np.median(zi) for zi in zero_indices), np.int)
    #     axs[0].vlines(zero_positions, 0, 1, color="y")

    # for r, t in zip(estimated_regions, estimated_peaks_list):
    #     axs[1].vlines(r.start + np.array(t[::2]), 0, -1, color="b")
    #     axs[1].vlines(r.start + np.array(t[1::2]), 0, 1, color="m")

    #     degg_region = estimated_degg[r]

    #     zero_cross_regions = np.stack((t[1::2], t[2::2]), axis=-1)

    # zero_indices = [
    #     r.start
    #     + rz[0]
    #     + 1
    #     + onehot2positions(zero_crossings((degg_region[rz[0] + 1 : rz[1] + 1])))
    #     for rz in zero_cross_regions
    # ]
    #     zero_indices = [
    #         r.start
    #         + rz[0]
    #         + 1
    #         + np.argmax(estimated_egg[r.start + rz[0] + 1 : r.start + rz[1] + 1])
    #         for rz in zero_cross_regions
    #     ]

    #     zero_positions = np.fromiter((np.median(zi) for zi in zero_indices), np.int)
    #     axs[1].vlines(zero_positions, 0, 1, color="y")

    # axs[1].vlines(estimated_gci, 0, -1, color="b", label="GCI")

    # axs[0].plot(true_egg, color="r", label="EGG")
    # axs[0].plot(true_degg, color="g", label="DEGG")
    # axs[1].plot(estimated_egg, color="r", label="EGG")
    # axs[1].plot(estimated_degg, color="g", label="DEGG")

    # for ax in axs:
    #     ax.plot(1 * regions, color="g")
    #     ax.axhline(0, *ax.get_xlim(), color="k")
    #     ax.set_ylabel("Amplitude")
    #     ax.set_xlabel("Sample")
    #     ax.legend(loc=1)
    #     ax.legend(
    #         handles=[gci_patch, goi_patch, sqpeak_patch, egg_patch, degg_patch], loc=1
    #     )
    # plt.subplots_adjust(
    #     top=0.92, bottom=0.09, left=0.08, right=0.95, hspace=0.1, wspace=0.2
    # )
    # mng = plt.get_current_fig_manager()
    # mng.window.showMaximized()
    # plt.show()

    # visualization ends

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
    for tr, tr_degg, reg in zip(true_peaks_list, true_degg_list, true_regions):
        for i in range(1, tr.shape[0], 2):
            count += 1
            metrics["CQ_true"] += (tr[i + 1] - tr[i]) / (tr[i + 1] - tr[i - 1])
            metrics["OQ_true"] += (tr[i] - tr[i - 1]) / (tr[i + 1] - tr[i - 1])

        # temp = extract_speed_quotient(tr_degg, tr)
        temp = extract_speed_quotient_egg(true_egg, tr, reg)
        metrics["SQ_true"] += temp[0]
        sq_count += temp[1]

    metrics["SQ_true"] /= sq_count
    metrics["CQ_true"] /= count
    metrics["OQ_true"] /= count

    count = 0
    sq_count = 0
    for er, er_degg, reg in zip(
        estimated_peaks_list, estimated_degg_list, estimated_regions
    ):
        for i in range(1, er.shape[0], 2):
            count += 1
            metrics["CQ_estimated"] += (er[i + 1] - er[i]) / (er[i + 1] - er[i - 1])
            metrics["OQ_estimated"] += (er[i] - er[i - 1]) / (er[i + 1] - er[i - 1])

        # temp = extract_speed_quotient(er_degg, er)
        temp = extract_speed_quotient_egg(estimated_egg, er, reg)
        metrics["SQ_estimated"] += temp[0]
        sq_count += temp[1]
    metrics["SQ_estimated"] /= sq_count
    metrics["CQ_estimated"] /= count
    metrics["OQ_estimated"] /= count

    metrics.update(fnames)
    return metrics
