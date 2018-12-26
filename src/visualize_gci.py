import argparse
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks_cwt, medfilt
from utils import positions2onehot, detrend, smooth
from extract_metrics import geneggfilter, genegg_process, groundegg_process, groundeggfilter, detectgenwaveletgci, detectgroundwaveletgci, detect_voiced_region, extract_metrics, corrected_naylor_metrics

warnings.filterwarnings("ignore")
plt.switch_backend("qt5agg")


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f",
        "--file",
        default="arctic_a0014.npy",
        dest="filename",
        type=str,
        help="file to visualize",
    )
    parser.add_argument(
        "-ef",
        "--egg-folder",
        dest="groundpath",
        type=str,
        default="data/bdl_clean/test/egg",
        help="path to ground truth egg files",
    )
    parser.add_argument(
        "-rf",
        "--reconstructedegg-folder",
        dest="generatedpath",
        type=str,
        default="data/bdl_clean/test/reconstructed_regressed_cos_trbdl",
        help="path to generated truth egg files",
    )
    parser.add_argument(
        "-d",
        "--detrend",
        type=bool,
        default=True,
        help="detrend ground truth egg, default (True)",
    )
    return parser.parse_args()


def main():
    args = parse()
    fname = args.filename
    eground = np.load(os.path.join(args.groundpath, fname))
    egen = np.load(os.path.join(args.generatedpath, fname))

    if len(eground) > len(egen):
        eground = eground[:len(egen)]
    if args.detrend:
        _, eground = detrend(None, eground)

    rawground, rawgen = eground, egen
    voicedground, voicedgen = detect_voiced_region(rawground, rawgen)

    voicedground = voicedground / np.max(np.abs(voicedground))
    voicedgen = voicedgen / np.max(np.abs(voicedgen))

    eground, egen = detect_voiced_region(eground, egen)
    eground, egen = groundeggfilter(eground), geneggfilter(egen)

    assert len(eground) == len(voicedground)

    degground = groundegg_process(eground)
    deggen = genegg_process(egen)

    peaksposground = detectgroundwaveletgci(eground)
    peaksground = positions2onehot(peaksposground, eground.shape)

    peaksposgen = detectgenwaveletgci(egen)
    peaksgen = positions2onehot(peaksposgen, egen.shape)
    assert len(peaksgen) == len(peaksground)

    metrics = corrected_naylor_metrics(peaksposground / 16e3,
                                       peaksposgen / 16e3)

    idr = metrics["identification_rate"]
    msr = metrics["miss_rate"]
    far = metrics["false_alarm_rate"]
    ida = metrics["identification_accuracy"]
    nhits = metrics["nhits"]
    nmisses = metrics["nmisses"]
    nfars = metrics["nfars"]
    ncycles = metrics["ncycles"]

    hits = metrics["hits"]
    hits, hit_distances = zip(*hits)
    fs = 16e3
    hits = np.array(hits).squeeze() * fs
    misses = np.array(metrics["misses"]).squeeze() * fs
    fars = np.array(metrics["fars"]).squeeze() * fs
    hits = hits.astype(np.int)
    misses = misses.astype(np.int)
    fars = fars.astype(np.int)

    ax = plt.subplot(411)
    plt.plot(voicedground, "r", label="ground truth")
    plt.plot(voicedgen, "g", label="generated egg")
    ax.set_xlim([-100, len(voicedground) + 100])
    plt.gca().set_ylabel("amplitude")
    plt.title("EGG")

    plt.legend(loc=1)

    plt.subplot(412, sharex=ax)
    plt.plot(eground, "r", label="ground truth")
    plt.plot(egen, "g", label="generated egg")
    plt.gca().set_ylabel("amplitude")
    plt.title("Proc EGG")

    plt.legend(loc=1)

    plt.subplot(413, sharex=ax)
    plt.plot(degground, "r", label="ground truth")
    plt.plot(deggen, "g", label="generated egg")
    plt.gca().set_ylabel("amplitude")
    plt.title("Neg DEGG")

    plt.legend(loc=1)

    plt.subplot(414, sharex=ax)
    x = np.arange(len(peaksground))
    lax = plt.gca()

    lax.axhline(x[0], x[-1], 0, color="k")
    lax.vlines(x, 0, peaksground, color="r", label="ground truth", linewidth=2)
    lax.vlines(
        x,
        0,
        2 * positions2onehot(hits, peaksground.shape),
        color="g",
        label="hits")
    lax.vlines(
        x,
        0,
        2 * positions2onehot(misses, peaksground.shape),
        color="b",
        label="misses")
    lax.vlines(
        x,
        0,
        2 * positions2onehot(fars, peaksground.shape),
        color="m",
        label="fars")
    # plt.plot(peaksground, "r", label="ground truth", linewidth=2)
    # plt.plot(2 * peaksgen, "g", label="generated egg")

    plt.gca().set_xlabel("sample")
    plt.gca().set_ylabel("amplitude")
    plt.title("GCI")
    plt.legend(loc=1)

    plt.subplots_adjust(
        top=0.91, bottom=0.045, left=0.035, right=0.99, hspace=0.2, wspace=0.2)
    plt.suptitle(
        "{} IDR: {:2.2f} MR: {:2.2f} FAR: {:2.2f} IDA {:2.3f}\n H: {} M: {} F: {} C: {}"
        .format(
            fname,
            idr * 100,
            msr * 100,
            far * 100,
            ida * 1000,
            nhits,
            nmisses,
            nfars,
            ncycles,
        ))

    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
    plt.show()


if __name__ == "__main__":
    main()
