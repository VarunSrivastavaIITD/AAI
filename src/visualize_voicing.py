import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from utils import detrend


import warnings

warnings.filterwarnings("ignore")
plt.switch_backend("QT5Agg")


def _get_signal_power(x, window):
    power = np.convolve(x ** 2, window / window.sum(), mode="same")
    return power


def _get_window(window_len=10, window="flat"):
    if window == "flat":  # average
        w = np.ones(window_len, "d")
    else:
        w = eval("np." + window + "(window_len)")

    return w


def detect_voiced_region(true_egg, reconstructed_egg, power_threshold=0.05):

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


def get_envelope(true_egg):
    true_scaler = pd.Series(np.abs(true_egg)).nlargest(100).median()

    true_egg = true_egg / true_scaler

    window = _get_window(window_len=501, window="hanning")
    power = _get_signal_power(true_egg, window)

    return power


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f",
        "--file",
        default="arctic_b0372.npy",
        dest="filename",
        type=str,
        help="file to visualize",
    )
    parser.add_argument(
        "-ef",
        "--egg-folder",
        dest="groundpath",
        type=str,
        default="../egg/egg_detrended_voiced",
        help="path to ground truth egg files",
    )
    parser.add_argument(
        "-rf",
        "--reconstructedegg-folder",
        dest="generatedpath",
        type=str,
        default="../egg/egg_reconstructed_voiced",
        help="path to generated truth egg files",
    )
    parser.add_argument(
        "-d", "--detrend", type=bool, default=False, help="detrend ground truth egg"
    )
    return parser.parse_args()


def main():
    args = parse()
    fname = args.filename
    eground = np.load(os.path.join(args.groundpath, fname))
    if args.detrend:
        _, eground = detrend(None, eground)

    power = get_envelope(eground)

    fig = plt.figure()
    plt.title("Voicing Visualization")
    plt.plot(eground / np.max(np.abs(eground)), "r", label="ground truth")
    plt.plot(power, "g", label="envelope")
    plt.xlabel("sample")
    plt.ylabel("amplitude")

    plt.legend()

    fig.tight_layout()
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
    plt.show()


if __name__ == "__main__":
    main()
