import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
from utils import minmaxnormalize, detrend
from pprint import pprint

plt.switch_backend("qt5agg")
plt.rc("text", usetex=True)
plt.rc("font", family="serif")
SMALL_SIZE = 22
MEDIUM_SIZE = 24
BIGGER_SIZE = 28

plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
plt.rc("axes", titlesize=BIGGER_SIZE)  # fontsize of the axes title
plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-ef",
        "--egg-folder",
        dest="eggfolder",
        type=str,
        default="../../../GCI/Irish/resampled_merged/egg",
        help="folder containing egg files",
    )
    parser.add_argument(
        "-d", "--detrend", type=bool, default=False, help="detrend ground truth egg"
    )
    parser.add_argument(
        "-f",
        "--file",
        dest="file",
        default="1_ mnorm.npy",
        type=str,
        help="File to visualize",
    )
    return parser.parse_args()


def main():
    args = parse()
    fname = args.file
    true_egg_path = os.path.join(args.eggfolder, fname)

    true_egg = np.load(true_egg_path)
    true_egg = minmaxnormalize(true_egg)

    if args.detrend:
        _, true_egg = detrend(None, true_egg)

    degg = np.gradient(true_egg, edge_order=2)

    # region = slice(11000, 12000)
    region = slice(8000, len(true_egg))

    plt.figure()
    fig = plt.gcf()
    ax1 = plt.subplot(211)
    plt.plot(true_egg[region], "b", label="EGG Waveform")
    plt.ylabel("Amplitude")

    ax2 = plt.subplot(212, sharex=ax1)
    plt.plot(degg[region], "b", label="DEGG waveform")
    plt.xlabel("Sample Number")
    plt.ylabel("Amplitude")

    plt.subplots_adjust(
        top=0.962, bottom=0.08, left=0.053, right=0.981, hspace=0.116, wspace=0.2
    )

    # ax1.spines["bottom"].set_visible(False)
    # ax2.spines["top"].set_visible(False)
    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_label_coords(-0.037, 0.5)
    ax2.get_yaxis().set_label_coords(-0.037, 0.5)
    ax1.set_xlim([19715, 20015])

    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()

    plt.show()


if __name__ == "__main__":
    main()
