import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
from utils import minmaxnormalize, detrend

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
    true_egg = true_egg[region]
    degg = degg[region]

    plt.figure()
    fig = plt.gcf()
    ax1 = plt.subplot(211)
    plt.plot(true_egg, "b", label="EGG Waveform")
    plt.ylabel("Amplitude")

    ax2 = plt.subplot(212, sharex=ax1)
    plt.plot(degg, "g", label="DEGG waveform")
    plt.xlabel("Sample Number")
    plt.ylabel("Amplitude")

    # Timepoints in region shifted coordinates
    t0 = 19750  # start of EGG cycle, GOI
    t4 = 19853  # end of EGG cycle
    t1 = t0 + np.argmax(true_egg[t0:t4])  # t0=19750 => t1=19760 EGG peak
    t2 = t1 + np.argmin(degg[t1:t4])  # t0=19750 => t1=19760 => t2=19834 GCI
    t5 = 19908  # start of new egg cycle
    t9 = 20010  # end of new egg cycle
    t6 = t5 + np.argmax(true_egg[t5:t9])  # EGG peak
    t8 = t6 + np.argmin(degg[t6:t9])  # GCI

    y1max = 1.0
    y2min = -0.05

    # Epochs

    goi_ax1_1 = (t0, true_egg[t0])
    goi_ax1_2 = (t5, true_egg[t5])

    goi_ax2_1 = (t0, degg[t0])
    goi_ax2_2 = (t5, degg[t5])

    gci_ax1_1 = (t2, true_egg[t2])
    gci_ax1_2 = (t8, true_egg[t8])

    gci_ax2_1 = (t2, degg[t2])
    gci_ax2_2 = (t8, degg[t8])

    eggpeak_ax1_1 = (t1, true_egg[t1])
    eggpeak_ax1_2 = (t6, true_egg[t6])

    # eggstart_ax1_1 = (t0, true_egg[t0])
    # eggstart_ax1_2 = (t5, true_egg[t5])

    eggend_ax1_1 = (t4, true_egg[t4])
    eggend_ax1_2 = (t9, true_egg[t9])

    # Epoch Markers
    goi_ax1 = np.array([goi_ax1_1, goi_ax1_2])
    gci_ax1 = np.array([gci_ax1_1, gci_ax1_2])

    goi_ax2 = np.array([goi_ax2_1, goi_ax2_2])
    gci_ax2 = np.array([gci_ax2_1, gci_ax2_2])

    eggpeak_ax1 = np.array([eggpeak_ax1_1, eggpeak_ax1_2])

    # eggstart_ax1 = np.array([eggstart_ax1_1, eggstart_ax1_2])

    eggend_ax1 = np.array([eggend_ax1_1, eggend_ax1_2])

    # GOI scatter
    ax1.scatter(
        goi_ax1[:, 0], goi_ax1[:, 1], c="tab:pink", marker="x", s=100, label="GOI"
    )
    ax2.scatter(
        goi_ax2[:, 0], goi_ax2[:, 1], c="tab:pink", marker="x", s=100, label="GOI"
    )

    # GCI Scatter
    ax1.scatter(gci_ax1[:, 0], gci_ax1[:, 1], c="r", marker="*", s=100, label="GCI")
    ax2.scatter(gci_ax2[:, 0], gci_ax2[:, 1], c="r", marker="*", s=100, label="GCI")

    # EGG Peak Scatter
    ax1.scatter(
        eggpeak_ax1[:, 0],
        eggpeak_ax1[:, 1],
        # c="g",
        marker="s",
        s=100,
        label="EGG Peak",
        facecolors="none",
        edgecolors="g",
    )

    # EGG Start Scatter
    # ax1.scatter(
    #     eggstart_ax1[:, 0],
    #     eggstart_ax1[:, 1],
    #     # c="g",
    #     marker="^",
    #     s=100,
    #     label="EGG Cycle Start",
    #     facecolors="none",
    #     edgecolors="b",
    # )

    # EGG End Scatter
    ax1.scatter(
        eggend_ax1[:, 0],
        eggend_ax1[:, 1],
        # c="g",
        marker="v",
        s=100,
        label="EGG Cycle End",
        facecolors="none",
        edgecolors="tab:purple",
    )

    # Across axes vertical lines

    # Annotations
    yoffset = 0.1
    xoffset = 3
    ax1.text(
        t0 + xoffset,
        true_egg[t0] + yoffset,
        r"$t_0$",
        ha="center",
        va="center",
        transform=ax1.transData,
        fontsize=30,
    )
    ax1.text(
        t1 + xoffset,
        true_egg[t1] + yoffset,
        r"$t_1$",
        ha="center",
        va="center",
        transform=ax1.transData,
        fontsize=30,
    )
    ax1.text(
        t2 + xoffset,
        true_egg[t2] + yoffset,
        r"$t_2$",
        ha="center",
        va="center",
        transform=ax1.transData,
        fontsize=30,
    )
    ax1.text(
        t4 + xoffset,
        true_egg[t4] + yoffset,
        r"$t_4$",
        ha="center",
        va="center",
        transform=ax1.transData,
        fontsize=30,
    )

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

    ax1.legend()
    ax2.legend()
    plt.show()


if __name__ == "__main__":
    main()
