import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
from utils import minmaxnormalize, detrend

plt.switch_backend("QT5Agg")
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
        "-sf",
        "--speech-folder",
        dest="speechfolder",
        type=str,
        # default="/media/varun/Home/varun/CNN/ECNN/GCI/DotModel/data/bdl_clean/test/speech",
        default="/media/varun/Home/varun/CNN/ECNN/perfect_files/speech",
        help="folder containing speech files",
    )
    parser.add_argument(
        "-ef",
        "--egg-folder",
        dest="eggfolder",
        type=str,
        # default="/media/varun/Home/varun/CNN/ECNN/GCI/DotModel/data/bdl_clean/test/egg",
        default="/media/varun/Home/varun/CNN/ECNN/perfect_files/egg_detrended",
        help="folder containing egg files",
    )
    parser.add_argument(
        "-rf",
        "--reconstructedegg-folder",
        # default="/media/varun/Home/varun/CNN/ECNN/GCI/DotModel/data/bdl_clean/test/reconstructed_regressed_cos",
        default="/media/varun/Home/varun/CNN/ECNN/perfect_files/egg_reconstructed",
        type=str,
        dest="geneggfolder",
        help="folder for reconstructed files",
    )
    parser.add_argument(
        "-d", "--detrend", type=bool, default=False, help="detrend ground truth egg"
    )
    parser.add_argument(
        "-f",
        "--file",
        dest="file",
        default="arctic_a0015.npy",
        type=str,
        help="File to visualize",
    )
    return parser.parse_args()


def main():
    args = parse()
    fname = args.file
    speech_path = os.path.join(args.speechfolder, fname)
    true_egg_path = os.path.join(args.eggfolder, fname)
    estimated_egg_path = os.path.join(args.geneggfolder, fname)

    speech = np.load(speech_path)
    speech = minmaxnormalize(speech)
    true_egg = np.load(true_egg_path)
    true_egg = minmaxnormalize(true_egg)

    if args.detrend:
        _, true_egg = detrend(None, true_egg)

    estimated_egg = np.load(estimated_egg_path)
    estimated_egg = minmaxnormalize(estimated_egg)

    # region = slice(11000, 12000)
    region = slice(0, len(speech))

    plt.figure()
    fig = plt.gcf()
    ax = plt.subplot(211)
    plt.plot(speech[region], "k", label="Speech Waveform")
    plt.ylabel("Amplitude")

    plt.subplot(212, sharex=ax)
    plt.plot(true_egg[region], "k", label="Ground Truth EGG")
    plt.plot(estimated_egg[region], "b", label="Estimated EGG")
    plt.xlabel("Sample Number")
    plt.ylabel("Amplitude")

    plt.subplots_adjust(
        top=0.962, bottom=0.087, left=0.057, right=0.981, hspace=0.212, wspace=0.2
    )

    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()

    plt.show()
    # fig.savefig("images/speech_egg.png")


if __name__ == "__main__":
    main()
