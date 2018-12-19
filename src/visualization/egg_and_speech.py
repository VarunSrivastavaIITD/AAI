import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
from utils import minmaxnormalize, detrend

plt.switch_backend("QT5Agg")
plt.rc("text", usetex=True)
plt.rc("font", family="serif")


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-sf",
        "--speech-folder",
        dest="speechfolder",
        type=str,
        default="/media/varun/Home/varun/CNN/ECNN/GCI/DotModel/data/bdl_clean/test/speech",
        help="folder containing speech files",
    )
    parser.add_argument(
        "-ef",
        "--egg-folder",
        dest="eggfolder",
        type=str,
        default="/media/varun/Home/varun/CNN/ECNN/GCI/DotModel/data/bdl_clean/test/egg",
        help="folder containing egg files",
    )
    parser.add_argument(
        "-rf",
        "--reconstructedegg-folder",
        default="/media/varun/Home/varun/CNN/ECNN/GCI/DotModel/data/bdl_clean/test/reconstructed_regressed_cos",
        type=str,
        dest="geneggfolder",
        help="folder for reconstructed files",
    )
    parser.add_argument(
        "-d", "--detrend", type=bool, default=True, help="detrend ground truth egg"
    )
    parser.add_argument(
        "-f",
        "--file",
        dest="file",
        default="arctic_a0014.npy",
        type=str,
        help="File to visualize",
    )
    return parser.parse_args()


def main():
    args = parse()
    file = args.file
    speech_path = os.path.join(args.speechfolder, file)
    true_egg_path = os.path.join(args.eggfolder, file)
    estimated_egg_path = os.path.join(args.geneggfolder, file)

    speech = np.load(speech_path)
    speech = minmaxnormalize(speech)
    true_egg = np.load(true_egg_path)
    true_egg = minmaxnormalize(true_egg)

    if args.detrend:
        _, true_egg = detrend(None, true_egg)

    estimated_egg = np.load(estimated_egg_path)
    estimated_egg = minmaxnormalize(estimated_egg)

    plt.figure()
    ax = plt.subplot(211)
    plt.plot(speech, "k", label="Speech Waveform")
    plt.ylabel("Amplitude")

    plt.subplot(212, sharex=ax)
    plt.plot(true_egg, "k", label="Ground Truth EGG")
    plt.plot(estimated_egg, "b", label="Estimated EGG")
    plt.xlabel("Sample Number")
    plt.ylabel("Amplitude")

    plt.show()


if __name__ == "__main__":
    main()
