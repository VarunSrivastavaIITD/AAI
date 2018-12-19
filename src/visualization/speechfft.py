import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from utils import minmaxnormalize, detrend, smooth

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
        default="../data/bdl_clean/test/speech",
        help="folder containing speech files",
    )
    parser.add_argument(
        "-ef",
        "--egg-folder",
        dest="eggfolder",
        type=str,
        default="../data/bdl_clean/test/egg",
        help="folder containing egg files",
    )
    parser.add_argument(
        "-rf",
        "--reconstructedegg-folder",
        default="../data/bdl_clean/test/reconstructed_regressed_cos",
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
    speech = smooth(speech, 21)
    true_egg = np.load(true_egg_path)
    true_egg = minmaxnormalize(true_egg)

    if args.detrend:
        _, true_egg = detrend(None, true_egg)

    estimated_egg = np.load(estimated_egg_path)
    estimated_egg = minmaxnormalize(estimated_egg)

    srange = slice(0, speech.shape[0])
    srange = slice(10700, 15400)
    speech = speech[srange]

    plt.figure()
    plt.subplot(211)
    plt.title("Speech")
    plt.plot(speech, "k", label="Speech Waveform")
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")

    dft = np.fft.rfft(speech)
    freqs = np.fft.rfftfreq(np.size(speech, 0), 1 / 16e3)

    assert freqs.shape == dft.shape

    plt.subplot(212)
    plt.title("Fourier Spectra")
    plt.gca().semilogx(freqs, np.abs(dft) ** 2, "b", label="DFT")
    # plt.plot(freqs, np.abs(dft), "b", label="DFT")
    plt.xlabel("Frequency")
    plt.ylabel("PSD")

    plt.subplots_adjust(
        top=0.962, bottom=0.061, left=0.041, right=0.99, hspace=0.204, wspace=0.2
    )
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()

    plt.show()


if __name__ == "__main__":
    main()
