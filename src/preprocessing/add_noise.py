import numpy as np
from glob import glob
import argparse
import os
from scipy.io.wavfile import read as wavread


def energy(signal):
    return np.sqrt(np.sum(signal ** 2) / signal.size)


def add_noise(y: np.ndarray, noise: np.ndarray, snr: float):
    temp = np.linalg.norm(y)
    y = y / temp
    sigp = energy(y)
    sigma = (10 ** (-0.05 * snr)) * sigp
    noisep = energy(noise)

    noise = noise * (sigma / noisep)

    # print(20 * np.log10(sigp / energy(noise)))
    # noisep1 = energy(noise)
    # snr_out = 20 * np.log10(sigp / noisep1)
    return (y + noise) * temp


def main(args):
    input_folder = args.input_folder
    output_folder = args.output_folder
    os.makedirs(output_folder, exist_ok=True)
    noise_type = args.noise.lower()
    babble = wavread(
        os.path.join("/media/varun/Home/varun/CNN", "cmu/CMU/prathoshscripts/noise.wav")
    )[1]
    babble = babble / np.linalg.norm(babble)
    snr = args.snr

    input_files = glob(os.path.join(input_folder, "*.npy"))

    for file in input_files:
        print(file)
        fname = os.path.basename(file).split(".")[0]
        y: np.ndarray = np.load(file)
        if noise_type == "white":
            noise = np.random.standard_normal(y.size)
        else:
            np.random.shuffle(babble)
            noise = babble[: y.size]
        y = add_noise(y, noise, snr)
        np.save(os.path.join(output_folder, fname), y)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input-folder",
        dest="input_folder",
        default="/media/varun/Home/varun/cmu/CMU_DATA/bdl_speech_raw",
        help="Input folder containing single channel npy files",
    )
    parser.add_argument(
        "-o", "--output-folder", dest="output_folder", help="Output folder"
    )
    parser.add_argument(
        "-n",
        "--noise",
        default="white",
        choices=["white", "babble"],
        help="Type of noise to add",
    )
    parser.add_argument(
        "-s", "--snr", default=0, type=int, help="SNR of noise to be added"
    )
    args = parser.parse_args()
    main(args)
