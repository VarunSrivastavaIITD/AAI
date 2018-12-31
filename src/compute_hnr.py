import argparse
import os
import warnings
import numpy as np
import natsort
import matplotlib.pyplot as plt

from extract_metrics import (
    detectgenwaveletgci,
    detectgroundwaveletgci,
    extract_goi,
    groundeggfilter,
    geneggfilter,
    detect_voiced_region,
    apply_region_to_positions,
)
from utils import lowpass, detrend, positions2onehot, onehot2positions, minmaxnormalize
from hnr import harmonic_to_noise_ratio, extract_hnr
from scipy.ndimage import label, find_objects

warnings.filterwarnings("ignore")
plt.switch_backend("qt5agg")


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-ef",
        "--egg-folder",
        dest="eggfolder",
        type=str,
        default="data/slt_clean/test/egg_detrended",
        help="folder containing egg filies",
    )
    parser.add_argument(
        "-rf",
        "--reconstructedegg-folder",
        dest="geneggfolder",
        type=str,
        default="data/slt_clean/test/egg_reconstructed",
        help="folder for reconstructed files",
    )
    parser.add_argument(
        "-d", "--detrend", type=bool, default=True, help="detrend ground truth egg"
    )
    parser.add_argument(
        "-n",
        "--num-files",
        dest="numfiles",
        default=1000000,
        type=int,
        help="number of files to compute metrics on",
    )
    parser.add_argument(
        "-f", "--files", nargs="*", help="file[s] to compute metrics on"
    )
    parser.add_argument(
        "-l",
        "--lowpass",
        action="store_true",
        help="lowpass filter the ground truth egg and estimated egg",
    )
    return parser.parse_args()


def main():
    args = parse()
    files = natsort.natsorted(os.listdir(args.geneggfolder))
    true_egg_path = args.eggfolder
    estimated_egg_path = args.geneggfolder

    if args.files:
        files = set(args.files).intersection({os.path.basename(f) for f in files})

    count = 0

    true_hnrs = []
    est_hnrs = []
    for f in [os.path.basename(f) for f in files]:
        if count == args.numfiles:
            break

        count += 1

        true_egg = np.load(os.path.join(true_egg_path, f))
        estimated_egg = np.load(os.path.join(estimated_egg_path, f))

        if args.detrend:
            _, true_egg = detrend(None, true_egg)
        if args.lowpass:
            true_egg = lowpass(true_egg)
            estimated_egg = lowpass(estimated_egg)

        true_gci = detectgroundwaveletgci(true_egg)
        estimated_gci = detectgenwaveletgci(estimated_egg)

        true_egg, estimated_egg = groundeggfilter(true_egg), geneggfilter(estimated_egg)

        regions = detect_voiced_region(
            true_egg, estimated_egg, power_threshold=0.01, return_regions=True
        )

        segmented_regions = [obj[0] for obj in find_objects(label(regions)[0])]

        true_hnr = extract_hnr(
            true_egg, positions2onehot(true_gci, regions.shape), segmented_regions
        )
        estimated_hnr = extract_hnr(
            estimated_egg,
            positions2onehot(estimated_gci, regions.shape),
            segmented_regions,
        )
        true_hnrs.append(true_hnr)
        est_hnrs.append(estimated_hnr)
        print("True HNR: {:5.3}\nEstimated HNR: {:5.3}".format(true_hnr, estimated_hnr))

    true_hnrs = np.array(true_hnrs)
    est_hnrs = np.array(est_hnrs)

    c1 = true_hnrs > 0
    c2 = est_hnrs > 0
    c3 = ~np.isnan(true_hnrs)
    c4 = ~np.isnan(est_hnrs)

    idxs = c1 & c2 & c3 & c4

    true_hnrs = true_hnrs[idxs]
    est_hnrs = est_hnrs[idxs]

    av_true_hnr = np.mean(true_hnrs)
    av_est_hnr = np.mean(est_hnrs)
    print(
        "Av True HNR: {:5.3}\nAv Estimated HNR: {:5.3}".format(av_true_hnr, av_est_hnr)
    )


if __name__ == "__main__":
    main()
