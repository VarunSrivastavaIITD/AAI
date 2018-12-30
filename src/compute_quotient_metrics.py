import os

from extract_metrics import extract_quotient_metrics
import concurrent.futures as cf
import warnings
import argparse
import natsort

warnings.filterwarnings("ignore")


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-ef",
        "--egg-folder",
        dest="eggfolder",
        type=str,
        default="data/slt_clean/test/egg_detrended",
        help="folder containing egg files",
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
        "-d", "--detrend", type=bool, default=False, help="detrend ground truth egg"
    )
    parser.add_argument(
        "-n",
        "--num-files",
        dest="numfiles",
        default=1,
        type=int,
        help="number of files to compute metrics on",
    )
    parser.add_argument(
        "-f", "--files", nargs="*", help="file[s] to compute metrics on"
    )
    return parser.parse_args()


def main():
    args = parse()
    files = natsort.natsorted(os.listdir(args.geneggfolder))
    true_egg_path = args.eggfolder
    estimated_egg_path = args.geneggfolder

    sum_cq_true = 0
    sum_oq_true = 0
    sum_sq_true = 0
    sum_cq_estimated = 0
    sum_oq_estimated = 0
    sum_sq_estimated = 0

    if args.files:
        files = set(args.files).intersection({os.path.basename(f) for f in files})

    count = 0

    for f in [os.path.basename(f) for f in files]:
        if count == args.numfiles:
            break

        res = extract_quotient_metrics(
            os.path.join(true_egg_path, f),
            os.path.join(estimated_egg_path, f),
            args.detrend,
        )

        print(
            "{}: CQ_true: {:4.3}\t CQ_estimated: {:4.3}\t OQ_true: {:4.3}\t OQ_estimated: {:4.3}\t SQ_true: {:4.3}\t SQ_estimated: {:4.3}".format(
                os.path.basename(res["true_file"]),
                float(res["CQ_true"]),
                float(res["CQ_estimated"]),
                float(res["OQ_true"]),
                float(res["OQ_estimated"]),
                float(res["SQ_true"]),
                float(res["SQ_estimated"]),
            )
        )

        sum_cq_true += res["CQ_true"]
        sum_oq_true += res["OQ_true"]
        sum_sq_true += res["SQ_true"]
        sum_cq_estimated += res["CQ_estimated"]
        sum_oq_estimated += res["OQ_estimated"]
        sum_sq_estimated += res["SQ_estimated"]

        count += 1

    mean = {
        "CQ_true": sum_cq_true / count,
        "CQ_estimated": sum_cq_estimated / count,
        "OQ_true": sum_oq_true / count,
        "OQ_estimated": sum_oq_estimated / count,
        "SQ_true": sum_sq_true / count,
        "SQ_estimated": sum_sq_estimated / count,
    }

    print()
    print(" ".join("{}:{:5.4}".format(k, v) for k, v in mean.items()), sep="\t")


def pmain():
    args = parse()
    files = natsort.natsorted(os.listdir(args.geneggfolder))
    true_egg_path = args.eggfolder
    estimated_egg_path = args.geneggfolder

    sum_cq_true = 0
    sum_oq_true = 0
    sum_sq_true = 0
    sum_cq_estimated = 0
    sum_oq_estimated = 0
    sum_sq_estimated = 0
    count = 0
    workers = 8
    futures = []

    if args.files:
        files = set(args.files).intersection({os.path.basename(f) for f in files})

    with cf.ProcessPoolExecutor(max_workers=workers) as executor:
        for f in [os.path.basename(f) for f in files]:
            futures.append(
                executor.submit(
                    extract_quotient_metrics,
                    os.path.join(true_egg_path, f),
                    os.path.join(estimated_egg_path, f),
                    args.detrend,
                )
            )
            count += 1
            if count == args.numfiles:
                break

        for comp_future in cf.as_completed(futures):
            res = comp_future.result()
            print(
                "{}: CQ_true: {:4.3}\t CQ_estimated: {:4.3}\t OQ_true: {:4.3}\t OQ_estimated: {:4.3}\t SQ_true: {:4.3}\t SQ_estimated: {:4.3}".format(
                    os.path.basename(res["true_file"]),
                    float(res["CQ_true"]),
                    float(res["CQ_estimated"]),
                    float(res["OQ_true"]),
                    float(res["OQ_estimated"]),
                    float(res["SQ_true"]),
                    float(res["SQ_estimated"]),
                )
            )

            sum_cq_true += res["CQ_true"]
            sum_oq_true += res["OQ_true"]
            sum_sq_true += res["SQ_true"]
            sum_cq_estimated += res["CQ_estimated"]
            sum_oq_estimated += res["OQ_estimated"]
            sum_sq_estimated += res["SQ_estimated"]

        mean = {
            "CQ_true": sum_cq_true / count,
            "CQ_estimated": sum_cq_estimated / count,
            "OQ_true": sum_oq_true / count,
            "OQ_estimated": sum_oq_estimated / count,
            "SQ_true": sum_sq_true / count,
            "SQ_estimated": sum_sq_estimated / count,
        }

        print()
        print(" ".join("{}:{:5.4}".format(k, v) for k, v in mean.items()), sep="\t")


if __name__ == "__main__":
    pmain()
