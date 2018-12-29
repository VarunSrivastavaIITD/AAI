import os

from extract_metrics import extract_gci_metrics
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
        default="data/test/egg",
        help="folder containing egg files",
    )
    parser.add_argument(
        "-rf",
        "--reconstructedegg-folder",
        dest="geneggfolder",
        type=str,
        default="data/test/reconstructed",
        help="folder for reconstructed files",
    )
    parser.add_argument(
        "-d", "--detrend", type=bool, default=False, help="detrend ground truth egg"
    )
    parser.add_argument(
        "-n",
        "--num-files",
        dest="numfiles",
        default=1000000,
        type=int,
        help="number of files to compute metrics on",
    )
    return parser.parse_args()


def main():
    args = parse()
    files = natsort.natsorted(os.listdir(args.geneggfolder))
    true_egg_path = args.eggfolder
    estimated_egg_path = args.geneggfolder

    mean_idr = 0
    mean_msr = 0
    mean_far = 0
    mean_ida = 0
    count = 0
    for f in [os.path.basename(f) for f in files]:
        result = extract_gci_metrics(
            os.path.join(true_egg_path, f),
            os.path.join(estimated_egg_path, f),
            args.detrend,
        )
        print(f, result)

        mean_idr += result["identification_rate"]
        mean_msr += result["miss_rate"]
        mean_far += result["false_alarm_rate"]
        mean_ida += result["identification_accuracy"]
        count += 1

        if count == args.numfiles:
            break

    mean = {
        "identification_rate": mean_idr / count,
        "miss_rate": mean_msr / count,
        "false_alarm_rate": mean_far / count,
        "identification_accuracy": mean_ida / count,
    }

    print("Mean", mean)


def pmain():
    args = parse()
    files = natsort.natsorted(os.listdir(args.geneggfolder))
    true_egg_path = args.eggfolder
    estimated_egg_path = args.geneggfolder

    sum_idr = 0
    sum_msr = 0
    sum_far = 0
    sum_ida = 0
    count = 0
    workers = 8
    futures = []

    with cf.ProcessPoolExecutor(max_workers=workers) as executor:
        for f in [os.path.basename(f) for f in files]:

            futures.append(
                executor.submit(
                    extract_gci_metrics,
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
                "{}: IDR: {:4.2f}%\t MR: {:4.2f}%\t FAR: {:4.2f}%\t IDA: {:.3f} ms".format(
                    os.path.basename(res["true_file"]),
                    float(res["identification_rate"]) * 100,
                    float(res["miss_rate"]) * 100,
                    float(res["false_alarm_rate"]) * 100,
                    float(res["identification_accuracy"]) * 1000,
                )
            )

            sum_idr += res["identification_rate"]
            sum_msr += res["miss_rate"]
            sum_far += res["false_alarm_rate"]
            sum_ida += res["identification_accuracy"]

        mean = {
            "identification_rate": sum_idr / count,
            "miss_rate": sum_msr / count,
            "false_alarm_rate": sum_far / count,
            "identification_accuracy": sum_ida / count,
        }

        print()
        print(" ".join("{}:{:5.4}".format(k, v) for k, v in mean.items()), sep="\t")


if __name__ == "__main__":
    main()
