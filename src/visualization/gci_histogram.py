import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter


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
        "-m",
        "--metric",
        type=str.lower,
        choices=["idr", "mr", "far", "ida"],
        default="idr",
        help="quantity to plot histogram on",
    )
    parser.add_argument("file", type=str, help="file to visualize")
    parser.add_argument(
        "-s",
        "--save",
        dest="save",
        nargs="?",
        help="specifies if and where to save the output figure, specifying without an output file will save in figure.png",
        const="figure.png",
    )
    parser.add_argument(
        "-no",
        "--no-show",
        dest="noshow",
        action="store_true",
        help="whether not to display the generated figure (False by default)",
    )
    return parser.parse_args()


def main():
    args = parse()

    if args.noshow and args.save is None:
        print("Selecting noshow and no output is a no-op, exiting")
        sys.exit(0)

    with open(args.file) as f:
        data = f.read().splitlines()

    data = (d.split()[1:-1] for d in data)
    data = [
        float(v.strip("%"))
        for d in data
        for k, v in zip(d[::2], d[1::2])
        if k.lower()[:-1] == args.metric
    ]

    data = np.array(data, dtype=np.float).ravel()

    fig, ax = plt.subplots(tight_layout=True)
    N, bins, patches = ax.hist(data, label=args.metric.upper())

    mean = np.mean(data)
    std = np.std(data)

    title_dict = {
        "idr": "Identification Rate",
        "mr": "Miss Rate",
        "far": "False Identification Rate",
        "ida": "Identification Accuracy",
    }

    ax.set_title(title_dict[args.metric])
    ax.text(
        0.5,
        0.95,
        "$\\mu={:3.2f}, \\sigma={:3.2f}$".format(mean, std),
        horizontalalignment="center",
        verticalalignment="center",
        transform=ax.transAxes,
    )
    ax.set_xlabel("bins")
    ax.set_ylabel("sample percentages")
    ax.legend()

    # plt.subplots_adjust(
    #     top=0.72, bottom=0.08, left=0.053, right=0.981, hspace=0.116, wspace=0.2
    # )

    # fracs = N / N.max()

    # # we need to normalize the data to 0..1 for the full range of the colormap
    # norm = colors.Normalize(fracs.min(), fracs.max())

    # # Now, we'll loop through our objects and set the color of each accordingly
    # for thisfrac, thispatch in zip(fracs, patches):
    #     color = plt.cm.viridis(norm(thisfrac))
    #     thispatch.set_facecolor(color)

    ax.yaxis.set_major_formatter(PercentFormatter(xmax=len(data)))

    if not args.noshow:
        mng = plt.get_current_fig_manager()
        mng.window.showMaximized()
        plt.show()

    if args.save:
        fig.savefig(args.save)


if __name__ == "__main__":
    main()
