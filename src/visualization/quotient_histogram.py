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
        choices=["cq", "sq", "oq"],
        default="cq",
        help="quantity to plot histogram on",
    )
    parser.add_argument(
        "file",
        type=str,
        help="file to visualize",
        default="/media/varun/Home/varun/CNN/ECNN/metric_files/childers_0_babble_bdl_0_babble/Quotients.txt",
        nargs="?",
    )
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

    data = (d.split() for d in data)

    metrics_idxs = {"cq": (2, 4), "oq": (6, 8), "sq": (10, 12)}
    idxs = metrics_idxs[args.metric]
    data = [[float(d[idxs[0]]), float(d[idxs[1]])] for d in data if len(d) == 13]

    data = np.array(data, dtype=np.float)

    fig, ax = plt.subplots(tight_layout=True)
    N_0, bins_0, patches_0 = ax.hist(
        data[:, 0], facecolor="b", label="true {}".format(args.metric.upper())
    )
    N_1, bins_1, patches_1 = ax.hist(
        data[:, 1], facecolor="g", label="estimated {}".format(args.metric.upper())
    )

    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)

    title_dict = {"cq": "Closed Quotient", "oq": "Open Quotient", "sq": "Skew Quotient"}

    ax.set_title(title_dict[args.metric])
    ax.text(
        0.5,
        0.95,
        "\\noindent $\\mu_t={0:3.2f}, \\sigma_t={2:3.2f}$ \\newline $\\mu_e={1:3.2f}, \\sigma_e={3:3.2f}$".format(
            *mean, *std
        ),
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
