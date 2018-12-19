import os
import random

import numpy as np
import torch as th
import torch.nn as nn
from torch.utils.data import DataLoader

from model import Discriminator, SpeechEggEncoder
from utils import Saver
import argparse
from torch_utils import AudioFileDataset
from utils import detrend
from glob import glob
import warnings

warnings.filterwarnings("ignore")

# INFO: Set random seeds
q_ = 4
np.random.seed(q_)
th.manual_seed(q_)
th.cuda.manual_seed_all(q_)
random.seed(q_)


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f",
        "--model-file",
        dest="modelfile",
        type=str,
        default="epoch_100.pt",
        help="model file to use for inference",
    )
    parser.add_argument(
        "-m",
        "--model-folder",
        dest="modelfolder",
        type=str,
        default="",
        help="model folder containing saved models",
    )
    parser.add_argument(
        "-sf",
        "--speech-folder",
        dest="speechfolder",
        type=str,
        default="",
        help="folder containing speech files",
    )
    parser.add_argument(
        "-ef",
        "--egg-folder",
        dest="eggfolder",
        type=str,
        default="",
        help="folder containing egg files",
    )
    parser.add_argument(
        "-rf",
        "--reconstructed-folder",
        dest="outputfolder",
        type=str,
        default="",
        help="folder for output files",
    )
    parser.add_argument(
        "-g",
        "--use-gpu",
        dest="cuda",
        type=bool,
        default=False,
        help="use GPU for inference",
    )
    parser.add_argument(
        "-w", "--window", dest="window", type=int, default=200, help="window size"
    )
    parser.add_argument(
        "-s", "--stride", dest="stride", type=int, default=200, help="stride size"
    )
    return parser.parse_args()


def test(
    model_G: nn.Module,
    model_D: nn.Module,
    test_loader: DataLoader,
    use_cuda: bool = False,
):
    device = "cuda" if th.cuda.is_available() and use_cuda else "cpu"
    if use_cuda:
        model_G.to(device)
        model_D.to(device)
    model_G.eval()
    model_D.eval()

    reconstruction_loss = 0
    with th.no_grad():
        for data, egg_data, spfile, _ in test_loader:
            if use_cuda:
                data, egg_data = data.to(device), egg_data.to(device)
            data = th.squeeze(data)

            egg_data = th.squeeze(data)

            # Test model_G
            reconstructions, _ = model_G(data)
            loss_reconstruction = (egg_data * reconstructions).sum(dim=1) / (
                egg_data.norm(dim=1) * reconstructions.norm(dim=1)
            )
            loss_reconstruction = th.acos(loss_reconstruction) * 180 / np.pi
            loss_reconstruction = loss_reconstruction.mean()
            reconstruction_loss += loss_reconstruction.item()

            yield reconstructions.cpu().detach().numpy().ravel(), spfile


def main():
    args = parse()
    save_model = Saver(args.modelfolder)
    use_cuda = args.cuda

    model_G = SpeechEggEncoder()
    model_D = Discriminator()
    model_G, model_D, _, _, _, _ = save_model.load_checkpoint(
        model_G, model_D, file_name=args.modelfile
    )

    speechfiles = glob(os.path.join(args.speechfolder, "*.npy"))
    eggfiles = glob(os.path.join(args.eggfolder, "*.npy"))
    reconstruction_save_path = args.outputfolder

    test_data = AudioFileDataset(
        speechfiles, eggfiles, args.window, args.stride, transform=detrend
    )
    test_dataloader = DataLoader(test_data, 1, num_workers=4, shuffle=False)

    os.makedirs(reconstruction_save_path, exist_ok=True)

    for egg_reconstructed, f in test(
        model_G, model_D, test_dataloader, use_cuda=use_cuda
    ):
        outputfile = os.path.join(reconstruction_save_path, f[0])

        np.save(outputfile, egg_reconstructed)


if __name__ == "__main__":
    main()
