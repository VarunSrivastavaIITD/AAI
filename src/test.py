import random

import matplotlib.pyplot as plt
import numpy as np
import torch as th
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

from loader import create_dataloader
from model import Discriminator, SpeechEggEncoder
from utils import Saver

# INFO: Set random seeds
q_ = 42
np.random.seed(q_)
th.manual_seed(q_)
th.cuda.manual_seed_all(q_)
random.seed(q_)


def test(
    model_G: nn.Module,
    model_D: nn.Module,
    test_loader: DataLoader,
    use_cuda: bool = False,
):
    if use_cuda:
        model_G.cuda()
        model_D.cuda()
    model_G.eval()
    model_D.eval()

    batches = len(test_loader)
    reconstruction_loss = 0
    dump_array_reconstructions = []
    dump_array_egg = []
    for data, egg_data in test_loader:
        if use_cuda:
            data, egg_data = data.cuda(), egg_data.cuda()
        data, egg_data = Variable(data), Variable(egg_data)

        # Test model_G
        reconstructions, _ = model_G(data)
        loss_reconstruction = (egg_data * reconstructions).sum(dim=1) / (
            egg_data.norm(dim=1) * reconstructions.norm(dim=1)
        )
        loss_reconstruction = th.acos(loss_reconstruction) * 180 / np.pi
        loss_reconstruction = loss_reconstruction.mean()
        reconstruction_loss += loss_reconstruction.item()

        dump_array_reconstructions.append(
            reconstructions.cpu().detach().numpy().ravel()
        )
        dump_array_egg.append(egg_data.cpu().detach().numpy().ravel())

    dump_array_reconstructions = np.concatenate(dump_array_reconstructions)
    dump_array_egg = np.concatenate(dump_array_egg)

    # dump_array_egg = np.diff(dump_array_egg)

    # dump_array_reconstructions = smooth(dump_array_reconstructions, window_len = 20, window = "flat")
    # dump_array_reconstructions = smooth(dump_array_reconstructions, window_len = 20, window = "hanning")
    # dump_array_reconstructions = smooth(dump_array_reconstructions, window_len = 15, window = "savgol")
    # dump_array_reconstructions = np.diff(dump_array_reconstructions)

    plt.figure()
    plt.title("DotModel")
    plt.plot(dump_array_reconstructions, "b")
    plt.plot(dump_array_egg, "r")
    plt.show()

    print("Angle loss", reconstruction_loss / batches)


def main():
    test_data = create_dataloader(
        64,
        "CMU_new/bdl_test/speech",
        "CMU_new/bdl_test/egg_detrended",
        # "Childers/M_test/speech",
        # "Childers/M_test/egg_detrended",
        # "Temp1/speech",
        # "Temp1/egg_detrended",
        200,
        200,
        select=1,
    )

    # save_model = Saver('Models/DotModel/Childers_clean')
    save_model = Saver("checkpoints/clean300")
    use_cuda = True

    model_G = SpeechEggEncoder()
    model_D = Discriminator()
    model_G, model_D, _, _, _, _ = save_model.load_checkpoint(
        model_G, model_D, file_name="bce_epoch_45.pt"
    )

    test(model_G, model_D, test_data, use_cuda=use_cuda)


if __name__ == "__main__":
    main()
