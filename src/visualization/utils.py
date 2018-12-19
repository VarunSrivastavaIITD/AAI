import os
from time import localtime, strftime

import numpy as np
import torch as th
import torch.nn as nn
import torch.optim as optim
from scipy.signal import butter, filtfilt, medfilt, savgol_filter


def detrend(speech, egg):
    def butter_highpass(cutoff, fs, order=5):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype="high", analog=False)
        return b, a

    def butter_highpass_filter(data, cutoff, fs, order=5):
        b, a = butter_highpass(cutoff, fs, order=order)
        y = filtfilt(b, a, data)
        return y

    order = 4
    fs = 16000
    cutoff = 50
    egg = butter_highpass_filter(egg, cutoff, fs, order)

    return speech, egg


def minmaxnormalize(signal):
    maxval = np.max(np.abs(signal))
    return signal / maxval


def positions2onehot(pos, shape):
    onehot = np.zeros(shape)
    onehot[pos] = 1
    return onehot


def smooth(s, window_len=10, window="hanning"):
    if window_len < 3:
        return s

    if window == "median":
        y = medfilt(s, kernel_size=window_len)
    elif window == "savgol":
        y = savgol_filter(s, window_len, 0)
    else:
        if window == "flat":  # moving average
            w = np.ones(window_len, "d")
        else:
            w = eval("np." + window + "(window_len)")

        y = np.convolve(w / w.sum(), s, mode="same")

    return y


def strided_app(a, L, S):  # Window len = L, Stride len/stepsize = S
    nrows = ((a.size - L) // S) + 1
    n = a.strides[0]
    return np.lib.stride_tricks.as_strided(a, shape=(nrows, L), strides=(S * n, n))


class Saver_Encoder:
    def __init__(self, directory: str = "pytorch_model", iteration: int = 0) -> None:
        self.directory = directory
        self.iteration = iteration

    def save_checkpoint(
        self, state, file_name: str = "pytorch_model.pt", append_time=True
    ):
        os.makedirs(self.directory, exist_ok=True)
        timestamp = strftime("%Y_%m_%d__%H_%M_%S", localtime())
        filebasename, fileext = file_name.split(".")
        if append_time:
            filepath = os.path.join(
                self.directory, "_".join([filebasename, ".".join([timestamp, fileext])])
            )
        else:
            filepath = os.path.join(self.directory, file_name)
        if isinstance(state, nn.Module):
            checkpoint = {"model_dict": state.state_dict()}
            th.save(checkpoint, filepath)
        elif isinstance(state, dict):
            th.save(state, filepath)
        else:
            raise TypeError("state must be a nn.Module or dict")

    def load_checkpoint(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer = None,
        file_name: str = "pytorch_model.pt",
    ):
        filepath = os.path.join(self.directory, file_name)
        checkpoint = th.load(filepath)
        model.load_state_dict(checkpoint["model_dict"])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint["optimizer_dict"])

        hyperparam_dict = {
            k: v
            for k, v in checkpoint.items()
            if k != "model_dict" or k != "optimizer_dict"
        }

        return model, optimizer, hyperparam_dict

    def create_checkpoint(
        self, model: nn.Module, optimizer: optim.Optimizer, hyperparam_dict
    ):
        model_dict = model.state_dict()
        optimizer_dict = optimizer.state_dict()

        state_dict = {
            "model_dict": model_dict,
            "optimizer_dict": optimizer_dict,
            "timestamp": strftime("%I:%M%p GMT%z on %b %d, %Y", localtime()),
        }
        checkpoint = {**state_dict, **hyperparam_dict}

        return checkpoint


class Saver:
    def __init__(self, directory: str = "pytorch_model", iteration: int = 0) -> None:
        self.directory = directory
        self.iteration = iteration

    def save_checkpoint(
        self, state, file_name: str = "pytorch_model.pt", append_time=True
    ):
        os.makedirs(self.directory, exist_ok=True)
        timestamp = strftime("%Y_%m_%d__%H_%M_%S", localtime())
        filebasename, fileext = file_name.split(".")
        if append_time:
            filepath = os.path.join(
                self.directory, "_".join([filebasename, ".".join([timestamp, fileext])])
            )
        else:
            filepath = os.path.join(self.directory, file_name)
        # if isinstance(state, nn.Module):
        #     checkpoint = {'model_dict': state.state_dict()}
        #     th.save(checkpoint, filepath)
        if isinstance(state, dict):
            th.save(state, filepath)
        else:
            raise TypeError("state must be dict")

    def load_checkpoint(
        self,
        model_G: nn.Module,
        model_D: nn.Module,
        optimizer_G: optim.Optimizer = None,
        optimizer_R: optim.Optimizer = None,
        optimizer_D: optim.Optimizer = None,
        file_name: str = "pytorch_model.pt",
    ):
        filepath = os.path.join(self.directory, file_name)
        checkpoint = th.load(filepath)
        model_G.load_state_dict(checkpoint["model_G_dict"])
        model_D.load_state_dict(checkpoint["model_D_dict"])
        if optimizer_G is not None:
            optimizer_G.load_state_dict(checkpoint["optimizer_G_dict"])
            optimizer_R.load_state_dict(checkpoint["optimizer_R_dict"])
            optimizer_D.load_state_dict(checkpoint["optimizer_D_dict"])

        hyperparam_dict = {
            k: v
            for k, v in checkpoint.items()
            if k != "model_G_dict"
            or k != "model_D_dict"
            or k != "optimizer_G_dict"
            or k != "optimizer_D_dict"
        }

        return model_G, model_D, optimizer_G, optimizer_R, optimizer_D, hyperparam_dict

    def create_checkpoint(
        self,
        model_G: nn.Module,
        model_D: nn.Module,
        optimizer_G: optim.Optimizer,
        optimizer_R: optim.Optimizer,
        optimizer_D: optim.Optimizer,
        hyperparam_dict,
    ):
        model_G_dict = model_G.state_dict()
        model_D_dict = model_D.state_dict()
        optimizer_G_dict = optimizer_G.state_dict()
        optimizer_R_dict = optimizer_R.state_dict()
        optimizer_D_dict = optimizer_D.state_dict()

        state_dict = {
            "model_G_dict": model_G_dict,
            "model_D_dict": model_D_dict,
            "optimizer_G_dict": optimizer_G_dict,
            "optimizer_R_dict": optimizer_R_dict,
            "optimizer_D_dict": optimizer_D_dict,
            "timestamp": strftime("%I:%M%p GMT%z on %b %d, %Y", localtime()),
        }
        checkpoint = {**state_dict, **hyperparam_dict}

        return checkpoint

    def load_checkpoint_reconstructions(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer = None,
        file_name: str = "pytorch_model.pt",
    ):
        filepath = os.path.join(self.directory, file_name)
        checkpoint = th.load(filepath)
        model.load_state_dict(checkpoint["model_dict"])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint["optimizer_dict"])

        hyperparam_dict = {
            k: v
            for k, v in checkpoint.items()
            if k != "model_dict" or k != "optimizer_dict"
        }

        return model, optimizer, hyperparam_dict
