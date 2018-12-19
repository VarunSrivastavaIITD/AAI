import torch as th
from torch.utils.data import Dataset
import numpy as np
import os


class AudioDataset(Dataset):
    def __init__(self, speechfiles, eggfiles, window, stride, transform=None):
        self.speechfiles = sorted(speechfiles)
        self.eggfiles = sorted(eggfiles)
        self.transform = transform
        self.window = window
        self.stride = stride

    def __len__(self):
        return len(self.speechfiles)

    def __getitem__(self, idx):
        speech = np.load(self.speechfiles[idx])
        egg = np.load(self.eggfiles[idx])

        speech, egg = speech / np.max(np.abs(speech)), egg / np.max(np.abs(egg))

        if self.transform is not None:
            speech, egg = self.transform(speech, egg)
        speech = speech.astype(np.float32)
        egg = egg.astype(np.float32)
        speech, egg = (
            th.from_numpy(speech).unfold(0, self.window, self.stride),
            th.from_numpy(egg).unfold(0, self.window, self.stride),
        )

        return speech, egg


class AudioFileDataset(Dataset):
    def __init__(self, speechfiles, eggfiles, window, stride, transform=None):
        self.speechfiles = sorted(speechfiles)
        self.eggfiles = sorted(eggfiles)
        self.transform = transform
        self.window = window
        self.stride = stride

    def __len__(self):
        return len(self.speechfiles)

    def __getitem__(self, idx):
        speech = np.load(self.speechfiles[idx])
        egg = np.load(self.eggfiles[idx])

        numsamples = len(speech)

        speech, egg = speech / np.max(np.abs(speech)), egg / np.max(np.abs(egg))

        if self.transform is not None:
            speech, egg = self.transform(speech, egg)
        speech = speech.astype(np.float32)
        egg = egg.astype(np.float32)
        speech, egg = (
            th.from_numpy(speech).unfold(0, self.window, self.stride),
            th.from_numpy(egg).unfold(0, self.window, self.stride),
        )

        return speech, egg, os.path.basename(self.speechfiles[idx]), numsamples
