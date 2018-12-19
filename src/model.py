import torch as th
import numpy as np
import torch.nn as nn
import math
import random


# INFO: Set random seeds
np.random.seed(42)
th.manual_seed(42)
th.cuda.manual_seed_all(42)
random.seed(42)


class SpeechEggEncoder(nn.Module):
    def __init__(self, window=100, dropout=0):
        super().__init__()

        self.c1 = nn.Sequential(
            nn.Linear(200, 175),
            nn.BatchNorm1d(175),
            nn.ReLU(),
            nn.Linear(175, 125),
            nn.BatchNorm1d(125),
            nn.ReLU(),
            nn.Linear(125, 100),
            nn.BatchNorm1d(100),
            nn.Tanh(),
        )

        self.feature_extractor = nn.Sequential(
            nn.Linear(100, 125),
            nn.BatchNorm1d(125),
            nn.ReLU(),
            nn.Linear(125, 175),
            nn.BatchNorm1d(175),
            nn.ReLU(),
            nn.Linear(175, 200),
        )

        self._initialize_submodules()

    def _initialize_submodules(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # init.kaiming_normal(m.weight.data)
                n = m.weight.size(1)
                m.weight.data.normal_(0, math.sqrt(1.0 / n))
            elif isinstance(m, nn.Conv1d):
                # n = m.kernel_size[0] * m.out_channels
                n = m.kernel_size[0] * m.in_channels
                m.weight.data.normal_(0, math.sqrt(1.0 / n))

    def forward(self, x):
        # Feature extractor
        embeddings_ = self.c1(x)
        y = self.feature_extractor(embeddings_)

        return y, embeddings_


class EGGEncoder(nn.Module):
    def __init__(self, window=100, dropout=0):
        super().__init__()

        self.c1 = nn.Sequential(
            nn.Linear(200, 175),
            nn.BatchNorm1d(175),
            nn.ReLU(),
            nn.Linear(175, 125),
            nn.BatchNorm1d(125),
            nn.ReLU(),
            nn.Linear(125, 100),
            nn.BatchNorm1d(100),
            nn.Tanh(),
        )

        self.feature_extractor = nn.Sequential(
            nn.Linear(100, 125),
            nn.BatchNorm1d(125),
            nn.ReLU(),
            nn.Linear(125, 175),
            nn.BatchNorm1d(175),
            nn.ReLU(),
            nn.Linear(175, 200),
            nn.Tanh(),
        )

        self._initialize_submodules()

    def _initialize_submodules(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # init.kaiming_normal(m.weight.data)
                n = m.weight.size(1)
                m.weight.data.normal_(0, math.sqrt(1.0 / n))
            elif isinstance(m, nn.Conv1d):
                # n = m.kernel_size[0] * m.out_channels
                n = m.kernel_size[0] * m.in_channels
                m.weight.data.normal_(0, math.sqrt(1.0 / n))

    def forward(self, x):
        y = self.c1(x)

        return y


class Discriminator(nn.Module):
    def __init__(self, window=100, dropout=0):
        super().__init__()

        self.feature_extractor = nn.Sequential(
            nn.Linear(100, 50),
            nn.BatchNorm1d(50),
            nn.ReLU(),
            nn.Linear(50, 25),
            nn.BatchNorm1d(25),
            nn.ReLU(),
            nn.Linear(25, 1),
            nn.BatchNorm1d(1),
            nn.Sigmoid(),
        )

        self._initialize_submodules()

    def _initialize_submodules(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # init.kaiming_normal(m.weight.data)
                n = m.weight.size(1)
                m.weight.data.normal_(0, math.sqrt(1.0 / n))
            elif isinstance(m, nn.Conv1d):
                # n = m.kernel_size[0] * m.out_channels
                n = m.kernel_size[0] * m.in_channels
                m.weight.data.normal_(0, math.sqrt(1.0 / n))

    def forward(self, x):
        y = self.feature_extractor(x)

        return y
