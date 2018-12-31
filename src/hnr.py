import numpy as np
from utils import onehot2positions, positions2onehot
from typing import List

SVector = List[slice]


def harmonic_to_noise_ratio(y, mean_pitch):
    y = y * np.hanning(len(y))
    a = np.correlate(y, y, mode="full")
    a = a / np.max(a)
    a1 = a[int(np.floor(len(a) / 2)) : len(a)]
    T = mean_pitch
    r_max = np.max(a1[int(np.rint(T / 2)) : int(np.rint(1.5 * T))])
    hnr = 10 * np.log10(r_max / (1 - r_max))

    return hnr


def extract_hnr(
    egg: np.ndarray,
    gci_onehot: np.ndarray,
    segmented_regions: SVector,
    min_cycles: int = 4,
):
    for reg in segmented_regions:
        voiced_egg = egg[reg]
        gci_positions = onehot2positions(gci_onehot[reg])

        if len(gci_positions) < min_cycles:
            continue

        hnrs = []
        for idx in range(0, len(gci_positions) - min_cycles, min_cycles):
            start = gci_positions[idx]
            end = gci_positions[idx + min_cycles]
            mean_pitch = np.mean(np.diff(gci_positions[idx : idx + min_cycles + 1]))
            y = voiced_egg[start:end]
            hnr = harmonic_to_noise_ratio(y, mean_pitch)
            hnrs.append(hnr)

        return np.mean(hnrs)

    raise ValueError('Very short signal')
