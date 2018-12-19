import os
from glob import glob

import numpy as np
import torch as th
from torch.utils.data import DataLoader, TensorDataset

from utils import strided_app, detrend
import random
import warnings
from sklearn.model_selection import train_test_split
from torch_utils import AudioDataset

warnings.filterwarnings("ignore")

__RANDOM_SEED = 42
np.random.seed(__RANDOM_SEED)
th.manual_seed(__RANDOM_SEED)
th.cuda.manual_seed_all(__RANDOM_SEED)
random.seed(__RANDOM_SEED)


def custom_loader(speechfolder, eggfolder, window, stride, select=None):
    speechfiles = sorted(glob(os.path.join(speechfolder, "*.npy")))
    eggfiles = sorted(glob(os.path.join(eggfolder, "*.npy")))

    if select is not None:
        ind = np.random.permutation(len(speechfiles))
        ind = ind[:select]
        speechfiles = [speechfiles[i] for i in ind]
        eggfiles = [eggfiles[i] for i in ind]
        print("Selected {} files".format(select))

    speech_data = [np.load(f) for f in speechfiles]
    egg_data = [np.load(f) for f in eggfiles]

    for i in range(len(egg_data)):
        egg_data[i] = egg_data[i] / np.max(np.abs(egg_data[i]))

    for i in range(len(speech_data)):
        speech_data[i] = speech_data[i] / np.max(np.abs(speech_data[i]))

    speech_data = np.concatenate(speech_data)
    egg_data = np.concatenate(egg_data)

    speech_windowed_data = strided_app(speech_data, window, stride)
    egg_windowed_data = strided_app(egg_data, window, stride)

    return speech_windowed_data, egg_windowed_data


def create_dataloader(batch_size, speechfolder, eggfolder, window, stride, select=None):
    print(select, "files to be selected")
    speech_windowed_data, egg_windowed_data = custom_loader(
        speechfolder, eggfolder, window, stride, select=select
    )
    dataset = TensorDataset(
        th.from_numpy(speech_windowed_data.astype(np.float32)),
        th.from_numpy(egg_windowed_data.astype(np.float32)),
    )
    dataloader = DataLoader(
        dataset, batch_size=batch_size, pin_memory=False, drop_last=False, shuffle=True
    )
    return dataloader


def train_validate_test_loader(
    speechfolder,
    eggfolder,
    window=200,
    stride=1,
    batch_size=512,
    split=0.7,
    numfiles=None,
    pin_memory=False,
    model_folder=None,
    workers=6,
):
    def _make_train_validate_test_splitter(split_kw):
        def fn(X, y):
            train_frac = split_kw["train"]
            test_frac = split_kw["test"]
            validation_frac = split_kw["validate"]

            Xtrainval, Xtest, ytrainval, ytest = train_test_split(
                X, y, test_size=test_frac, random_state=__RANDOM_SEED
            )
            Xtrain, Xval, ytrain, yval = train_test_split(
                Xtrainval,
                ytrainval,
                test_size=validation_frac / (validation_frac + train_frac),
                random_state=__RANDOM_SEED,
            )

            return (Xtrain, Xval, Xtest), (ytrain, yval, ytest)

        return fn

    def _collate_fn(data):
        speech, egg = zip(*data)
        return [th.cat(speech, 0), th.cat(egg, 0)]

    if isinstance(batch_size, int):
        batch_size = {k: batch_size for k in ["train", "validate", "test"]}
    elif isinstance(batch_size, dict):
        batch_size.setdefault("train", 512)
        batch_size.setdefault("validate", 512)
        batch_size.setdefault("test", 512)
    else:
        raise ValueError("Incorrect Batch Size Argument, Must be an integer or dict")

    if isinstance(split, float):
        split = {"train": split, "validate": 1 - split, "test": 0}
    elif not isinstance(split, dict):
        raise ValueError("Incorrect split Argument, Must be a float or dict")

    if isinstance(stride, int):
        stride = {k: stride for k in ["train", "validate", "test"]}
    elif isinstance(stride, dict):
        stride.setdefault("train", 1)
        stride.setdefault("validate", 1)
        stride.setdefault("test", 1)
    else:
        raise ValueError("stride must be a dict or integer")

    speechfiles = sorted(glob(os.path.join(speechfolder, "*.npy")))
    speechfiles = [os.path.basename(f) for f in speechfiles]
    eggfiles = sorted(glob(os.path.join(eggfolder, "*.npy")))
    eggfiles = [os.path.basename(f) for f in eggfiles]

    fileset = set(speechfiles).intersection(set(eggfiles))
    fileset = list(fileset)
    fileset.sort()
    random.Random(__RANDOM_SEED).shuffle(fileset)
    speechfiles = eggfiles = fileset

    if numfiles is None:
        numfiles = len(speechfiles)

    numfiles = int(sum(split.values()) * numfiles)
    speechfiles = speechfiles[:numfiles]
    eggfiles = speechfiles[:numfiles]

    nsplit = {k: v / sum(split.values()) for k, v in split.items()}

    create_train_validate_test_split = _make_train_validate_test_splitter(nsplit)

    speechtuple, eggtuple = create_train_validate_test_split(speechfiles, eggfiles)
    speechtuple = [
        [os.path.join(speechfolder, sp) for sp in splitfilelist]
        for splitfilelist in speechtuple
    ]
    eggtuple = [
        [os.path.join(eggfolder, sp) for sp in splitfilelist]
        for splitfilelist in eggtuple
    ]

    train, validation, test = zip(speechtuple, eggtuple)

    traindataset = AudioDataset(*train, window, stride["train"], transform=detrend)
    validationdataset = AudioDataset(
        *validation, window, stride["validate"], transform=detrend
    )
    testdataset = AudioDataset(*test, window, stride["test"], transform=detrend)

    trainloader = DataLoader(
        traindataset,
        batch_size=batch_size["train"],
        pin_memory=pin_memory,
        drop_last=False,
        shuffle=True,
        collate_fn=_collate_fn,
        num_workers=workers,
    )
    validationloader = DataLoader(
        validationdataset,
        batch_size=batch_size["validate"],
        pin_memory=pin_memory,
        drop_last=False,
        shuffle=True,
        collate_fn=_collate_fn,
        num_workers=workers,
    )
    testloader = DataLoader(
        testdataset,
        batch_size=batch_size["test"],
        pin_memory=pin_memory,
        drop_last=False,
        shuffle=True,
        collate_fn=_collate_fn,
        num_workers=workers,
    )

    if model_folder is not None:
        try:
            os.makedirs(model_folder, exist_ok=True)
            train_speech_dir = os.path.join(model_folder, "train/speech")
            train_egg_dir = os.path.join(model_folder, "train/egg")
            validate_speech_dir = os.path.join(model_folder, "validate/speech")
            validate_egg_dir = os.path.join(model_folder, "validate/egg")
            test_speech_dir = os.path.join(model_folder, "test/speech")
            test_egg_dir = os.path.join(model_folder, "test/egg")
            os.makedirs(train_speech_dir)
            os.makedirs(train_egg_dir)
            os.makedirs(validate_speech_dir)
            os.makedirs(validate_egg_dir)
            os.makedirs(test_speech_dir)
            os.makedirs(test_egg_dir)

            for files, dest_dir in zip(
                [*train, *validation, *test],
                [
                    train_speech_dir,
                    train_egg_dir,
                    validate_speech_dir,
                    validate_egg_dir,
                    test_speech_dir,
                    test_egg_dir,
                ],
            ):
                for file in files:
                    src = os.path.join(os.getcwd(), file)
                    basename = os.path.basename(src)
                    dst = os.path.join(os.getcwd(), dest_dir, basename)
                    os.symlink(src, dst)
        except OSError as ose:
            print("Symlink Directories already exist. Please remove old data first")
            print(ose)

    return trainloader, validationloader, testloader


def main():
    train, validation, test = train_validate_test_loader(
        "../data/bdl/speech",
        "../data/bdl/egg",
        split={"train": 0.01, "validate": 0.01, "test": 0.5},
        batch_size=20,
        workers=4,
        # numfiles=100,
        model_folder="data/bdl_clean",
    )


if __name__ == "__main__":
    main()
