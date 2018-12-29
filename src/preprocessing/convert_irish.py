import matlab.engine
import matlab
from glob import glob
import scipy.io.wavfile as wf
import os
import numpy as np
from tqdm import tqdm


def main():
    basedir = 'Irish'
    resultdir = os.path.join(basedir, 'resampled_merged')
    files = glob(os.path.join(basedir, "**/*.wav"), recursive=True)
    try:
        eng = matlab.engine.start_matlab()

        for f in tqdm(files):
            fs, data = wf.read(f)
            assert fs == 44100
            path, fname = os.path.split(f)

            split_name = fname.split(".")
            # ext = split_name[-1]
            basename = '_'.join(split_name[:-1])
            eggdata = data[:, 1]
            speechdata = data[:, 0]
            # _, eggdata = wf.read(os.path.join(path, basename + ".egg"))

            speechdata = speechdata.astype(np.float32)
            eggdata = eggdata.astype(np.float32)

            mspeechdata = matlab.double(speechdata.tolist())
            meggdata = matlab.double(eggdata.tolist())

            resspeechdata = eng.resample(mspeechdata, 16000.0, 44100.0)
            reseggdata = eng.resample(meggdata, 16000.0, 44100.0)

            resspeechdata = np.array(resspeechdata).ravel()
            reseggdata = np.array(reseggdata).ravel()

            # finaldata = np.stack((resdata, reseggdata), axis=-1)

            assert resspeechdata.shape == reseggdata.shape

            # wf.write(
            #     os.path.join('SampledAplawd', path[1:], 'R' + fname),
            #     16000, resdata)

            # speechdir = os.path.join(resultdir, *path.split("/")[1:], "speech")
            # eggdir = os.path.join(resultdir, *path.split("/")[1:], "egg")
            speechdir = os.path.join(resultdir, "speech")
            eggdir = os.path.join(resultdir, "egg")

            os.makedirs(speechdir, exist_ok=True)
            os.makedirs(eggdir, exist_ok=True)

            # print(os.path.join(speechdir, basename))
            # print(os.path.join(eggdir, basename))

            np.save(os.path.join(speechdir, basename), resspeechdata)
            np.save(os.path.join(eggdir, basename), reseggdata)
    finally:
        eng.quit()


if __name__ == "__main__":
    main()
