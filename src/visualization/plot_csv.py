import csv

import matplotlib.pyplot as plt
import numpy as np

plt.switch_backend("qt5agg")

csvfile = "gci_results.csv"

results = []
with open(csvfile) as csvfile:
    readCSV = csv.reader(csvfile, delimiter=",")
    for row in readCSV:
        results.append(row)


csvfile = "MultispeakerResults.csv"
yolo_results = []
with open(csvfile) as csvfile:
    readCSV = csv.reader(csvfile, delimiter=",")
    for row in readCSV:
        yolo_results.append(row)


IDR_babble = np.zeros((6, 3))
FAR_babble = np.zeros((6, 3))
MR_babble = np.zeros((6, 3))
IDA_babble = np.zeros((6, 3))

IDR_white = np.zeros((6, 3))
FAR_white = np.zeros((6, 3))
MR_white = np.zeros((6, 3))
IDA_white = np.zeros((6, 3))

IDR_raw = np.zeros((1, 3))
FAR_raw = np.zeros((1, 3))
MR_raw = np.zeros((1, 3))
IDA_raw = np.zeros((1, 3))

IDR_YOLO_babble = np.zeros((6, 1))
FAR_YOLO_babble = np.zeros((6, 1))
MR_YOLO_babble = np.zeros((6, 1))
IDA_YOLO_babble = np.zeros((6, 1))


IDR_YOLO_white = np.zeros((6, 1))
FAR_YOLO_white = np.zeros((6, 1))
MR_YOLO_white = np.zeros((6, 1))
IDA_YOLO_white = np.zeros((6, 1))


for r in yolo_results:
    if "babble" in r[0] and "RAW" in r[0]:
        name = r[0]
        s = "".join(x for x in name if x.isdigit())
        index = int(int(s) / 5.0)
        IDR_YOLO_babble[index, :] = np.array([float(r[1]) / 100.0])
        MR_YOLO_babble[index, :] = np.array([float(r[2]) / 100.0])
        FAR_YOLO_babble[index, :] = np.array([float(r[3]) / 100.0])
        IDA_YOLO_babble[index, :] = np.array([float(r[4])])

    if "white" in r[0] and "RAW" in r[0]:
        name = r[0]
        s = "".join(x for x in name if x.isdigit())
        index = int(int(s) / 5.0)
        IDR_YOLO_white[index, :] = np.array([float(r[1]) / 100.0])
        MR_YOLO_white[index, :] = np.array([float(r[2]) / 100.0])
        FAR_YOLO_white[index, :] = np.array([float(r[3]) / 100.0])
        IDA_YOLO_white[index, :] = np.array([float(r[4])])

print(IDR_YOLO_babble)
print(IDR_YOLO_white)


for r in results:
    if "babble" in r[-1]:
        print(r[-1])
        name = r[-1]
        s = "".join(x for x in name if x.isdigit())
        index = int(int(s) / 5.0)
        print(index)
        IDR_babble[index, :] = np.array([r[0], r[4], r[8]])
        FAR_babble[index, :] = np.array([r[2], r[6], r[10]])
        MR_babble[index, :] = np.array([r[1], r[5], r[9]])
        IDA_babble[index, :] = np.array(
            [float(r[3]) / 16.0, float(r[7]) / 16.0, float(r[11]) / 16.0]
        )

    if "white" in r[-1]:
        print(r[-1])
        name = r[-1]
        s = "".join(x for x in name if x.isdigit())
        index = int(int(s) / 5.0)
        print(index)
        IDR_white[index, :] = np.array([r[0], r[4], r[8]])
        FAR_white[index, :] = np.array([r[2], r[6], r[10]])
        MR_white[index, :] = np.array([r[1], r[5], r[9]])
        IDA_white[index, :] = np.array(
            [float(r[3]) / 16.0, float(r[7]) / 16.0, float(r[11]) / 16.0]
        )

    if "raw" in r[-1]:
        print(r[-1])
        name = r[-1]
        index = 0
        IDR_raw[index, :] = np.array([r[0], r[4], r[8]])
        FAR_raw[index, :] = np.array([r[2], r[6], r[10]])
        MR_raw[index, :] = np.array([r[1], r[5], r[9]])
        IDA_raw[index, :] = np.array(
            [float(r[3]) / 16.0, float(r[7]) / 16.0, float(r[11]) / 16.0]
        )


print(IDR_raw)

fig = plt.figure(figsize=(14, 5))
plt.subplots_adjust(hspace=0.4, wspace=0.5)
plt.subplot(2, 4, 1)

names = ["SEDREAMS", "DPI", "MMF"]

for i in range(3):
    index = 5 * np.arange(6)
    plt.plot(index, IDR_babble[:, i], label=names[i])

plt.plot(index, IDR_YOLO_babble, label="YOLO")
plt.xlabel("SNR")
plt.ylabel("IDR")
plt.title("IDR for Babble Noise")
# plt.legend()
# plt.savefig('babbleIDR.png')


plt.subplot(2, 4, 2)

names = ["SEDREAMS", "DPI", "MMF"]

for i in range(3):
    index = 5 * np.arange(6)
    plt.plot(index, IDR_white[:, i])

plt.plot(index, IDR_YOLO_white)
plt.xlabel("SNR")
plt.ylabel("IDR")
plt.title("IDR for White Noise")
# plt.legend()
# plt.savefig('whiteIDR.png')

plt.subplot(2, 4, 3)
names = ["SEDREAMS", "DPI", "MMF"]

for i in range(3):
    index = 5 * np.arange(6)
    plt.plot(index, FAR_babble[:, i])

plt.plot(index, FAR_YOLO_babble)
plt.xlabel("SNR")
plt.ylabel("FAR")
plt.title("FAR for Babble Noise")
# plt.legend()
# plt.savefig('babbleFAR.png')


plt.subplot(2, 4, 4)

names = ["SEDREAMS", "DPI", "MMF"]

for i in range(3):
    index = 5 * np.arange(6)
    plt.plot(index, FAR_white[:, i])

plt.plot(index, FAR_YOLO_white)
plt.xlabel("SNR")
plt.ylabel("FAR")
plt.title("FAR for White Noise")
# plt.legend()
# plt.savefig('whiteFAR.png')

# plt.figure()
plt.subplot(2, 4, 5)
names = ["SEDREAMS", "DPI", "MMF"]

for i in range(3):
    index = 5 * np.arange(6)
    plt.plot(index, MR_babble[:, i])

plt.plot(index, MR_YOLO_babble)
plt.xlabel("SNR")
plt.ylabel("MR")
plt.title("MR for Babble Noise")
# plt.legend()
# plt.savefig('babbleMR.png')


# plt.figure()
plt.subplot(2, 4, 6)

names = ["SEDREAMS", "DPI", "MMF"]

for i in range(3):
    index = 5 * np.arange(6)
    plt.plot(index, MR_white[:, i])

plt.plot(index, MR_YOLO_white)
plt.xlabel("SNR")
plt.ylabel("MR")
plt.title("MR for White Noise")
# plt.legend()
# plt.savefig('whiteMR.png')


# plt.figure()
plt.subplot(2, 4, 7)
names = ["SEDREAMS", "DPI", "MMF"]

for i in range(3):
    index = 5 * np.arange(6)
    plt.plot(index, IDA_babble[:, i])

plt.plot(index, IDA_YOLO_babble)
plt.xlabel("SNR")
plt.ylabel("IDA")
plt.title("IDA for Babble Noise")
# plt.legend()
# plt.savefig('babbleIDA.png')


# plt.figure()
plt.subplot(2, 4, 8)

names = ["SEDREAMS", "DPI", "MMF"]

for i in range(3):
    index = 5 * np.arange(6)
    plt.plot(index, IDA_white[:, i])

plt.plot(index, IDA_YOLO_white)
plt.xlabel("SNR")
plt.ylabel("IDA")
plt.title("IDA for White Noise")
# plt.legend()
fig.legend(loc=0, ncol=4, mode="expand", borderaxespad=0.0)
plt.savefig("out.png")
# plt.tight_layout()
plt.show()
