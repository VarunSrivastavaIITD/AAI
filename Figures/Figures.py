import argparse
import os

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
plt.switch_backend("QT5Agg")


def GetGCIMetrix(file):
    file = open(file)
    metrix = file.readlines()
    file.close()

    metrix = metrix[:-2]

    idr = []
    mr = []
    far = []
    ida = []
    for i in metrix:
        i = i.rstrip().split()
        idr.append(float(i[2][:-1]))
        mr.append(float(i[4][:-1]))
        far.append(float(i[6][:-1]))
        ida.append(float(i[8]))

    idr = np.array(idr)
    mr = np.array(mr)
    far = np.array(far)
    ida = np.array(ida)

    return idr, mr, far, ida


def GetQuotients(file):
    file = open(file)
    metrix = file.readlines()
    file.close()

    metrix = metrix[:-2]

    cq_true = []
    cq_estimated = []
    oq_true = []
    oq_estimated = []
    sq_true = []
    sq_estimated = []
    for i in metrix:
        i = i.rstrip().split()
        cq_true.append(float(i[2]))
        cq_estimated.append(float(i[4]))
        oq_true.append(float(i[6]))
        oq_estimated.append(float(i[8]))
        sq_true.append(float(i[10]))
        sq_estimated.append(float(i[12]))

    cq_true = np.array(cq_true)
    cq_estimated = np.array(cq_estimated)
    oq_true = np.array(oq_true)
    oq_estimated = np.array(oq_estimated)
    sq_true = np.array(sq_true)
    sq_estimated = np.array(sq_estimated)

    return cq_true, cq_estimated, oq_true, oq_estimated, sq_true, sq_estimated


def main():
    ############################################### GCI ###############################################
    idr0 = []
    idr0.append(GetGCIMetrix(
        "Results_DotModel/Childers_clean_bdl_clean/GCI.txt")[0])
    idr0.append(GetGCIMetrix(
        "Results_DotModel/Childers_clean_jmk_clean/GCI.txt")[0])
    idr0.append(GetGCIMetrix(
        "Results_DotModel/Childers_clean_slt_clean/GCI.txt")[0])
    idr0 = np.concatenate(idr0)

    idr1 = []
    idr1.append(GetGCIMetrix(
        "Results_DotModel/Childers_0_babble_bdl_0_white/GCI.txt")[0])
    idr1.append(GetGCIMetrix(
        "Results_DotModel/Childers_0_babble_jmk_0_white/GCI.txt")[0])
    idr1.append(GetGCIMetrix(
        "Results_DotModel/Childers_0_babble_slt_0_white/GCI.txt")[0])
    idr1 = np.concatenate(idr1)

    idr2 = []
    idr2.append(GetGCIMetrix(
        "Results_DotModel/Childers_0_babble_bdl_0_babble/GCI.txt")[0])
    idr2.append(GetGCIMetrix(
        "Results_DotModel/Childers_0_babble_jmk_0_babble/GCI.txt")[0])
    idr2.append(GetGCIMetrix(
        "Results_DotModel/Childers_0_babble_slt_0_babble/GCI.txt")[0])
    idr2 = np.concatenate(idr2)

    idr3 = []
    idr3.append(GetGCIMetrix(
        "Results_DotModel/Irish_clean_Irish_clean/GCI.txt")[0])
    idr3 = np.array(idr3)

    idr4 = []
    idr4.append(GetGCIMetrix(
        "Results_DotModel/German_clean_German_clean/GCI/Bulbarparalyse.txt")[0])
    idr4.append(GetGCIMetrix(
        "Results_DotModel/German_clean_German_clean/GCI/Dysplastischer Kehlkopf.txt")[0])
    idr4.append(GetGCIMetrix(
        "Results_DotModel/German_clean_German_clean/GCI/Hyperasthenie.txt")[0])
    idr4.append(GetGCIMetrix(
        "Results_DotModel/German_clean_German_clean/GCI/Hyperfunktionelle Dysphonie.txt")[0])
    idr4.append(GetGCIMetrix(
        "Results_DotModel/German_clean_German_clean/GCI/Non-fluency-Syndrom.txt")[0])
    idr4.append(GetGCIMetrix(
        "Results_DotModel/German_clean_German_clean/GCI/Sigmatismus.txt")[0])
    idr4.append(GetGCIMetrix(
        "Results_DotModel/German_clean_German_clean/GCI/Stimmlippenkarzinom.txt")[0])
    idr4.append(GetGCIMetrix(
        "Results_DotModel/German_clean_German_clean/GCI/Taschenfaltenhyperplasie.txt")[0])
    idr4 = np.concatenate(idr4)

    ax = plt.subplot(411)
    ax.set_title("GCI (identification rate)")
    ax.boxplot([idr0, idr1, idr2, idr3, idr4], sym=" ")
    plt.xticks([1, 2, 3, 4, 5], ["CMU clean",
                                 "CMU 0dB white", "CMU 0dB babble", "Irish", "German"])

    ############################################### GOI ###############################################
    idr0 = []
    idr0.append(GetGCIMetrix(
        "Results_DotModel/Childers_clean_bdl_clean/GOI.txt")[0])
    idr0.append(GetGCIMetrix(
        "Results_DotModel/Childers_clean_jmk_clean/GOI.txt")[0])
    idr0.append(GetGCIMetrix(
        "Results_DotModel/Childers_clean_slt_clean/GOI.txt")[0])
    idr0 = np.concatenate(idr0)

    idr1 = []
    idr1.append(GetGCIMetrix(
        "Results_DotModel/Childers_0_babble_bdl_0_white/GOI.txt")[0])
    idr1.append(GetGCIMetrix(
        "Results_DotModel/Childers_0_babble_jmk_0_white/GOI.txt")[0])
    idr1.append(GetGCIMetrix(
        "Results_DotModel/Childers_0_babble_slt_0_white/GOI.txt")[0])
    idr1 = np.concatenate(idr1)

    idr2 = []
    idr2.append(GetGCIMetrix(
        "Results_DotModel/Childers_0_babble_bdl_0_babble/GOI.txt")[0])
    idr2.append(GetGCIMetrix(
        "Results_DotModel/Childers_0_babble_jmk_0_babble/GOI.txt")[0])
    idr2.append(GetGCIMetrix(
        "Results_DotModel/Childers_0_babble_slt_0_babble/GOI.txt")[0])
    idr2 = np.concatenate(idr2)

    idr3 = []
    idr3.append(GetGCIMetrix(
        "Results_DotModel/Irish_clean_Irish_clean/GOI.txt")[0])
    idr3 = np.array(idr3)

    idr4 = []
    idr4.append(GetGCIMetrix(
        "Results_DotModel/German_clean_German_clean/GOI/Bulbarparalyse.txt")[0])
    idr4.append(GetGCIMetrix(
        "Results_DotModel/German_clean_German_clean/GOI/Dysplastischer Kehlkopf.txt")[0])
    idr4.append(GetGCIMetrix(
        "Results_DotModel/German_clean_German_clean/GOI/Hyperasthenie.txt")[0])
    idr4.append(GetGCIMetrix(
        "Results_DotModel/German_clean_German_clean/GOI/Hyperfunktionelle Dysphonie.txt")[0])
    idr4.append(GetGCIMetrix(
        "Results_DotModel/German_clean_German_clean/GOI/Non-fluency-Syndrom.txt")[0])
    idr4.append(GetGCIMetrix(
        "Results_DotModel/German_clean_German_clean/GOI/Sigmatismus.txt")[0])
    idr4.append(GetGCIMetrix(
        "Results_DotModel/German_clean_German_clean/GOI/Stimmlippenkarzinom.txt")[0])
    idr4.append(GetGCIMetrix(
        "Results_DotModel/German_clean_German_clean/GOI/Taschenfaltenhyperplasie.txt")[0])
    idr4 = np.concatenate(idr4)

    ax = plt.subplot(412)
    ax.set_title("GOI (identification rate)")
    ax.boxplot([idr0, idr1, idr2, idr3, idr4], sym=" ")
    plt.xticks([1, 2, 3, 4, 5], ["CMU clean",
                                 "CMU 0dB white", "CMU 0dB babble", "Irish", "German"])

    ############################################### Quotients ###############################################
    cq_true0 = []
    cq_estimated0 = []
    oq_true0 = []
    oq_estimated0 = []
    sq_true0 = []
    sq_estimated0 = []
    cq_true_, cq_estimated_, oq_true_, oq_estimated_, sq_true_, sq_estimated_ = GetQuotients(
        "Results_DotModel/Childers_clean_bdl_clean/Quotients.txt")
    cq_true0.append(cq_true_)
    cq_estimated0.append(cq_estimated_)
    oq_true0.append(oq_true_)
    oq_estimated0.append(oq_estimated_)
    sq_true0.append(sq_true_)
    sq_estimated0.append(sq_estimated_)
    # cq_true_, cq_estimated_, oq_true_, oq_estimated_, sq_true_, sq_estimated_ = GetQuotients(
    #     "Results_DotModel/Childers_clean_jmk_clean/Quotients.txt")
    # cq_true0.append(cq_true_)
    # cq_estimated0.append(cq_estimated_)
    # oq_true0.append(oq_true_)
    # oq_estimated0.append(oq_estimated_)
    # sq_true0.append(sq_true_)
    # sq_estimated0.append(sq_estimated_)
    # cq_true_, cq_estimated_, oq_true_, oq_estimated_, sq_true_, sq_estimated_ = GetQuotients(
    #     "Results_DotModel/Childers_clean_slt_clean/Quotients.txt")
    # cq_true0.append(cq_true_)
    # cq_estimated0.append(cq_estimated_)
    # oq_true0.append(oq_true_)
    # oq_estimated0.append(oq_estimated_)
    # sq_true0.append(sq_true_)
    # sq_estimated0.append(sq_estimated_)

    # cq_true1 = []
    # cq_estimated1 = []
    # oq_true1 = []
    # oq_estimated1 = []
    # sq_true1 = []
    # sq_estimated1 = []
    # cq_true_, cq_estimated_, oq_true_, oq_estimated_, sq_true_, sq_estimated_ = GetQuotients(
    #     "Results_DotModel/Childers_0_babble_bdl_0_white/Quotients.txt")
    # cq_true1.append(cq_true_)
    # cq_estimated1.append(cq_estimated_)
    # oq_true1.append(oq_true_)
    # oq_estimated1.append(oq_estimated_)
    # sq_true1.append(sq_true_)
    # sq_estimated1.append(sq_estimated_)
    # cq_true_, cq_estimated_, oq_true_, oq_estimated_, sq_true_, sq_estimated_ = GetQuotients(
    #     "Results_DotModel/Childers_0_babble_jmk_0_white/Quotients.txt")
    # cq_true1.append(cq_true_)
    # cq_estimated1.append(cq_estimated_)
    # oq_true1.append(oq_true_)
    # oq_estimated1.append(oq_estimated_)
    # sq_true1.append(sq_true_)
    # sq_estimated1.append(sq_estimated_)
    # cq_true_, cq_estimated_, oq_true_, oq_estimated_, sq_true_, sq_estimated_ = GetQuotients(
    #     "Results_DotModel/Childers_0_babble_slt_0_white/Quotients.txt")
    # cq_true1.append(cq_true_)
    # cq_estimated1.append(cq_estimated_)
    # oq_true1.append(oq_true_)
    # oq_estimated1.append(oq_estimated_)
    # sq_true1.append(sq_true_)
    # sq_estimated1.append(sq_estimated_)

    # cq_true2 = []
    # cq_estimated2 = []
    # oq_true2 = []
    # oq_estimated2 = []
    # sq_true2 = []
    # sq_estimated2 = []
    # cq_true_, cq_estimated_, oq_true_, oq_estimated_, sq_true_, sq_estimated_ = GetQuotients(
    #     "Results_DotModel/Childers_0_babble_bdl_0_white/Quotients.txt")
    # cq_true2.append(cq_true_)
    # cq_estimated2.append(cq_estimated_)
    # oq_true2.append(oq_true_)
    # oq_estimated2.append(oq_estimated_)
    # sq_true2.append(sq_true_)
    # sq_estimated2.append(sq_estimated_)
    # cq_true_, cq_estimated_, oq_true_, oq_estimated_, sq_true_, sq_estimated_ = GetQuotients(
    #     "Results_DotModel/Childers_0_babble_jmk_0_white/Quotients.txt")
    # cq_true2.append(cq_true_)
    # cq_estimated2.append(cq_estimated_)
    # oq_true2.append(oq_true_)
    # oq_estimated2.append(oq_estimated_)
    # sq_true2.append(sq_true_)
    # sq_estimated2.append(sq_estimated_)
    # cq_true_, cq_estimated_, oq_true_, oq_estimated_, sq_true_, sq_estimated_ = GetQuotients(
    #     "Results_DotModel/Childers_0_babble_slt_0_white/Quotients.txt")
    # cq_true2.append(cq_true_)
    # cq_estimated2.append(cq_estimated_)
    # oq_true2.append(oq_true_)
    # oq_estimated2.append(oq_estimated_)
    # sq_true2.append(sq_true_)
    # sq_estimated2.append(sq_estimated_)

    # cq_true3 = []
    # cq_estimated3 = []
    # oq_true3 = []
    # oq_estimated3 = []
    # sq_true3 = []
    # sq_estimated3 = []
    # cq_true_, cq_estimated_, oq_true_, oq_estimated_, sq_true_, sq_estimated_ = GetQuotients(
    #     "Results_DotModel/Irish_clean_Irish_clean/Quotients.txt")
    # cq_true3.append(cq_true_)
    # cq_estimated3.append(cq_estimated_)
    # oq_true3.append(oq_true_)
    # oq_estimated3.append(oq_estimated_)
    # sq_true3.append(sq_true_)
    # sq_estimated3.append(sq_estimated_)

    # cq_true4 = []
    # cq_estimated4 = []
    # oq_true4 = []
    # oq_estimated4 = []
    # sq_true4 = []
    # sq_estimated4 = []
    # cq_true_, cq_estimated_, oq_true_, oq_estimated_, sq_true_, sq_estimated_ = GetQuotients(
    #     "Results_DotModel/German_clean_German_clean/GOI/Bulbarparalyse.txt")
    # cq_true4.append(cq_true_)
    # cq_estimated4.append(cq_estimated_)
    # oq_true4.append(oq_true_)
    # oq_estimated4.append(oq_estimated_)
    # sq_true4.append(sq_true_)
    # sq_estimated4.append(sq_estimated_)
    # cq_true_, cq_estimated_, oq_true_, oq_estimated_, sq_true_, sq_estimated_ = GetQuotients(
    #     "Results_DotModel/German_clean_German_clean/GOI/Dysplastischer Kehlkopf.txt")
    # cq_true4.append(cq_true_)
    # cq_estimated4.append(cq_estimated_)
    # oq_true4.append(oq_true_)
    # oq_estimated4.append(oq_estimated_)
    # sq_true4.append(sq_true_)
    # sq_estimated4.append(sq_estimated_)
    # cq_true_, cq_estimated_, oq_true_, oq_estimated_, sq_true_, sq_estimated_ = GetQuotients(
    #     "Results_DotModel/German_clean_German_clean/GOI/Hyperasthenie.txt")
    # cq_true4.append(cq_true_)
    # cq_estimated4.append(cq_estimated_)
    # oq_true4.append(oq_true_)
    # oq_estimated4.append(oq_estimated_)
    # sq_true4.append(sq_true_)
    # sq_estimated4.append(sq_estimated_)
    # cq_true_, cq_estimated_, oq_true_, oq_estimated_, sq_true_, sq_estimated_ = GetQuotients(
    #     "Results_DotModel/German_clean_German_clean/GOI/Hyperfunktionelle Dysphonie.txt")
    # cq_true4.append(cq_true_)
    # cq_estimated4.append(cq_estimated_)
    # oq_true4.append(oq_true_)
    # oq_estimated4.append(oq_estimated_)
    # sq_true4.append(sq_true_)
    # sq_estimated4.append(sq_estimated_)
    # cq_true_, cq_estimated_, oq_true_, oq_estimated_, sq_true_, sq_estimated_ = GetQuotients(
    #     "Results_DotModel/German_clean_German_clean/GOI/Non-fluency-Syndrom.txt")
    # cq_true4.append(cq_true_)
    # cq_estimated4.append(cq_estimated_)
    # oq_true4.append(oq_true_)
    # oq_estimated4.append(oq_estimated_)
    # sq_true4.append(sq_true_)
    # sq_estimated4.append(sq_estimated_)
    # cq_true_, cq_estimated_, oq_true_, oq_estimated_, sq_true_, sq_estimated_ = GetQuotients(
    #     "Results_DotModel/German_clean_German_clean/GOI/Sigmatismus.txt")
    # cq_true4.append(cq_true_)
    # cq_estimated4.append(cq_estimated_)
    # oq_true4.append(oq_true_)
    # oq_estimated4.append(oq_estimated_)
    # sq_true4.append(sq_true_)
    # sq_estimated4.append(sq_estimated_)
    # cq_true_, cq_estimated_, oq_true_, oq_estimated_, sq_true_, sq_estimated_ = GetQuotients(
    #     "Results_DotModel/German_clean_German_clean/GOI/Stimmlippenkarzinom.txt")
    # cq_true4.append(cq_true_)
    # cq_estimated4.append(cq_estimated_)
    # oq_true4.append(oq_true_)
    # oq_estimated4.append(oq_estimated_)
    # sq_true4.append(sq_true_)
    # sq_estimated4.append(sq_estimated_)
    # cq_true_, cq_estimated_, oq_true_, oq_estimated_, sq_true_, sq_estimated_ = GetQuotients(
    #     "Results_DotModel/German_clean_German_clean/GOI/Taschenfaltenhyperplasie.txt")
    # cq_true4.append(cq_true_)
    # cq_estimated4.append(cq_estimated_)
    # oq_true4.append(oq_true_)
    # oq_estimated4.append(oq_estimated_)
    # sq_true4.append(sq_true_)
    # sq_estimated4.append(sq_estimated_)

    # ax = plt.subplot(413)
    # ax.set_title("Quotients")
    # ax.boxplot([cq_true0, cq_true1, cq_true2, cq_true3, cq_true4], sym=" ")
    # plt.xticks([1, 2, 3, 4, 5], ["CMU clean",
    #                              "CMU 0dB white", "CMU 0dB babble", "Irish", "German"])

    # ax = plt.subplot(414)
    # ax.set_title("GCI (identification rate)")
    # ax.boxplot([idr0, idr1, idr2, idr3, idr4], sym=" ")
    # plt.xticks([1, 2, 3, 4, 5], ["CMU clean",
    #                              "CMU 0dB white", "CMU 0dB babble", "Irish", "German"])
    plt.show()


if(__name__ == "__main__"):
    main()
