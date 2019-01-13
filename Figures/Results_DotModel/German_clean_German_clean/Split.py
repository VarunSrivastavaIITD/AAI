file = open("Results_DotModel/German_clean_German_clean/Quotients.txt").readlines()

pathology = file[0].rstrip()
file1 = open("Results_DotModel/German_clean_German_clean/Quotients/" +
             pathology + ".txt", "w")
data = []
i = 1
while(i < len(file)):
    line = file[i]

    if(".npy" in line):
        data.append(line)
    else:
        i += 1
        data.append("\n")
        data.append(file[i])
        file1.writelines(data)

        data = []
        file1.close()

        i += 2
        pathology = file[i].rstrip()

        try:
            file1 = open(
                "Results_DotModel/German_clean_German_clean/Quotients/" + pathology + ".txt", "w")
        except:
            exit(0)

    i += 1
