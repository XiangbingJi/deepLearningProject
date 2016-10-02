from os import listdir
from os.path import isfile, join

if __name__ == "__main__":
    mypath = "ministdata/"
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

    datadict = {}
    for files in onlyfiles:
        if files[-3:]=="txt":
            datadict[files] = []
            with open("ministdata/"+files,"r") as f:
                for lines in f:
                    lines = lines.strip()
                    elem = lines.split("  ")
                    #print elem
                    if elem[0] == "test error rate:":
                        d = float(elem[-1][:-1])
                        datadict[files].append(d*0.01)

    for k,v in datadict.iteritems():
        print k, len(v)
        print v

    f1 = open("plot.csv", "w")
    f1.write(" , ")
    for i in range(249):
        f1.write(str(i+1)+", ")
    f1.write("250\n")


    for k,v in datadict.iteritems():
        f1.write(k+", ")
        i = 0
        for ele in v:
            if i<249:
                f1.write(str(ele)+", ")
            else:
                f1.write(str(ele)+"\n")
            i += 1
