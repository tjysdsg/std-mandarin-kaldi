import os
import sys


phone_dir = sys.argv[1]
ctm = sys.argv[2]
phone_ctm = sys.argv[3]

dict_phones = dict({item.split()[1]: item.split()[0] for item in open(phone_dir)})

f1 = open(phone_ctm,"w")


for item in open(ctm):
    data = item.strip().split()
    # print(dict_phones)
    data[-1] = dict_phones[data[-1]]
    f1.writelines(" ".join(data) + "\n")
f1.close()


