import argparse, json, matplotlib, logging, os
import matplotlib.pyplot as plt
from dataloader import Parser
import numpy as np


#--------------------------------------------------------------------------------------------------
# Data Loader
trmor         = Parser(sentence_start="<S", sentence_end="</S", part_seperator="\t", tag_seperator="+", unk_token="?")
train_data    = trmor.parse_file("/kuacc/users/merdogan18/hpc_run/TrMor2023/data/trmor2018.txt")

#train_data    = trmor.parse_file("/kuacc/users/merdogan18/hpc_run/TrMor2023/data/train_deneme.txt")

print("Train Data Length: ", len(train_data))
#--------------------------------------------------------------------------------------------------
data = []
ct_unk = 0
ct_wordlen = 0
ct_inpcharlen = 0
ct_outcharlen = 0
tot_inpcharlen = 0
tot_outcharlen = 0
tot_outstrlen = 0

for i in range(len(train_data)):
    put = True
    w_sent = len(train_data[i])
    if(len(train_data[i]) > 64 or len(train_data[i]) < 3):
        put = False
        ct_wordlen += 1
    inp_toks = 0
    out_toks = 0
    out_strlen = 0
    for tg in range(len(train_data[i])):
        inp_toks += len(train_data[i][tg][0])
        out_toks += len(train_data[i][tg][1])
        out_strlen += 1
        for out_str in train_data[i][tg][1]:
            out_strlen += len(out_str)
        if(train_data[i][tg][1] == "?" or len(train_data[i][tg][1]) == 1):
            put = False
            ct_unk += 1
        if(len(train_data[i][tg][0])>64):
            put = False
            ct_inpcharlen += 1
        if(len(train_data[i][tg][1])>64):
            put = False
            ct_outcharlen += 1
    if(inp_toks > 256):
        put = False
        tot_inpcharlen += 1
    if(out_toks > 256):
        put = False
        tot_outcharlen += 1
    if(out_strlen > 512):
        put = False
        tot_outstrlen += 1

    if(put):
        data.append(train_data[i])

split = 2
np.random.seed(split)
indexes = np.random.randint(10, size = len(data))

f1 = open("xtest_data"+str(split)+".txt", "w", encoding="utf-8")
f2 = open("xval_data"+str(split)+".txt", "w", encoding="utf-8")
f3 = open("xtrain_data"+str(split)+".txt", "w", encoding="utf-8")
f4 = open("xall_data"+str(split)+".txt", "w", encoding="utf-8")
f5 = open("xmorse_data"+str(split)+".txt", "w", encoding="utf-8")

def write_instance(file, sentence):
    file.write("<S>\n")
    for s in sentence:
        lemma = "".join(s[0])
        index = 0
        for i in range(len(s[1])):
            index+=1
            if(len(s[1][i])>1):
                break
        root = "".join(s[1][:index-1])
        tags = "+".join(s[1][index-1:]).replace("+^","^")
        file.write(lemma + "	"+root+"+"+tags+"	"+"xxxx\n")
    file.write("</S>\n")


for i, prob in enumerate(indexes):
    if(prob == 9):
        write_instance(f1, data[i])
    elif(prob == 8):
        write_instance(f2, data[i])
        write_instance(f5, data[i])
    else:
        write_instance(f3, data[i])
        write_instance(f5, data[i])

for i in range(len(data)):
    write_instance(f4, data[i])

f1.close()
f2.close()
f3.close()

print(ct_unk)
print(ct_wordlen)
print(ct_inpcharlen)
print(ct_outcharlen)
print(tot_inpcharlen)
print(tot_outcharlen)
print(tot_outstrlen)
print(len(data))
