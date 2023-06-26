import argparse, json, matplotlib, logging, os
import matplotlib.pyplot as plt
from dataloader import Parser
import numpy as np


#--------------------------------------------------------------------------------------------------
# Data Loader
trmor         = Parser(sentence_start="<S", sentence_end="</S", part_seperator="\t", tag_seperator="+", unk_token="?")
#train_data    = trmor.parse_file("/kuacc/users/merdogan18/hpc_run/TrMor2023/data/trmor2018.txt")

train_data    = trmor.parse_file("/kuacc/users/merdogan18/hpc_run/TrMor2023/data/xall_data0.txt")

print("Train Data Length: ", len(train_data))
#--------------------------------------------------------------------------------------------------
data = []
ct_unk = 0
ct_words = 0
ct_wordlen = 0
ct_inpcharlen = 0
ct_outcharlen = 0
tot_inpcharlen = 0
tot_outcharlen = 0
tot_outstrlen = 0

inp_toks = 0
out_toks = 0

unk_sent = 0
inps = []
outs = []

for i in range(len(train_data)):
    unk_flag = False
    w_sent = len(train_data[i])
    for tg in range(len(train_data[i])):
        ct_words += 1
        inp_toks += len(train_data[i][tg][0])
        out_toks += len(train_data[i][tg][1])
        inps.extend(train_data[i][tg][0])
        outs.extend(train_data[i][tg][1])
        if(train_data[i][tg][1] == "?" or len(train_data[i][tg][1]) == 1):
            ct_unk += 1
            unk_flag = True
    if(unk_flag):
        unk_sent+=1


print(len(train_data))
print(ct_words)
print(ct_unk)
print(unk_sent)

print(inp_toks)
print(out_toks)
print(len(set(inps)))
print(len(set(outs)))

print(set(inps))