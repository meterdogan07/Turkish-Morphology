from dataloader import Parser, WordLoader, DataLoader, Vocab
import argparse, json, matplotlib, logging, os
import matplotlib.pyplot as plt
from dataloader import Parser, WordLoader, DataLoader, Vocab
import numpy as np
import io

def edit_distance(str1, str2):
    """Simple Levenshtein implementation for evalm."""
    table = np.zeros([len(str2) + 1, len(str1) + 1])
    for i in range(1, len(str2) + 1):
        table[i][0] = table[i - 1][0] + 1
    for j in range(1, len(str1) + 1):
        table[0][j] = table[0][j - 1] + 1
    for i in range(1, len(str2) + 1):
        for j in range(1, len(str1) + 1):
            if str1[j - 1] == str2[i - 1]:
                dg = 0
            else:
                dg = 1
            table[i][j] = min(
                table[i - 1][j] + 1, table[i][j - 1] + 1, table[i - 1][j - 1] + dg
            )
    return int(table[len(str2)][len(str1)])  


def processOutputFile2(filename):
    digits = "0123456789"
    ct_word = 0
    totalsent = 0
    correctsent = 0
    ct_char = 0
    char_true = 0

    word_true = 0
    st_true = 0
    total_edit = 0
    edit_word_ct = 0

    f = io.open(filename,encoding="utf8")
    for x in f:
        line = x.strip()
        if(line == ""):
            continue
            
        totalsent += 1
        if(totalsent == 200):
            break
        pred_first = line.find(" => ")
        true = line[0:pred_first]
        pred = line[pred_first+4:]          

        true_w = true.split("~")
        pred_w = pred.split("~")
        
        sent_flag = True
        ct_word += len(true_w)
        
        for i in range(min(len(pred_w),len(true_w))):
            flag_word = True
            true_word = [c for c in true_w[i].split("+") if c != ""]
            pred_word = [c for c in pred_w[i].split("+") if c != ""]
            ct_char += len(true_word)

            total_edit += edit_distance(true_word, pred_word)
            edit_word_ct += 1

            true_lemma = [i for i in true_word if len(i)==1]
            true_tags = [i for i in true_word if len(i)!=1]

            pred_lemma = [i for i in pred_word if len(i)==1]
            pred_tags = [i for i in pred_word if len(i)!=1]

            for c in range(min(len(true_lemma), len(pred_lemma))):
                if(true_lemma[c] == pred_lemma[c]):
                    char_true += 1
                else:
                    flag_word = False

            for c in range(min(len(true_tags), len(pred_tags))):
                if(true_tags[c] == pred_tags[c]):
                    char_true += 1
                else:
                    flag_word = False
            if(len(pred_lemma)!=len(true_lemma) or len(pred_tags)!=len(true_tags)):
                flag_word = False

            if(flag_word):
                word_true += 1
            else:
                sent_flag = False

        if(sent_flag):
            st_true += 1

    print(word_true)
    print(ct_word)
    print(edit_word_ct)
    print("Char Accuracy: %",100*char_true/ct_char)
    print("Word Accuracy: %",100*word_true/ct_word)
    print("Sentence Accuracy: %",100*st_true/totalsent)
    print("Edit Distance:", total_edit/ct_word)

    return [100*char_true/ct_char, 100*word_true/ct_word, 100*st_true/totalsent, total_edit/ct_word]

v1 = []
v1.append(processOutputFile2("/kuacc/users/merdogan18/hpc_run/TrMor2023/versions/prefix-tuning/xlastcontext_out_test9_run.txt"))
print(v1)