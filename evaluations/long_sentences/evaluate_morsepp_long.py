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

def processOutputFile(save_dir, data):
    digits = "0123456789"
    ct_word = 0
    ct_char = 0
    char_true = 0

    word_true = 0
    st_true = 0
    total_edit = 0
    sent_total = 0
    edit_word_ct = 0
    ct_word_total = 0
    ct = 0
    data_len = len(data)
    word_true_in_sent = 0
    sentences = []
    with open(save_dir, "r", encoding="utf-8") as fp:
        sentence = []
        for line in fp.readlines():
            linef = line.strip()
            st_len = (len(data[ct]))
            if(linef == ""):
                continue
            if(ct_word == st_len):
                sentences.append(sentence)
                sentence = []
                ct_word = 0
                ct += 1
                
            ind = linef.find('| Pred: ')
            tru = linef[6:ind-1]
            pred = linef[ind+8:]
            if("~" in tru):
                sep_token = "~"
            else:
                sep_token = "+"

            true_word = [c for c in tru.split(sep_token) if c != ""]
            pred_word = [c for c in pred.split(sep_token) if c != ""]
            sentence.append((true_word, pred_word))
            ct_word += 1
            
        ct = 0
        for sent in sentences:
            if(len(sent)<20):
                continue
            ct += 1
            sent_flag = True
            for word in sent:
                true_word = word[0]
                pred_word = word[1]

                ct_char += len(true_word)
                total_edit += edit_distance(true_word, pred_word)
                edit_word_ct += 1

                true_lemma = [i for i in true_word if len(i)==1]
                true_tags = [i for i in true_word if len(i)!=1]

                pred_lemma = [i for i in pred_word if len(i)==1]
                pred_tags = [i for i in pred_word if len(i)!=1]

                flag_word = True
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
                    word_true_in_sent += 1
                else:
                    sent_flag = False
                    
                ct_word_total += 1

            if(sent_flag):
                st_true += 1

    print(word_true)
    print(ct_word_total)
    print(edit_word_ct)
    print(ct)
    print("Char Accuracy: %",100*char_true/ct_char)
    print("Word Accuracy: %",100*word_true/ct_word_total)
    print("Sentence Accuracy: %",100*st_true/ct)
    print("Edit Distance:", total_edit/edit_word_ct)

    return [100*char_true/ct_char, 100*word_true/ct_word_total, 100*st_true/ct, total_edit/edit_word_ct]

trmor = Parser(sentence_start="<S", sentence_end="</S", part_seperator="\t", tag_seperator="+", unk_token="?")
v1,v2,v3 = [],[],[]

test_dir = "/kuacc/users/merdogan18/hpc_run/TrMor2023/data/xtest_data0.txt"
test_dir1 = "/kuacc/users/merdogan18/hpc_run/TrMor2023/data/xtest_data1.txt"
test_dir2 = "/kuacc/users/merdogan18/hpc_run/TrMor2023/data/xtest_data2.txt"

test_data = trmor.parse_file(test_dir)
test_data1 = trmor.parse_file(test_dir1)
test_data2 = trmor.parse_file(test_dir2)


dir1 = "/kuacc/users/merdogan18/hpc_run/TrMor2023/versions/morse_pp/loggers/Morse/exp_split_0_Test_all_predictions.txt"
dir2 = "/kuacc/users/merdogan18/hpc_run/TrMor2023/versions/morse_pp/loggers/Morse/exp_split_1_Test_all_predictions.txt"
dir3 = "/kuacc/users/merdogan18/hpc_run/TrMor2023/versions/morse_pp/loggers/Morse/exp_split_2_Test_all_predictions.txt"
v1.append(processOutputFile(dir1, test_data))
v2.append(processOutputFile(dir2, test_data1))
v3.append(processOutputFile(dir3, test_data2))

print(v1)
print(v2)
print(v3)




