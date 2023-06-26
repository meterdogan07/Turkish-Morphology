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


def processOutputFile2(filename, char_dict):
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
        if(line[:11] == "IndexedDict"):
            continue
            
        pred_first = line.find("] => trues: ")
        pred = line[9:pred_first]
        pred = pred.split("; ")

        predx = []
        for i in pred:
            w = []
            for c in i.split(" "):
                if(c == "3"):
                    break
                else:
                    w.append(char_dict[int(c)-1])
            predx.append(w)
        pred = predx

        true = line[pred_first+13:-1]
        true = true.split("; ")

        truex = []
        for i in true:
            w = []
            for c in i.split(" "):
                if(c == "3"):
                    break
                else:
                    w.append(char_dict[int(c)-1])
            truex.append(w)
        true = truex

        if(len(true)>20):
            totalsent += 1
            sent_flag = True
            ct_word += len(true)
            for i in range(len(true)):
                flag_word = True
                ct_char += len(true[i])

                total_edit += edit_distance(true[i], pred[i])
                edit_word_ct += 1

                true_lemma = [i for i in true[i] if len(i)==1]
                true_tags = [i for i in true[i] if len(i)!=1]

                pred_lemma = [i for i in pred[i] if len(i)==1]
                pred_tags = [i for i in pred[i] if len(i)!=1]

                check_word = True
                for c in range(min(len(true_lemma), len(pred_lemma))):
                    if(true_lemma[c] == "❓"):
                        ct_char -= 1
                        check_word = False

                if(check_word):
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
                    #print("+".join(true[i]) + " => " + "+".join(pred[i]))

            if(sent_flag):
                st_true += 1

    print(word_true)
    print(ct_word)
    print(edit_word_ct)
    print(totalsent)
    print("Char Accuracy: %",100*char_true/ct_char)
    print("Word Accuracy: %",100*word_true/ct_word)
    print("Sentence Accuracy: %",100*st_true/totalsent)
    print("Edit Distance:", total_edit/edit_word_ct)

    return [100*char_true/ct_char, 100*word_true/ct_word, 100*st_true/totalsent, total_edit/edit_word_ct]

d1 = ["❓", "⭕️", "🏁", "🎬", "ö", "z", "e", "t", "a", "b", "l", "o", "i", "s", "ü", "y", "g", "n", "ş", "v", "d", "r", ".", "ı", "k", "1", "7", "5", "c", "f", "3", "0", "p", "ç", "m", "u", ":", ",", "h", "j", "ğ", "-", ";", "\"", "!", "9", "6", "2", "4", "8", "w", "'", "`", "/", "x", "(", ")", "*", "%", "=", "\$", "ý", "&", "+", "^", "ð", "[", "]", "þ", "{", "}", "Noun", "A3sg", "Pnon", "Nom", "P3sg", "Loc", "A3pl", "Acc", "Verb", "^DB", "Caus", "Pos", "Inf2", "Conj", "Punct", "Num", "Card", "Dat", "Adj", "PastPart", "P2pl", "Postp", "PCDat", "Adverb", "AfterDoingSo", "Gen", "Pron", "Demons", "Prog1", "Det", "Zero", "Past", "Acquire", "Imp", "A2pl", "P1pl", "Pers", "With", "Pass", "PresPart", "Related", "Ins", "P1sg", "Neg", "Narr", "Abl", "Inf1", "Aor", "A1sg", "Pres", "FutPart", "Cond", "Fut", "Quant", "Reflex", "Rel", "PCNom", "ByDoingSo", "Become", "Cop", "P3pl", "Able", "AorPart", "P2sg", "Ness", "PCAbl", "Ques", "Without", "Desr", "A1pl", "A2sg", "Equ", "While", "NarrPart", "Ord", "Since", "Neces", "Agt", "When", "PCIns", "Prog2", "PCGen", "Opt", "Ly", "Dist", "i̇", "AsIf", "Inf3", "Dim", "Interj", "EverSince", "AsLongAs", "Real", "Time", "WithoutHavingDoneSo", "Recip", "NotState", "Hastily", "InBetween", "FeelLike", "Dup", "Adamantly", "WithoutBeingAbleToHaveDoneSo", "ActOf", "SinceDoingSo", "JustLike", "Inf", "Ratio", "Stay", "Distrib"]
d2 = ["❓", "⭕️", "🏁", "🎬", "t", "a", "b", "i", "d", "v", "e", "n", "f", "r", ",", "o", "y", "ı", "m", "s", "l", ".", "ü", "g", "u", "ç", "k", "ş", "z", "p", ";", "h", "c", "ğ", "x", "ö", ":", "\"", "1", "9", "0", "3", "j", "!", "4", "'", "-", "`", "/", "5", "(", ")", "2", "8", "7", "6", "*", "+", "w", "^", "%", "&", "=", "\$", "ý", "{", "}", "[", "]", "þ", "ð", "Adj", "Noun", "A3sg", "Pnon", "Abl", "Conj", "Nom", "Verb", "Pos", "Prog1", "A3pl", "Punct", "Pron", "Pers", "Gen", "^DB", "Zero", "P3sg", "Acc", "Caus", "PresPart", "Det", "Ness", "Neg", "P1pl", "Loc", "Ins", "With", "A1pl", "Pres", "Quant", "P3pl", "Postp", "PCDat", "Adverb", "A1sg", "P1sg", "Pass", "Fut", "PCNom", "Cop", "PastPart", "FutPart", "Inf1", "Dat", "Able", "Narr", "Aor", "Inf2", "Past", "Agt", "P2pl", "Imp", "A2pl", "Ques", "Cond", "While", "Acquire", "Rel", "Num", "Card", "Demons", "AfterDoingSo", "Opt", "ByDoingSo", "AsLongAs", "Become", "PCGen", "Prog2", "PCAbl", "Inf3", "P2sg", "NarrPart", "Desr", "Ord", "A2sg", "AorPart", "Without", "Interj", "i̇", "Neces", "Real", "Time", "Dim", "WithoutBeingAbleToHaveDoneSo", "Reflex", "AsIf", "Ly", "PCIns", "SinceDoingSo", "When", "Dist", "Equ", "JustLike", "Dup", "WithoutHavingDoneSo", "Since", "Related", "FeelLike", "InBetween", "NotState", "EverSince", "Hastily", "Stay", "Recip", "Ratio", "ActOf", "Distrib", "Adamantly", "Inf"]
d3 = ["❓", "⭕️", "🏁", "🎬", "y", "e", "t", "k", "i", "d", "p", "o", "a", "n", "ı", "m", "l", "g", "ö", "r", "v", "s", ".", "b", "z", "ü", "ç", "u", "ş", ",", "ğ", "h", "c", "\"", "f", "*", ";", "'", "!", ":", "1", "5", "0", "6", "3", "j", "4", "2", "-", "8", "ð", "w", "9", "7", "(", ")", "/", "`", "x", "%", "=", "^", "[", "]", "+", "&", "{", "}", "ý", "\$", "þ", "Noun", "A3sg", "Pnon", "Nom", "^DB", "Verb", "Acquire", "Caus", "Pos", "Inf2", "P3sg", "Loc", "Pass", "Adj", "PresPart", "A3pl", "Acc", "Aor", "Punct", "Abl", "Det", "Conj", "Past", "Agt", "Gen", "Dat", "Inf1", "Postp", "PCNom", "Imp", "A2pl", "Inf3", "Adverb", "Ins", "Able", "Dim", "Pron", "Pers", "A1sg", "Num", "Card", "Zero", "Prog1", "With", "PCDat", "Ques", "Pres", "Ness", "Cond", "FutPart", "P1sg", "P3pl", "Rel", "Neg", "While", "PastPart", "P2pl", "i̇", "Reflex", "Demons", "Without", "A2sg", "Narr", "Fut", "PCAbl", "Cop", "InBetween", "ByDoingSo", "A1pl", "P1pl", "AorPart", "Quant", "P2sg", "Interj", "PCGen", "Real", "Ord", "AsLongAs", "Become", "AsIf", "Inf", "Desr", "AfterDoingSo", "WithoutHavingDoneSo", "NarrPart", "When", "Equ", "PCIns", "Neces", "Opt", "FeelLike", "Prog2", "Dup", "Since", "Dist", "Related", "Hastily", "NotState", "Ly", "Ratio", "SinceDoingSo", "Recip", "Stay", "WithoutBeingAbleToHaveDoneSo", "ActOf", "Time", "JustLike", "EverSince", "Distrib", "Adamantly"]
v1 = []
v1.append(processOutputFile2("/kuacc/users/merdogan18/hpc_run/TrMor2023/evaluations/Morse/morse_outnum.txt", d1))
v1.append(processOutputFile2("/kuacc/users/merdogan18/hpc_run/TrMor2023/evaluations/Morse/morse_outnum1.txt", d2))
v1.append(processOutputFile2("/kuacc/users/merdogan18/hpc_run/TrMor2023/evaluations/Morse/morse_outnum2.txt", d3))
print(v1)