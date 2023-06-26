import argparse, torch, json, matplotlib, logging, os
import torch.nn as nn
from torch import optim
import matplotlib.pyplot as plt
from dataloader import Parser, WordLoader, DataLoader, Vocab
from model.LSTM1 import *
from model.Transformer1 import Transformer
from training import train
from utils import *
from training import *


#### DON'T FORGET TO CHANGE THIS !!! ####
task_no = 0
tasks = ["Seq2Seqv2", "Seq2SeqAttentionv2", "Transformerv2"]
logger_file   = 'exp_1'            # Add ExpNUMBER !!!         
task_name = tasks[task_no]    # Add ExpNUMBER !!!
logger_folder_name = "loggers/"+task_name
save_dir = logger_folder_name+"/"+logger_file
#########################################

# configuration
parser = argparse.ArgumentParser(description='')
args = parser.parse_args()

dtype = torch.float32 # we will be using float throughout this tutorial
args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# training
args.task = task_name
args.seq_to_no_pad = 'surface'
args.batchsize = 128
args.epochs = 200

if(task_no != 2):
    args.lr = 1e-3
    args.warmup_steps = 4000
    args.embed_dim = 256
    args.src_hid_size = 256
    args.src_nb_layers = 2
    args.trg_hid_size = 256
    args.trg_nb_layers = 2
    args.dropout_p = 0.3
else:
    args.lr = 5e-4
    args.warmup_steps = 4000
    args.embed_dim = 256
    args.src_hid_size = 256
    args.src_nb_layers = 4
    args.trg_hid_size = 256
    args.trg_nb_layers = 4
    args.dropout_p = 0.3

args.label_smooth = 0
args.nbheads = 16

CLIP = 1
TRG_PAD_IDX = 0

#--------------------------------------------------------------------------------------------------
# Data Loader
trmor         = Parser(sentence_start="<S", sentence_end="</S", part_seperator="\t", tag_seperator="+", unk_token="?")
train_data    = trmor.parse_file("/kuacc/users/merdogan18/hpc_run/TrMor2023/data/train_final.txt")
val_data      = trmor.parse_file("/kuacc/users/merdogan18/hpc_run/TrMor2023/data/val_final.txt")
test_data     = trmor.parse_file("/kuacc/users/merdogan18/hpc_run/TrMor2023/data/test_final.txt")

vocab = Vocab(train_data)
dataset_train = WordLoader(vocab, train_data)
dataset_val   = WordLoader(vocab, val_data)
dataset_test  = WordLoader(vocab, test_data)

surf_vocab    = vocab.surf_decoder
feature_vocab = vocab.feat_decoder

train_loader  = DataLoader(dataset_train, batch_size=args.batchsize, shuffle=False, collate_fn=collate_wordbased)
val_loader    = DataLoader(dataset_val, batch_size=args.batchsize, shuffle=False, collate_fn=collate_wordbased)
test_loader   = DataLoader(dataset_test, batch_size=args.batchsize, shuffle=False, collate_fn=collate_wordbased)

args.src_vocab_size = len(surf_vocab)
args.trg_vocab_size = len(feature_vocab)
criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX)

if(task_no == 0):
    model = Seq2SeqLSTM(args.src_vocab_size, args.trg_vocab_size, args.embed_dim, args.src_hid_size, args.src_nb_layers, 
                        args.trg_hid_size, args.trg_nb_layers, args.dropout_p, args.device).to(args.device)
elif(task_no == 1):
    model = SeqtoSeqAttentionLSTM(args.src_vocab_size, args.trg_vocab_size, args.embed_dim, args.src_hid_size, args.src_nb_layers, 
                        args.trg_hid_size, args.trg_nb_layers, args.dropout_p, args.device).to(args.device)
elif(task_no == 2):
    model = Transformer(args.src_vocab_size, args.trg_vocab_size, args.embed_dim, args.nbheads, args.src_hid_size, args.src_nb_layers, 
                        args.trg_hid_size, args.trg_nb_layers, args.dropout_p, args.device).to(args.device)


model.load_state_dict(torch.load("/kuacc/users/merdogan18/hpc_run/TrMor2023/versions/non_context/"+save_dir+"_model_dict.pt", map_location=args.device))
model.eval()

with torch.no_grad():
    
    #train_loss, train_acc, train_word_acc, train_edit = evaluate(model, train_loader, criterion, args.device, feature_vocab, savedir = save_dir+"_Train2")
    #print(f"\tTest Loss: {train_loss:.6f} | Levenstein: {train_edit:.6f} | Train Acc: {train_acc:.6f} | Train Word Acc: {train_word_acc:.6f}")

    valid_loss, valid_acc, valid_word_acc, valid_edit = evaluate(model, val_loader, criterion, args.device, feature_vocab, savedir = save_dir+"_Val2")
    print(f"\tValid Loss: {valid_loss:.6f} | Levenstein: {valid_edit:.6f} | Valid Acc: {valid_acc:.6f} | Valid Word Acc: {valid_word_acc:.6f}")
    
    test_loss, test_acc, test_word_acc, test_edit = evaluate(model, test_loader, criterion, args.device, feature_vocab, savedir = save_dir+"_Test2")
    print(f"\tTest Loss: {test_loss:.6f} | Levenstein: {test_edit:.6f} | Test Acc: {test_acc:.6f} | Test Word Acc: {test_word_acc:.6f}")


def sentence_accuracy(save_dir, set):
    if(set == "Train2"):
        data = train_data
    elif(set == "Test2"):
        data = test_data
    elif(set == "Val2"):
        data = val_data
    data_len = len(data)
    ct = 0
    ct_word = 0
    word_true = 0
    st_true = 0
    with open("/kuacc/users/merdogan18/hpc_run/TrMor2023/versions/non_context/"+save_dir+"_"+set+"_all_predictions.txt", "r", encoding="utf-8") as fp:
        for line in fp.readlines():
            linef = line.strip()
            st_len = (len(data[ct]))
            if(linef == ""):
                continue
            if(ct_word == st_len):
                if(st_len == word_true):
                    st_true += 1
                ct_word = 0
                word_true = 0
                ct += 1

            ind = linef.find('| Pred: ')
            tru = linef[6:ind-1]
            pred = linef[ind+8:]
            if(tru == pred):
                word_true += 1
            ct_word += 1

        print(set + " sentence accuracy: ",str(100*st_true/data_len))
    return st_true/data_len

#sentence_accuracy(save_dir, "Train")
sentence_accuracy(save_dir, "Test2")
sentence_accuracy(save_dir, "Val2")
