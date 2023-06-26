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
tasks = ["Morse"]
logger_file   = 'exp_last2xx2'            # Add ExpNUMBER !!!         
task_name = tasks[task_no]    # Add ExpNUMBER !!!
logger_folder_name = "loggers/"+task_name
save_dir = logger_folder_name+"/"+logger_file
#########################################

# configuration
parser = argparse.ArgumentParser(description='')
args = parser.parse_args()

dtype = torch.float32 # we will be using float throughout this tutorial
args.device = torch.device("cpu") #torch.device("cuda" if torch.cuda.is_available() else "cpu")
#args.device = torch.device("cpu")

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
train_data    = trmor.parse_file("/kuacc/users/merdogan18/hpc_run/TrMor2023/data/xtrain_data2.txt")
all_data    = trmor.parse_file("/kuacc/users/merdogan18/hpc_run/TrMor2023/data/xall_data2.txt")
val_data      = trmor.parse_file("/kuacc/users/merdogan18/hpc_run/TrMor2023/data/xval_data2.txt")
test_data     = trmor.parse_file("/kuacc/users/merdogan18/hpc_run/TrMor2023/data/xtest_data2.txt")

vocab = Vocab(all_data)
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
"""
if(task_no == 0):
    model = Seq2SeqLSTM(args.src_vocab_size, args.trg_vocab_size, args.embed_dim, args.src_hid_size, args.src_nb_layers, 
                        args.trg_hid_size, args.trg_nb_layers, args.dropout_p, args.device).to(args.device)
elif(task_no == 1):
    model = SeqtoSeqAttentionLSTM(args.src_vocab_size, args.trg_vocab_size, args.embed_dim, args.src_hid_size, args.src_nb_layers, 
                        args.trg_hid_size, args.trg_nb_layers, args.dropout_p, args.device).to(args.device)
elif(task_no == 2):
    model = Transformer(args.src_vocab_size, args.trg_vocab_size, args.embed_dim, args.nbheads, args.src_hid_size, args.src_nb_layers, 
                        args.trg_hid_size, args.trg_nb_layers, args.dropout_p, args.device).to(args.device)
"""
#model.load_state_dict(torch.load("/kuacc/users/merdogan18/hpc_run/TrMor2023/versions/non_context/deneme_dict.pt", map_location=torch.device('cpu')))
model.load_state_dict(torch.load(save_dir+'_model_dict.pt'))
model.eval()

with torch.no_grad():
    #train_loss, train_acc, train_word_acc, train_edit = evaluate(model, train_loader, criterion, args.device, feature_vocab, savedir = save_dir+"_Train")
    #logger.info(f"\tTest Loss: {train_loss:.3f} | Levenstein: {train_edit:.3f} | Train Acc: {train_acc:.3f} | Train Word Acc: {train_word_acc:.3f}")
    valid_loss, valid_acc, valid_word_acc, valid_edit = evaluate(model, val_loader, criterion, args.device, feature_vocab, savedir = save_dir+"_Val")
    logger.info(f"\tValid Loss: {valid_loss:.3f} | Levenstein: {valid_edit:.3f} | Valid Acc: {valid_acc:.3f} | Valid Word Acc: {valid_word_acc:.3f}")
    test_loss, test_acc, test_word_acc, test_edit = evaluate(model, test_loader, criterion, args.device, feature_vocab, savedir = save_dir+"_Test")
    logger.info(f"\tTest Loss: {test_loss:.3f} | Levenstein: {test_edit:.3f} | Test Acc: {test_acc:.3f} | Test Word Acc: {test_word_acc:.3f}")

    #print(f"\tTrain Loss: {train_loss:.3f} | Levenstein: {train_edit:.3f} | Train Acc: {train_acc:.3f} | Train Word Acc: {train_word_acc:.3f}")
    print(f"\tValid Loss: {valid_loss:.3f} | Levenstein: {valid_edit:.3f} | Valid Acc: {valid_acc:.3f} | Valid Word Acc: {valid_word_acc:.3f}")
    print(f"\tTest Loss: {test_loss:.3f} | Levenstein: {test_edit:.3f} | Test Acc: {test_acc:.3f} | Test Word Acc: {test_word_acc:.3f}")