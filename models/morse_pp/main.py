import argparse, torch, json, matplotlib, logging, os
import torch.nn as nn
from torch import optim
import matplotlib.pyplot as plt
from dataloader import Parser, WordLoader, DataLoader, Vocab
from model.morsepp_model import Morse
from training import train
from utils import *
from training import *


#### DON'T FORGET TO CHANGE THIS !!! ####
task_no = 0
tasks = ["Morse"]
logger_file   = 'exp_split_1xvm1nolinear'            # Add ExpNUMBER !!!   
task_name = tasks[task_no]    # Add ExpNUMBER !!!
logger_folder_name = "loggers/"+task_name
save_dir = logger_folder_name+"/"+logger_file
#########################################


# Loggers
if not os.path.exists(logger_folder_name):
    os.mkdir(logger_folder_name)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s | | %(levelname)s | | %(message)s')

logger_file_name = os.path.join(logger_folder_name, logger_file)
file_handler = logging.FileHandler(logger_file_name,'w')
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.info('Code started \n')

# configuration
parser = argparse.ArgumentParser(description='')
args = parser.parse_args()

dtype = torch.float32 # we will be using float throughout this tutorial
args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#args.device = torch.device("cpu")

# training
args.task = task_name
args.seq_to_no_pad = 'surface'
args.batchsize = 8
args.epochs = 300
args.lr = 5e-4
args.warmup_steps = 4000
args.embed_dim = 256
args.src_hid_size = 256
args.src_nb_layers = 6
args.trg_hid_size = 256
args.trg_nb_layers = 6
args.dropout_p = 0.3
args.label_smooth = 0
args.nbheads = 16

CLIP = 1
TRG_PAD_IDX = 0

logger.info(f"Task: {args.task}")
logger.info(f"Using device: {str(args.device)}")
logger.info("We now give the hyperparameters")
logger.info(f"Number of Epochs: {args.epochs}")
logger.info(f"Batch Size: {args.batchsize}")
logger.info(f"Learning rate: {args.lr}")
logger.info(f"warmup_steps: {args.warmup_steps}")
logger.info(f"Embed dim: {args.embed_dim}")
logger.info(f"src_hid_size: {args.src_hid_size}")
logger.info(f"src_nb_layers: {args.src_nb_layers}")
logger.info(f"trg_hid_size: {args.trg_hid_size}")
logger.info(f"trg_nb_layers: {args.trg_nb_layers}")
logger.info(f"nbheads: {args.nbheads}")
logger.info(f"dropout_p: {args.dropout_p}")

#--------------------------------------------------------------------------------------------------
# Data Loader
trmor         = Parser(sentence_start="<S", sentence_end="</S", part_seperator="\t", tag_seperator="+", unk_token="?")
train_data    = trmor.parse_file("/kuacc/users/merdogan18/hpc_run/TrMor2023/data/xtrain_data1.txt")
all_data    = trmor.parse_file("/kuacc/users/merdogan18/hpc_run/TrMor2023/data/xall_data1.txt")
val_data      = trmor.parse_file("/kuacc/users/merdogan18/hpc_run/TrMor2023/data/xval_data1.txt")
test_data     = trmor.parse_file("/kuacc/users/merdogan18/hpc_run/TrMor2023/data/xtest_data1.txt")

logger.info(f"Train Data Length: {len(train_data)}")
logger.info(f"Valid Data Length: {len(val_data)}")
logger.info(f"Test Data Length: {len(test_data)}")

vocab = Vocab(all_data,pad_feat_to=64, pad_surf_to=64)
dataset_train = WordLoader(vocab, train_data)
dataset_val   = WordLoader(vocab, val_data)
dataset_test  = WordLoader(vocab, test_data)
surf_vocab    = vocab.surf_decoder
feature_vocab = vocab.feat_decoder

train_loader  = DataLoader(dataset_train, batch_size=args.batchsize, shuffle=True, collate_fn=collate_morse)
val_loader    = DataLoader(dataset_val, batch_size=args.batchsize, shuffle=False, collate_fn=collate_morse)
test_loader   = DataLoader(dataset_test, batch_size=args.batchsize, shuffle=False, collate_fn=collate_morse)

args.src_vocab_size = len(surf_vocab)
args.trg_vocab_size = len(feature_vocab)

logger.info(f"src_vocab_size: {args.src_vocab_size}")
logger.info(f"trg_vocab_size: {args.trg_vocab_size}")

#--------------------------------------------------------------------------------------------------

model = Morse(args.src_vocab_size, args.trg_vocab_size, args.embed_dim, args.nbheads, args.src_hid_size, args.src_nb_layers, 
                    args.trg_hid_size, args.trg_nb_layers, args.dropout_p, args.device, args.batchsize).to(args.device)

optimizer = optim.AdamW(model.parameters(), lr = args.lr)
scheduler = WarmupInverseSquareRootSchedule(optimizer, args.warmup_steps)
criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX)

best_valid_loss = float('inf')
best_valid_acc = 0

#--------------------------------------------------------------------------------------------------

for epoch in range(args.epochs):
    print(epoch)
    start_time = time.time()
    train_loss = train(model, train_loader, optimizer, args.device, scheduler, criterion, CLIP, logger)
    
    if((epoch+1) % 10 == 0):
        valid_loss, valid_acc, word_acc, valid_edit = evaluate(model, val_loader, criterion, args.device, feature_vocab, savedir = save_dir+"_Val")
        logger.info(f"\tTrain Loss: {train_loss:.3f} | Train Acc: {valid_acc:.3f} | Word Acc: {word_acc:.3f}")
        logger.info(f"\tValid Loss: {valid_loss:.3f} | Levenstein: {valid_edit:.3f} | Valid Acc: {valid_acc:.3f} | Word Acc: {word_acc:.3f}")
        
        if word_acc > best_valid_acc:
            best_valid_acc = word_acc
            torch.save(model.state_dict(), save_dir+'_model_dict.pt')
    
    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    logger.info(f"Epoch: {epoch+1:02} | Time {epoch_mins}m {epoch_secs}s | lr: {scheduler.get_last_lr()[0]} | loss: {train_loss}")

model.load_state_dict(torch.load(save_dir+'_model_dict.pt'))
logger.info(f"Training ended. Test set evaluation:")
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