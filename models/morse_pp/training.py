import random, torch, time, logging
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from utils import *


dtype = torch.float32 # we will be using float throughout this tutorial

def train(model, iterator, optimizer, device, scheduler, criterion, clip, logger):
    
    model.train()
    epoch_loss, cnt = 0, 0
    
    for i, batch in enumerate(iterator):
        if(i == (len(iterator)-1)):
            break
        print(len(iterator))
        #print("in epoch: ", i)
        src = batch[0].to(device)
        trg = batch[1].to(device)
        ix = batch[2].to(device)
        # trg = [sen_len, batch_size]
        # output = [trg_len, batch_size, output_dim]
        optimizer.zero_grad()
        loss = model.get_loss(src, trg, ix)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        scheduler.step()
        
        epoch_loss += loss.item()
        cnt += 1
        if(i%1 == 0):
            print("Step: ",i, " _ ","Loss: ",loss.item())
        if(i%50 == 0):
            logger.info(f"Step: {i} | Loss: {epoch_loss/cnt}")
    return epoch_loss/cnt


def evaluate(model, iterator, criterion, device, decoder, savedir):
    epoch_acc = 0
    model.eval()
    epoch_loss, cnt = 0, 0
    epoch_word_acc = 0
    epoch_editd = 0
    f1 = open(savedir+"_wrong_predictions.txt", "w")
    f2 = open(savedir+"_correct_predictions.txt", "w")
    f3 = open(savedir+"_all_predictions.txt", "w")
    with torch.no_grad():
        
        for i, batch in enumerate(iterator):
            if(i == (len(iterator)-1)):
                continue
            print(len(iterator))
            src = batch[0].to(device)
            trg = batch[1].to(device)
            ix = batch[2].to(device)
            
            batch_size = trg.shape[1]

            output = model(src, trg, ix, 0) # turn off teacher forcing.
            loss = model.get_loss(src, trg, ix)
            # trg = [sen_len, batch_size]
            # output = [sen_len, batch_size, output_dim]
            output = output[:-1]
            trg = trg[1:]
            word_acc = word_accuracy(output, trg, 0, batch_size, f1, f2, f3, decoder)
            #editd = edit_distance_batch(output, trg, 0, batch_size)

            #output = output.view(-1)
            #trg = trg.view(-1)
            acc = categorical_accuracy(output, trg, 0)    
            
            epoch_acc += acc.item()
            epoch_loss += loss.item()
            epoch_word_acc += word_acc
            #epoch_editd += editd
            cnt += 1
    f1.close(); f2.close(); f3.close()
    return epoch_loss/cnt, epoch_acc/cnt, epoch_word_acc/cnt, epoch_editd/cnt

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time  / 60)
    elapsed_secs = int(elapsed_time -  (elapsed_mins * 60))
    return  elapsed_mins, elapsed_secs

def categorical_accuracy(preds, y, tag_pad_idx):
    """
    Returns the categorical accuracy between predictions and the ground truth, ignoring pad tokens.
    """
    #max_preds = preds.argmax(dim = 1, keepdim = True) # get the index of the max probability
    not_padded = y != tag_pad_idx
    correct = preds[not_padded].eq(y[not_padded])
    accuracy = correct.sum() / y[not_padded].shape[0]
    return accuracy

def word_accuracy(preds, y, tag_pad_idx, batch_size, f1, f2, f3, decoder):
    """
    Returns the categorical accuracy between predictions and the ground truth, ignoring pad tokens.
    """
    #max_preds = preds.argmax(dim = 2) # get the index of the max probability
    not_padded = (y != tag_pad_idx)
    words = preds.view(y.shape)*not_padded
    gt = y*not_padded
    trues = 0

    for i in range(batch_size):
        pred = "~".join([decoder[int(j)] for j in words[:,i]]).replace("~<p>", "")
        gttt = "~".join([decoder[int(j)] for j in gt[:,i]]).replace("~<p>", "")
        flag = True
        if(len(words[:,i]) != len(gt[:,i])):
            flag = False
        for c in range(len(gt[:,i])):
            if(not decoder[int(gt[c,i])].isdigit()):
                if(not torch.equal(words[c,i], gt[c,i])):
                    flag = False
                    break
        if(flag):  #torch.equal(words[:,i],gt[:,i])
            trues += 1
            f2.write(f"True: {gttt} | Pred: {pred} \n \n")
        else:
            f1.write(f"True: {gttt} | Pred: {pred} \n \n")
        f3.write(f"True: {gttt} | Pred: {pred} \n \n")
    return trues/batch_size

def edit_distance_batch(preds, y, tag_pad_idx, batch_size):
    #max_preds = preds.argmax(dim = 2) # get the index of the max probability
    not_padded = y != tag_pad_idx
    words = preds.view(y.shape)*not_padded
    gt = y*not_padded
    total = 0
    for i in range(batch_size):
        total += edit_distance(words[:,i], gt[:,i])
    return total/batch_size
