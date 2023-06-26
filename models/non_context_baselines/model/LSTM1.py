import numpy as np
import math
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from utils import *

EPSILON = 1e-7
PAD_IDX = 0

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, num_layers=n_layers, bidirectional=False, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src : [sen_len, batch_size]
        embedded = self.dropout(self.embedding(src))
        
        # embedded : [sen_len, batch_size, emb_dim]
        outputs, (hidden, cell) = self.rnn(embedded)
        # outputs = [sen_len, batch_size, hid_dim * n_directions]
        # hidden = [n_layers * n_direction, batch_size, hid_dim]
        # cell = [n_layers * n_direction, batch_size, hid_dim]
        return hidden, cell
    
class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        
        self.output_dim = output_dim
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, num_layers=self.n_layers, dropout=dropout)
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, cell):
        
        # input = [batch_size]
        # hidden = [n_layers * n_dir, batch_size, hid_dim]
        # cell = [n_layers * n_dir, batch_size, hid_dim]
        
        input = input.unsqueeze(0)
        # input : [1, ,batch_size]
        embedded = self.dropout(self.embedding(input))
        # embedded = [1, batch_size, emb_dim]
        
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        # output = [seq_len, batch_size, hid_dim * n_dir]
        # hidden = [n_layers * n_dir, batch_size, hid_dim]
        # cell = [n_layers * n_dir, batch_size, hid_dim]
        
        # seq_len and n_dir will always be 1 in the decoder
        prediction = self.fc_out(output.squeeze(0))
        # prediction = [batch_size, output_dim]
        return prediction, hidden, cell
    
class Seq2SeqLSTM(nn.Module):
    def __init__(self, 
            src_vocab_size,
            trg_vocab_size,
            embed_dim,
            src_hid_size,
            src_nb_layers,
            trg_hid_size,
            trg_nb_layers,
            dropout_p,
            device):
        super().__init__()
        
        self.encoder = Encoder(src_vocab_size, embed_dim, src_hid_size, src_nb_layers, dropout_p)
        self.decoder = Decoder(trg_vocab_size, embed_dim, trg_hid_size, trg_nb_layers, dropout_p)
        self.scale_enc_hs = nn.Linear(src_hid_size * 2, trg_hid_size)
        self.device = device
        """        
        assert self.encoder.hid_dim == self.decoder.hid_dim, \
            'hidden dimensions of encoder and decoder must be equal.'
        assert self.encoder.n_layers == self.decoder.n_layers, \
            'n_layers of encoder and decoder must be equal.'
        """
        
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src = [sen_len, batch_size]
        # trg = [sen_len, batch_size]
        # teacher_forcing_ratio : the probability to use the teacher forcing.
        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        # tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        # last hidden state of the encoder is used as the initial hidden state of the decoder
        hidden, cell = self.encoder(src)

        # first input to the decoder is the <sos> token.
        input = trg[0, :]
        for t in range(trg_len):
            # insert input token embedding, previous hidden and previous cell states 
            # receive output tensor (predictions) and new hidden and cell states.
            output, hidden, cell = self.decoder(input, hidden, cell)
            # replace predictions in a tensor holding predictions for each token
            outputs[t] = output
            # decide if we are going to use teacher forcing or not.
            teacher_force = random.random() < teacher_forcing_ratio
            # get the highest predicted token from our predictions.
            top1 = output.argmax(1)
            # update input : use ground_truth when teacher_force 
            input = trg[t] if teacher_force else top1
        return outputs

##-------------------------------------------------------------------------------------------------


class StackedLSTM(nn.Module):
    """
    step-by-step stacked LSTM
    """

    def __init__(self, input_siz, rnn_siz, nb_layers, dropout, device):
        """
        init
        """
        super().__init__()
        self.nb_layers = nb_layers
        self.rnn_siz = rnn_siz
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        self.device = device

        for _ in range(nb_layers):
            self.layers.append(nn.LSTMCell(input_siz, rnn_siz))
            input_siz = rnn_siz

    def get_init_hx(self, batch_size):
        """
        initial h0
        """
        h_0_s, c_0_s = [], []
        for _ in range(self.nb_layers):
            h_0 = torch.zeros((batch_size, self.rnn_siz), device=self.device)
            c_0 = torch.zeros((batch_size, self.rnn_siz), device=self.device)
            h_0_s.append(h_0)
            c_0_s.append(c_0)
        return (h_0_s, c_0_s)

    def forward(self, input, hidden):
        """
        dropout after all output except the last one
        """
        h_0, c_0 = hidden
        h_1, c_1 = [], []
        for i, layer in enumerate(self.layers):
            h_1_i, c_1_i = layer(input, (h_0[i], c_0[i]))
            input = self.dropout(h_1_i)
            h_1 += [h_1_i]
            c_1 += [c_1_i]

        h_1 = torch.stack(h_1)
        c_1 = torch.stack(c_1)

        return input, (h_1, c_1)


class Attention(nn.Module):
    """
    attention with mask
    """

    def forward(self, ht, hs, mask, weighted_ctx=True):
        """
        ht: batch x ht_dim
        hs: (seq_len x batch x hs_dim, seq_len x batch x ht_dim)
        mask: seq_len x batch
        """
        hs, hs_ = hs
        # seq_len, batch, _ = hs.size()
        hs = hs.transpose(0, 1)
        hs_ = hs_.transpose(0, 1)
        # hs: batch x seq_len x hs_dim
        # hs_: batch x seq_len x ht_dim
        # hs_ = self.hs2ht(hs)
        # Alignment/Attention Function
        # batch x ht_dim x 1
        ht = ht.unsqueeze(2)
        # batch x seq_len
        score = torch.bmm(hs_, ht).squeeze(2)
        # attn = F.softmax(score, dim=-1)
        attn = F.softmax(score, dim=-1) * mask.transpose(0, 1) + EPSILON
        attn = attn / attn.sum(-1, keepdim=True)

        # Compute weighted sum of hs by attention.
        # batch x 1 x seq_len
        attn = attn.unsqueeze(1)
        if weighted_ctx:
            # batch x hs_dim
            weight_hs = torch.bmm(attn, hs).squeeze(1)
        else:
            weight_hs = None

        return weight_hs, attn


class SeqtoSeqAttentionLSTM(nn.Module):
    """
    seq2seq with soft attention baseline
    """

    def __init__(
        self,
        src_vocab_size,
        trg_vocab_size,
        embed_dim,
        src_hid_size,
        src_nb_layers,
        trg_hid_size,
        trg_nb_layers,
        dropout_p,
        device
        ):
        """
        init
        """
        super().__init__()
        self.device = device
        self.src_vocab_size = src_vocab_size
        self.trg_vocab_size = trg_vocab_size
        self.embed_dim = embed_dim
        self.src_hid_size = src_hid_size
        self.src_nb_layers = src_nb_layers
        self.trg_hid_size = trg_hid_size
        self.trg_nb_layers = trg_nb_layers
        self.dropout_p = dropout_p
        self.src_embed = nn.Embedding(src_vocab_size, embed_dim, padding_idx=PAD_IDX)
        self.trg_embed = nn.Embedding(trg_vocab_size, embed_dim, padding_idx=PAD_IDX)
        self.enc_rnn = nn.LSTM(embed_dim, src_hid_size, src_nb_layers, bidirectional=True, dropout=dropout_p)
        self.dec_rnn = StackedLSTM(embed_dim, trg_hid_size, trg_nb_layers, dropout_p, self.device)
        self.out_dim = trg_hid_size + src_hid_size * 2
        self.scale_enc_hs = nn.Linear(src_hid_size * 2, trg_hid_size)
        self.attn = Attention()
        self.linear_out = nn.Linear(self.out_dim, self.out_dim)
        self.final_out = nn.Linear(self.out_dim, trg_vocab_size)
        self.dropout = nn.Dropout(dropout_p)
       

    def encode(self, src_batch):
        """
        encoder
        """
        enc_hs, _ = self.enc_rnn(self.dropout(self.src_embed(src_batch)))
        scale_enc_hs = self.scale_enc_hs(enc_hs)
        return enc_hs, scale_enc_hs

    def decode_step(self, enc_hs, enc_mask, input_, hidden):
        """
        decode step
        """
        h_t, hidden = self.dec_rnn(input_, hidden)
        ctx, attn = self.attn(h_t, enc_hs, enc_mask)
        # Concatenate the ht and ctx
        # weight_hs: batch x (hs_dim + ht_dim)
        ctx = torch.cat((ctx, h_t), dim=1)
        # ctx: batch x out_dim
        ctx = self.linear_out(ctx)
        ctx = torch.tanh(ctx)
        word_logprob = self.final_out(ctx)
        return word_logprob, hidden, attn

    def decode(self, enc_hs, enc_mask, trg_batch, teacher_forcing_ratio):
        """
        enc_hs: tuple(enc_hs, scale_enc_hs)
        """        
        _, bs = enc_mask.shape
        trg_seq_len = trg_batch.size(0)
        trg_bat_siz = trg_batch.size(1)
        trg_embed = self.dropout(self.trg_embed(trg_batch))
        outputs = torch.zeros(trg_seq_len, trg_bat_siz, self.trg_vocab_size).to(self.device)
        hidden = self.dec_rnn.get_init_hx(trg_bat_siz)
        # first input to the decoder is the <sos> token.
        input = trg_embed[0, :]
        for idx in range(trg_seq_len):
            output, hidden, _ = self.decode_step(enc_hs, enc_mask, input, hidden)
            outputs[idx] = output
            teacher_force = random.random() < teacher_forcing_ratio
            # update input : use ground_truth when teacher_force 
            input = trg_embed[idx,:] if teacher_force else self.dropout(self.trg_embed(output.argmax(1)))
        return outputs

    def forward(self, src_batch, trg_batch, teacher_forcing_ratio=0.5):
        src_mask = create_mask(src_batch)
        """
        only for training
        """
        # trg_seq_len, batch_size = trg_batch.size()
        enc_hs = self.encode(src_batch)
        # output: [trg_seq_len-1, batch_size, vocab_siz]
        output = self.decode(enc_hs, src_mask, trg_batch, teacher_forcing_ratio)
        return output

    def count_nb_params(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return params