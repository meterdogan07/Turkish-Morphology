import numpy as np
import math
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from utils import *

dtype = torch.float32 # we will be using float throughout this tutorial

EPSILON = 1e-7
PAD_IDX = 0
BOS_IDX = 1

class SinusoidalPositionalEmbedding(nn.Module):
    """This module produces sinusoidal positional embeddings of any length.
    Padding symbols are ignored.
    """

    def __init__(self, embedding_dim, padding_idx, init_size=1024):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.weights = SinusoidalPositionalEmbedding.get_embedding(init_size, embedding_dim, padding_idx)
        self.register_buffer("_float_tensor", torch.FloatTensor(1))

    @staticmethod
    def get_embedding(num_embeddings, embedding_dim, padding_idx=None):
        """Build sinusoidal embeddings.
        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        """
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        if embedding_dim % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        return emb

    def forward(self, input):
        """Input is expected to be of size [bsz x seqlen]."""
        #print("Pos Embedding: ", input.shape)
        bsz, seq_len = input.shape
        max_pos = self.padding_idx + 1 + seq_len
        if self.weights is None or max_pos > self.weights.size(0):
            # recompute/expand embeddings if needed
            self.weights = SinusoidalPositionalEmbedding.get_embedding(max_pos, self.embedding_dim, self.padding_idx)
        self.weights = self.weights.to(self._float_tensor)

        mask = input.ne(self.padding_idx).long()
        positions = torch.cumsum(mask, dim=0) * mask + self.padding_idx
        return (self.weights.index_select(0, positions.view(-1)).view(bsz, seq_len, -1).detach())


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        attention_dropout=0.1,
        activation_dropout=0.1,
        activation="relu",
        normalize_before=True,
    ):
        super(TransformerEncoderLayer, self).__init__()
        self.normalize_before = normalize_before
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=attention_dropout
        )
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.activation_dropout = nn.Dropout(activation_dropout)

        self.activation = {"relu": F.relu, "gelu": F.gelu}[activation]

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        r"""Pass the input through the encoder layer.
        Args:
            src: the sequnce to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
        """
        # Self attention block
        residual = src
        if self.normalize_before:
            src = self.norm1(src)
        src = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = residual + self.dropout(src)
        if not self.normalize_before:
            src = self.norm1(src)
        # Feed forward block
        residual = src
        if self.normalize_before:
            src = self.norm2(src)
        src = self.activation(self.linear1(src))
        src = self.activation_dropout(src)
        src = self.linear2(src)
        src = residual + self.dropout(src)
        if not self.normalize_before:
            src = self.norm2(src)
        return src


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        attention_dropout=0.1,
        activation_dropout=0.1,
        activation="relu",
        normalize_before=True,
    ):
        super(TransformerDecoderLayer, self).__init__()
        self.normalize_before = normalize_before
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=attention_dropout
        )
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.activation_dropout = nn.Dropout(activation_dropout)

        self.activation = {"relu": F.relu, "gelu": F.gelu}[activation]

    def forward(
        self,
        tgt,
        memory,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
    ):
        r"""Pass the inputs (and mask) through the decoder layer.

        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequnce from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).
        """
        # self attention block
        residual = tgt
        if self.normalize_before:
            tgt = self.norm1(tgt)
        tgt = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        tgt = residual + self.dropout(tgt)
        if not self.normalize_before:
            tgt = self.norm1(tgt)
        # cross attention block
        residual = tgt
        if self.normalize_before:
            tgt = self.norm2(tgt)
        tgt = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)[0]
        tgt = residual + self.dropout(tgt)
        if not self.normalize_before:
            tgt = self.norm2(tgt)
        # feed forward block
        residual = tgt
        if self.normalize_before:
            tgt = self.norm3(tgt)
        tgt = self.activation(self.linear1(tgt))
        tgt = self.activation_dropout(tgt)
        tgt = self.linear2(tgt)
        tgt = residual + self.dropout(tgt)
        if not self.normalize_before:
            tgt = self.norm3(tgt)
        return tgt


class Transformer(nn.Module):
    def __init__(self,
            src_vocab_size,
            trg_vocab_size,
            embed_dim,
            nb_heads,
            src_hid_size,
            src_nb_layers,
            trg_hid_size,
            trg_nb_layers,
            dropout_p,
            device):
        super().__init__()

        self.device = device
        self.src_vocab_size = src_vocab_size
        self.trg_vocab_size = trg_vocab_size
        self.embed_dim = embed_dim
        self.embed_scale = math.sqrt(embed_dim)
        self.nb_heads = nb_heads
        self.src_hid_size = src_hid_size
        self.src_nb_layers = src_nb_layers
        self.trg_hid_size = trg_hid_size
        self.trg_nb_layers = trg_nb_layers
        self.dropout_p = dropout_p
        self.tie_trg_embed = False
        self.src_embed = Embedding(src_vocab_size, embed_dim, padding_idx=PAD_IDX)
        self.trg_embed = Embedding(trg_vocab_size, embed_dim, padding_idx=PAD_IDX)
        self.position_embed = SinusoidalPositionalEmbedding(embed_dim, PAD_IDX)

        encoder_layer = TransformerEncoderLayer(d_model=embed_dim, nhead=nb_heads, dim_feedforward=src_hid_size,
            dropout=dropout_p, attention_dropout=dropout_p, activation_dropout=dropout_p, normalize_before=False)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=src_nb_layers, norm=nn.LayerNorm(embed_dim))

        decoder_layer = TransformerDecoderLayer(d_model=embed_dim, nhead=nb_heads, dim_feedforward=trg_hid_size,
            dropout=dropout_p, attention_dropout=dropout_p, activation_dropout=dropout_p, normalize_before=False)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=trg_nb_layers, norm=nn.LayerNorm(embed_dim))

        self.final_out = Linear(embed_dim, trg_vocab_size)
        if self.tie_trg_embed:
            self.final_out.weight = self.trg_embed.weight
        self.dropout = nn.Dropout(dropout_p)
        # self._reset_parameters()

    def embed(self, src_batch, src_mask):
        word_embed = self.embed_scale * self.src_embed(src_batch)
        pos_embed = self.position_embed(src_batch)
        embed = self.dropout(word_embed + pos_embed)
        return embed

    def encode(self, src_batch, src_mask):
        embed = self.embed(src_batch, src_mask)
        return self.encoder(embed, src_key_padding_mask=src_mask)

    def decode(self, enc_hs, src_mask, trg_batch, trg_mask):
        word_embed = self.embed_scale * self.trg_embed(trg_batch)
        pos_embed = self.position_embed(trg_batch)
        embed = self.dropout(word_embed + pos_embed)

        trg_seq_len = trg_batch.size(0)
        causal_mask = self.generate_square_subsequent_mask(trg_seq_len)
        dec_hs = self.decoder(embed, enc_hs, tgt_mask=causal_mask, tgt_key_padding_mask=trg_mask, memory_key_padding_mask=src_mask)
        return F.log_softmax(self.final_out(dec_hs), dim=-1) #self.final_out(dec_hs)
        
    def decode_greedy(self, enc_hs, src_batch, src_mask, trg_maskx):
        _, bs = src_batch.shape
        trg_seq_len = trg_maskx.size(1)
        output = torch.tensor([BOS_IDX] * bs, device=self.device)
        output = output.view(1, bs)
        outputs = torch.zeros(trg_seq_len, bs, self.trg_vocab_size).to(self.device)
        for i in range(trg_seq_len):
            trg_mask = dummy_mask(output)
            trg_mask = (trg_mask == 0).transpose(0, 1)
            word_logprob = self.decode(enc_hs, src_mask, output, trg_mask)
            word_logprob = word_logprob[-1]
            outputs[i] = word_logprob
            word = word_logprob.argmax(-1)
            output = torch.cat((output, word.view(1, bs)))
        return output[1:]

    def forward(self, src_batch, trg_batch, teacher_forcing_ratio=1):
        """
        only for training
        """
        src_mask = create_mask(src_batch)
        trg_mask = create_mask(trg_batch)

        src_mask = (src_mask == 0).transpose(0, 1)
        trg_mask = (trg_mask == 0).transpose(0, 1)
        # trg_seq_len, batch_size = trg_batch.size()
        enc_hs = self.encode(src_batch, src_mask)
        # output: [trg_seq_len, batch_size, vocab_siz]
        if(teacher_forcing_ratio==1):
            output = self.decode(enc_hs, src_mask, trg_batch, trg_mask)
        else:
            output = self.decode_greedy(enc_hs, src_batch, src_mask, trg_mask)
        return output
    
    def loss(self, predict, target, reduction=True):
        predict = predict.view(-1, self.trg_vocab_size)
        label_smooth = 0.1
        # nll_loss = F.nll_loss(predict.view(-1, self.trg_vocab_size), target.view(-1), ignore_index=PAD_IDX)
        target = target.view(-1, 1)
        non_pad_mask = target.ne(PAD_IDX)
        nll_loss = -predict.gather(dim=-1, index=target)[non_pad_mask].mean()
        smooth_loss = -predict.sum(dim=-1, keepdim=True)[non_pad_mask].mean()
        smooth_loss = smooth_loss / self.trg_vocab_size
        loss = (1.0 - label_smooth) * nll_loss + label_smooth * smooth_loss
        return loss

    def get_loss(self, src, trg, reduction=True):
        out = self.forward(src, trg, 1)
        loss = self.loss(out[:-1], trg[1:], reduction=reduction)
        return loss

    def count_nb_params(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return params
    
    def generate_square_subsequent_mask(self, sz):
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = (mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, float(0.0)))
        return mask.to(self.device)
    
def dummy_mask(seq):
    """
    create dummy mask (all 1)
    """
    if isinstance(seq, tuple):
        seq = seq[0]
    return torch.ones_like(seq, dtype=torch.float)

def Embedding(num_embeddings, embedding_dim, padding_idx=None):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim**-0.5)
    if padding_idx is not None:
        nn.init.constant_(m.weight[padding_idx], 0)
    return m

def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m