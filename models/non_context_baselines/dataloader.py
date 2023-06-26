import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class Parser:
    def __init__(self, sentence_start, sentence_end, part_seperator, tag_seperator, unk_token, replace_token="^", parse_all=False, special_unk="?", ignore_start=["<DOC", "</DOC"]):
        self.sentence_start = sentence_start
        self.sentence_end   = sentence_end
        self.part_seperator = part_seperator
        self.tag_seperator  = tag_seperator 
        self.unk_token      = unk_token
        self.replace_token  = replace_token
        self.special_unk    = special_unk
        self.ignore_start   = ignore_start

    def parse_file(self, file):
        data, sentence = [], []
        for line in open(file, "r", encoding="utf-8"):
            if any(line.startswith(i) for i in self.ignore_start):
                continue
            elif line.startswith(self.sentence_start):
                sentence = []
                continue
            elif line.startswith(self.sentence_end):
                data.append(sentence)
                continue
            try:
                splits     = line.split()
                source     = splits[0]
                lemma      = splits[1].split('+')[0]
                lemma_char = [j.lower() for j in lemma]
                tags       = splits[1].split(self.part_seperator)[0].replace(self.replace_token, self.tag_seperator).split(self.tag_seperator)
                tags       = lemma_char + tags[1:]
                tags = [i for i in tags if i!="Prop"]
            except:
                print("I am having some problems parsing the line : ", line)
                
            if not line.startswith(self.sentence_start) and not line.startswith(self.sentence_end):
                sentence.append([[j.lower() for j in source], tags])

        return data
    
class Vocab:
    def __init__(self, data, start_token="<s>", eos_token="</s>", pad_token="<p>", unk_token="<unk>", max_size=64):
        self.start_token  = start_token
        self.eos_token    = eos_token
        self.pad_token    = pad_token
        self.unk_token    = unk_token

        # Create encoder and decoder dictionary
        encoder_default = {pad_token:0, start_token:1, eos_token:2, unk_token:3}
        decoder_default = {0:pad_token, 1:start_token, 2:eos_token, 3:unk_token}
        # Surface dictionaries
        surf_encoder = dict(**encoder_default)
        surf_decoder = {0:pad_token, 1:start_token, 2:eos_token, 3:unk_token}
        # Feature dictionaries
        feat_encoder = dict(**encoder_default) 
        feat_decoder = {0:pad_token, 1:start_token, 2:eos_token, 3:unk_token}

        # Seperate sources and tags in nested list
        sources, targets = [], []
        for sentence in data:
            sentences_source, sentences_tag = [], []
            for word in sentence:
                sentences_source.append(word[0])
                sentences_tag.append(word[1])
            sources.append(sentences_source)
            targets.append(sentences_tag)

        lemmas, tags = [], []
        for sentence in data:
            for word in sentence:
                lemmas.extend(word[0])
                tags.extend(word[1]) 
        
        for j, tag in enumerate(np.sort(list(set(tags)))):
            feat_encoder[tag] = j+4
            feat_decoder[j+4] = tag
            
        for j, surf in enumerate(np.sort(list(set(lemmas)))):
            surf_encoder[surf] = j+4
            surf_decoder[j+4] = surf
        
        self.feat_encoder = feat_encoder
        self.feat_decoder = feat_decoder
        self.surf_encoder = surf_encoder
        self.surf_decoder = surf_decoder
        
        self.max_size = max_size
        self.data = data

    def encode(self, x):
        src = []
        for i in ([self.start_token] + x[0] + [self.eos_token]):
            if i in self.surf_encoder:
                src.append(self.surf_encoder[i])
            else:
                src.append(self.surf_encoder['<unk>'])
        
        tgt = []
        for i in ([self.start_token] + x[1] + [self.eos_token]):
            if i in self.feat_encoder:
                tgt.append(self.feat_encoder[i])
            else:
                tgt.append(self.feat_encoder['<unk>'])
        
        return torch.tensor(src).long(), torch.tensor(tgt).long()
    
    def decode(self, x):
        return [self.surf_decoder[int(i)] for i in x[0]], [self.feat_decoder[int(i)] for i in x[1]]

class WordLoader(Dataset):
    def __init__(self, vocab, data, start_token="<s>", eos_token="</s>", pad_token="<p>", unk_token="<unk>"):  
        self.vocab = vocab #Vocab(data, start_token=start_token, eos_token=eos_token, pad_token=pad_token, unk_token=unk_token)
        
        outs = []
        for sentence in data:
            for word in sentence:
                outs.append(word)
        self.data = outs
        
    def __getitem__(self, idx):
        return self.vocab.encode(self.data[idx])
            
    def __len__(self):
        return len(self.data)