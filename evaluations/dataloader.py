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
                tags = [i for i in tags if (i!="Prop" and i!= "")]
                if(source == "+"):
                    tags = ["+","Punct"]
            except:
                print("I am having some problems parsing the line : ", line)
                
            if not line.startswith(self.sentence_start) and not line.startswith(self.sentence_end):
                sentence.append([[j.lower() for j in source], tags])

        return data
    
class Vocab:
    def __init__(self, data, pad_feat_to=-1, pad_surf_to=-1, start_token="<s>", eos_token="</s>", pad_token="<p>",padding_mode="right", unk_token="<unk>", space_token=" ", max_size=64):
        self.pad_feat_to  = pad_feat_to
        self.pad_surf_to  = pad_surf_to
        self.padding_mode = padding_mode
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
        
        maxlen1 = 0
        maxlen2 = 0
        for ww in x:
            if(maxlen1 < len(ww[0])):
                maxlen1 = len(ww[0])
            if(maxlen2 < len(ww[1])):
                maxlen2 = len(ww[1])
        src = np.zeros((len(x), maxlen1+2))
        for ix, word in enumerate(x):
            #print(sentence)
            words = []
            for i in self.handle_input_word(word[0], maxlen1+2, input_type="surf"):
                if i in self.surf_encoder:
                    words.append(self.surf_encoder[i])
                else:
                    words.append(self.surf_encoder['<unk>'])
            src[ix,:] = words
       
        source = self.handle_input_sentence(src, input_type="surf")
        
        tgt = np.zeros((len(x), maxlen2+2))
        for ix, word in enumerate(x):
            words = []
            for i in self.handle_input_word(word[1], maxlen2+2, input_type="feat"):
                if i in self.feat_encoder:
                    words.append(self.feat_encoder[i])
                else:
                    words.append(self.feat_encoder['<unk>'])
            tgt[ix,:] = words
        target = self.handle_input_sentence(tgt, input_type="feat")
        
        return source, target
    
    def decode(self, x):
        return [self.surf_decoder[int(i)] for i in x[0]], [self.feat_decoder[int(i)] for i in x[1]]

    def handle_input_word(self, x, maxlen, input_type="surf"):

        padding = maxlen

        if len(x) > self.max_size:
            return x[:self.max_size]
        
        elif padding == -1:
            return x
        
        if self.padding_mode == "symmetric":
            diff = padding - len(x)
            left_padding = (diff - 2) // 2 
            right_padding = (diff - 2) - left_padding
        
        elif self.padding_mode == "left":
            left_padding = padding - len(x) - 2 
            right_padding = 0
        
        elif self.padding_mode == "right":
            left_padding = 0
            right_padding = padding - len(x) - 2
            
        return [self.pad_token] * left_padding + [self.start_token] + x + [self.eos_token] + [self.pad_token] * right_padding 
    
    def handle_input_sentence(self, x, input_type="surf"):

        padding = 0

        if len(x) > self.max_size:
            x = x[:self.max_size]
        
        elif padding == -1:
            return x
        
        x = torch.tensor(x).long()

        return x
    
class WordLoader(Dataset):
    def __init__(self, vocab, data, pad_feat_to=-1, pad_surf_to=-1, start_token="<s>", eos_token="</s>", pad_token="<p>", unk_token="<unk>"):  
        self.vocab = vocab #Vocab(data, start_token=start_token, eos_token=eos_token, pad_token=pad_token, unk_token=unk_token)
        
        outs = []
        for sentence in data:
            words = []
            for word in sentence:
                words.append(word)
            outs.append(words)
        self.data = outs
        
    def __getitem__(self, idx):
        return self.vocab.encode(self.data[idx])
            
    def __len__(self):
        return len(self.data)