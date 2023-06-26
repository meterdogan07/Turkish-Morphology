import numpy as np

class Parser:
    def __init__(self, sentence_start, sentence_end, part_seperator, tag_seperator, unk_token, replace_token="^", parse_all=False, special_unk="?", ignore_start=["<DOC", "</DOC","<DATA>","<CORPUS","</DATA>","</CORPUS"]):
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
                tags       = splits[1].split(self.part_seperator)[0].replace(self.replace_token, self.tag_seperator+self.replace_token).split(self.tag_seperator)
                tags       = lemma_char + tags[1:]
                if(source == "+"):
                    tags = ["+","Punct"]
            except:
                print("I am having some problems parsing the line : ", line)
                
            if not line.startswith(self.sentence_start) and not line.startswith(self.sentence_end):
                sentence.append([[j.lower() for j in source], tags])

        return data