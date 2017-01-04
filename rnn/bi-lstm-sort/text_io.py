import re, os, sys, argparse, logging, collections
import codecs

DEBUG = False

# -------------------------- build dict ---------------------------------------------------- 
def read_dict(path):
    word2idx = {'<EOS>' : 0, '<UNK>' : 1, '<PAD>' : 2}
    idx = 3
    with codecs.open(path, 'r', encoding='utf-8', errors='ignore') as fid:
        for line in fid:
            line = line.strip(' ').strip('\n')
            if len(line) == 0:
                continue
            if word2idx.get(line) == None:
                word2idx[line] = idx
            idx += 1
    return word2idx

# ------------------------ transform text into numbers --------------------------
# ------------------------ 1. post and cmnt in one file ----------------------------
def get_data_label_text_id(path, word2idx):
    data = []
    label = []
    index = 0
    white_spaces = re.compile(r'[ \n\r\t]+')
    with codecs.open(path, 'r', encoding='utf-8', errors='ignore') as fid:
        for line in fid:
            line = line.strip()
            line = line.split('\t=>\t')
            assert len(line) == 2
            if len(line[0]) == 0 or len(line[1]) == 0:
                continue
            #enc_list = white_spaces.split(line[0])
            #dec_list = white_spaces.split(line[1])
            data_list = list(line[0])
            label_list = list(line[1])
            line_data = [word2idx.get(word) if word2idx.get(word) != None else word2idx.get('<UNK>') for word in data_list]
            line_label = [word2idx.get(word) if word2idx.get(word) != None else word2idx.get('<UNK>') for word in label_list]
            data.append(line_data)
            label.append(line_label)
            index += 1
            if DEBUG:
                if index >= 20:
                    return data, label
    return data, label

# -------------------------- 2. post and cmnt in different files --------------------

def get_text_id(path, word2idx):
    white_spaces = re.compile(r'[ \n\r\t]+')
    data = []
    index = 0
    with codecs.open(path, 'r', encoding='utf-8', errors='ignore') as fid:
        for line in fid:
            line = line.strip()
            line = white_spaces.split(line)
            if len(line) == 0:
                continue
            tmp = [word2idx.get(word) if word2idx.get(word) != None else word2idx.get('<UNK>') for word in line]
            data.append(tmp)
            index += 1
            if DEBUG:
                if index >= 20:
                    return data
    return data 