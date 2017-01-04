import re, os, sys, argparse, logging, collections
import codecs

DEBUG = False


def build_char_dict(data_path, dict_path):
    with codecs.open(data_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
        content = set(content)
        content.remove('\n')
    with codecs.open(dict_path, 'w', encoding='utf-8', errors='ignore') as g:
        for ch in content:
            g.write(ch+'\n')

def get_text_char_id(path, word2idx):
    data = []
    label = []
    index = 0
    with open(path, 'r') as fid:
        for line in fid:
            line = line.strip('\n')
            line = list(line)
            if len(line) == 0:
                continue
            tmp = [word2idx.get(word) if word2idx.get(word) != None else word2idx.get('<UNK>') for word in line]
            data.append(tmp[:])
            tmp.append(word2idx.get('<EOS>'))
            label.append(tmp[1:])
            index += 1
            if DEBUG:
                if index >= 20:
                    return data, label
    return data, label

# -------------------------- build dict ---------------------------------------------------- 
def read_char_dict(path):
    word2idx = {'<EOS>' : 0, '<UNK>' : 1, '<PAD>' : 2}
    idx = 3
    with codecs.open(path, 'r', encoding='utf-8', errors='ignore') as fid:
        for line in fid:
            line = line.strip('\n')
            if len(line) == 0:
                continue
            if word2idx.get(line) == None:
                word2idx[line] = idx
            idx += 1
    print word2idx.get(' ')
    return word2idx