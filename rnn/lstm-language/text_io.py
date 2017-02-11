import re, os, sys, argparse, logging, collections
import codecs

def build_dict(data_dir, data_name, dict_name):
    data_path = data_dir + data_name
    dict_path = data_dir + dict_name
    with codecs.open(data_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read().decode("utf-8").replace("\n", "<eos>").split()
        counter = collections.Counter(content)
        count_pairs = sorted(counter.items(), key = lambda x: (-x[1], x[0]))
        words, _ = list(zip(*count_pairs))
    with codecs.open(dict_path, 'w', encoding='utf-8', errors='ignore') as g:
        for ch in words:
            g.write(ch+'\n')

def read_dict(path):
    word2idx = {'<eos>' : 0, '<unk>' : 1, '<pad>' : 2}
    idx = 3
    with codecs.open(path, 'r', encoding='utf-8', errors='ignore') as fid:
        for line in fid:
            line = line.strip('\n')
            if len(line) == 0:
                continue
            if word2idx.get(line) == None:
                word2idx[line] = idx
                idx += 1
    return word2idx

def get_text_id(path, word2idx):
    data = []
    label = []
    with codecs.open(path, 'r', encoding='utf-8', errors='ignore') as fid:
        for line in fid:
            line = line.strip('\n').split()
            if len(line) == 0:
                continue
            tmp = [word2idx.get(word) if word2idx.get(word) != None else word2idx.get('<unk>') for word in line]
            data.append(tmp[:])
            tmp.append(word2idx.get('<eos>'))
            label.append(tmp[1:])
    return data, label
