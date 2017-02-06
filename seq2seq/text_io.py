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
def get_enc_dec_text_id(path, enc_word2idx, dec_word2idx):
    enc_data = []
    dec_data = []
    index = 0
    white_spaces = re.compile(r'[ \n\r\t]+')
    with codecs.open(path, 'r', encoding='utf-8', errors='ignore') as fid:
        for line in fid:
            line = line.strip()
            line_list = line.split('\t=>\t')
            length = len(line_list)
            for i in xrange(1, length):
                dec_list = list((line_list[0].strip()))
                enc_list = list((line_list[i].strip()))
                enc = [enc_word2idx.get(word) if enc_word2idx.get(word) != None else enc_word2idx.get('<UNK>') for word in enc_list]
                dec = [dec_word2idx.get(word) if dec_word2idx.get(word) != None else  dec_word2idx.get('<UNK>') for word in dec_list]
                enc_data.append(enc)
                dec_data.append(dec)
            index += 1
            if DEBUG:
                if index >= 20:
                    return enc_data, dec_data
    return enc_data, dec_data 

