import re, os, sys, argparse, logging, collections
import codecs
import random

class DataGeneration(object):

    def __init__(self, high, train_num, valid_num, length):
        self.high = high
        self.train_num = train_num
        self.valid_num = valid_num
        self.length = length
        self.train_epoch = int(self.train_num / (self.high / self.length)) + 1
        self.valid_epoch = int(self.valid_num / (self.high / self.length)) + 1

    def generate_q1_pairs(self, data_dir, train_file = 'q1.train', valid_file = 'q1.valid'):
        v = open(os.path.join(data_dir, 'q1.vocab'), 'w')
        for i in range(0, self.high):
            v.write(str(i)+'\n')
        v.close()
        f = open(os.path.join(data_dir, train_file), 'w')
        g = open(os.path.join(data_dir, valid_file), 'w')
        lis = range(0, self.high)
        a = [0] * self.length
        for _ in range(0, self.train_epoch):
            random.shuffle(lis)
            for i in range(self.length, self.high, self.length):
                leng = random.randint(1, self.length)
                a[:] = lis[i - leng:i]
                b = sorted(a)
                f.write(str(a[0]))
                for digit in a[1:]:
                    f.write(' ' + str(digit))
                f.write('\t=>\t')
                f.write(str(b[0]))
                for digit in b[1:]:
                    f.write(' ' + str(digit))
                f.write('\n')
        f.close()
        for _ in range(0, self.valid_epoch):
            random.shuffle(lis)
            for i in range(self.length, self.high, self.length):
                leng = random.randint(1, self.length)
                a[:] = lis[i - leng:i]
                b = sorted(a)
                g.write(str(a[0]))
                for digit in a[1:]:
                    g.write(' ' + str(digit))
                g.write('\t=>\t')
                g.write(str(b[0]))
                for digit in b[1:]:
                    g.write(' ' + str(digit))
                g.write('\n')
        g.close()

    def generate_q2_pairs(self, data_dir, train_file = 'q2.train', valid_file = 'q2.valid'):
        def cmp(a, b):
            a_sum = sum(map(int, str(a)))
            b_sum = sum(map(int, str(b)))
            return a_sum - b_sum
        v = open(os.path.join(data_dir, 'q2.vocab'), 'w')
        for i in range(0, self.high):
            v.write(str(i)+'\n')
        v.close()
        f = open(os.path.join(data_dir, train_file), 'w')
        g = open(os.path.join(data_dir, valid_file), 'w')
        lis = range(0, self.high)
        a = [0] * self.length
        for _ in range(0, self.train_epoch):
            random.shuffle(lis)
            for i in range(self.length, self.high, self.length):
                a[:] = lis[i - self.length:i]
                b = sorted(a, cmp)
                f.write(str(a[0]))
                for digit in a[1:]:
                    f.write(' ' + str(digit))
                f.write('\t=>\t')
                f.write(str(b[0]))
                for digit in b[1:]:
                    f.write(' ' + str(digit))
                f.write('\n')
        f.close()
        for _ in range(0, self.valid_epoch):
            random.shuffle(lis)
            for i in range(self.length, self.high, self.length):
                a[:] = lis[i - self.length:i]
                b = sorted(a, cmp)
                g.write(str(a[0]))
                for digit in a[1:]:
                    g.write(' ' + str(digit))
                g.write('\t=>\t')
                g.write(str(b[0]))
                for digit in b[1:]:
                    g.write(' ' + str(digit))
                g.write('\n')
        g.close()

    # No srting, just digits sum transform 
    def generate_q3_pairs(self, data_dir, train_file = 'q3.train', valid_file = 'q3.valid'):
        self.sum_lists = []
        maxm = 0
        for i in range(0, self.high):
            if sum(map(int, str(i))) > maxm:
                maxm = sum(map(int, str(i)))
        for i in range(0, maxm+1):
            self.sum_lists.append([])
        for i in range(0, self.high):
            self.sum_lists[sum(map(int, str(i)))].append(i)
        v = open(os.path.join(data_dir, 'q3.vocab'), 'w')
        for i in range(0, self.high):
            v.write(str(i)+'\n')
        v.close()
        f = open(os.path.join(data_dir, train_file), 'w')
        g = open(os.path.join(data_dir, valid_file), 'w')
        lis = range(0, self.high)
        a = [0] * self.length
        for _ in range(0, self.train_epoch):
            random.shuffle(lis)
            for i in range(self.length, self.high, self.length):
                a[:] = lis[i - self.length:i]
                f.write(str(a[0]))
                for digit in a[1:]:
                    f.write(' ' + str(digit))
                f.write('\t=>\t')
                f.write(str(random.choice(self.sum_lists[sum(map(int, str(a[0])))])))
                for digit in a[1:]:
                    tmp = random.choice(self.sum_lists[sum(map(int, str(digit)))])
                    f.write(' ' + str(tmp))
                f.write('\n')
        f.close()
        for _ in range(0, self.valid_epoch):
            random.shuffle(lis)
            for i in range(self.length, self.high, self.length):
                a[:] = lis[i - self.length:i]
                g.write(str(a[0]))
                for digit in a[1:]:
                    g.write(' ' + str(digit))
                g.write('\t=>\t')
                g.write(str(random.choice(self.sum_lists[sum(map(int, str(a[0])))])))
                for digit in a[1:]:
                    tmp = random.choice(self.sum_lists[sum(map(int, str(digit)))])
                    g.write(' ' + str(tmp))
                g.write('\n')
        g.close()

if __name__ == '__main__':
    data_gen = DataGeneration(100, 50000, 1000, 10)
    data_gen.generate_q1_pairs('/slfs1/users/zjz17/github/data/sort/')

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
                enc_list = line_list[0].strip().split()
                dec_list = line_list[i].strip().split()
                enc = [enc_word2idx.get(word) if enc_word2idx.get(word) != None else enc_word2idx.get('<UNK>') for word in enc_list]
                dec = [dec_word2idx.get(word) if dec_word2idx.get(word) != None else  dec_word2idx.get('<UNK>') for word in dec_list]
                enc_data.append(enc)
                dec_data.append(dec)
            index += 1
            if DEBUG:
                if index >= 20:
                    return enc_data, dec_data
    return enc_data, dec_data

