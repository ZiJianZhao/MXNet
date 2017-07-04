#-*- coding:utf-8 -*-

import re, os, sys, argparse, logging, collections
import codecs
from collections import namedtuple, defaultdict
from nltk.translate.bleu_score import corpus_bleu
import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from pylab import * 
import numpy as np

def read_file(file):
    with codecs.open(file, 'r', encoding='utf-8', errors='ignore') as fid:
        lines = fid.readlines()[:3000]
    index = 0
    list_of_references = []
    list_of_hypothesis = []
    while index < len(lines):
        if lines[index].strip().startswith('cmnt'):
            cmnt_list = lines[index][5:].strip().split('\t')
            references = [s.split() for s in cmnt_list if len(s.strip()) > 0]
        if lines[index].strip().startswith('score'):
            hypothesis = lines[index].strip().split('sentence:')[1].strip().split()
            list_of_hypothesis.append(hypothesis)
            list_of_references.append(references)
        index += 1
    return list_of_hypothesis, list_of_references

def draw_confusion_matrix(conf_arr, x_annotations, y_annotations):
    # reference: http://blog.csdn.net/epsil/article/details/9171527
    # x_annotations: source sentence, y_annotations: target_sentence

    # for chinese display
    font = FontProperties(fname=r"/slfs1/users/zjz17/tools/fonts/simhei.ttf", size=14)
    
    norm_conf = conf_arr / np.sum(conf_arr, axis=1).reshape(-1,1)
    fig = plt.figure()  
    plt.clf()
    ax = fig.add_subplot(111)  
    ax.set_aspect(1)  
    res = ax.imshow(norm_conf, cmap=plt.cm.jet,  
                    interpolation='nearest')  
    y_length = conf_arr.shape[0]
    x_length = conf_arr.shape[1]
    cb = fig.colorbar(res)
    plt.xticks(fontsize=7)  
    plt.yticks(fontsize=7)  
    locs, labels = plt.xticks(range(x_length), x_annotations, fontproperties=font)  
    #for t in labels:  
         #t.set_rotation(90)  
    plt.yticks(range(y_length), y_annotations, fontproperties=font)
    plt.title('Epoch 5 Attention Distribution')  
    plt.savefig('confusion_matrix.png', format='png')      

def test():
    # judge the availabe chinese fonts
    from matplotlib.font_manager import FontManager
    import subprocess

    fm = FontManager()
    mat_fonts = set(f.name for f in fm.ttflist)

    output = subprocess.check_output(
        'fc-list :lang=zh -f "%{family}\n"', shell=True)
    # print '*' * 10, '系统可用的中文字体', '*' * 10
    print output
    zh_fonts = set(f.split(',', 1)[0] for f in output.split('\n'))
    available = mat_fonts & zh_fonts

    print '*' * 10, '可用的字体', '*' * 10
    for f in available:
        print f

if __name__ == '__main__':
    conf_arr = np.zeros((3,3))
    conf_arr[0,0] = 1
    conf_arr[1,1] = 1
    conf_arr[2,2] = 1
    x_annotations = [u'我', u'是', u'谁']
    y_annotations = [u'要', u'你', u'管']
    draw_confusion_matrix(conf_arr, x_annotations, y_annotations)