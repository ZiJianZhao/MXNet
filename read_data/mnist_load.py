# reference: http://www.jianshu.com/p/84f72791806f

import numpy as np
import struct
import collections

train_images_idx3_ubyte_file = '../data/mnist/train-images-idx3-ubyte'
train_labels_idx1_ubyte_file = '../data/mnist/train-labels-idx1-ubyte'
test_images_idx3_ubyte_file = '../data/mnist/t10k-images-idx3-ubyte'
test_labels_idx1_ubyte_file = '../data/mnist/t10k-labels-idx1-ubyte'

Images = collections.namedtuple('images', ['data', 'label'])

def decode_idx3_ubyte(file):
    print "load file ", file
    bin_data = open(file).read()
    magic_number, num_images, num_rows, num_cols = \
    struct.unpack_from('>iiii', bin_data, 0)
    imgs = struct.unpack_from('>'+str(num_images*num_rows*num_cols)+'B', \
        bin_data, struct.calcsize('>iiii'))
    imgs = np.reshape(imgs, [num_images, num_rows, num_cols])
    print "load finished, image shape ", imgs.shape
    return imgs

def decode_idx1_ubyte(file):
    print "load file ", file
    bin_data = open(file).read()
    magic_number, num_labels = struct.unpack_from('>ii', bin_data, 0)
    labels = struct.unpack_from('>'+str(num_labels)+'B', bin_data, struct.calcsize('>ii'))
    labels = np.reshape(labels, [num_labels, ])
    print "load finished, label shape ", labels.shape
    return labels

def load_data():
    tr_images = decode_idx3_ubyte(train_images_idx3_ubyte_file)
    tr_labels = decode_idx1_ubyte(train_labels_idx1_ubyte_file)
    te_images = decode_idx3_ubyte(test_images_idx3_ubyte_file)
    te_labels = decode_idx1_ubyte(test_labels_idx1_ubyte_file)
    train_num = 50000
    train_images = Images(
        data = tr_images[0:train_num],
        label = tr_labels[0:train_num])
    valid_images = Images(
        data = tr_images[train_num:],
        label = tr_labels[train_num:])
    test_images = Images(
        data = te_images,
        label = te_labels)
    return train_images, valid_images, test_images
