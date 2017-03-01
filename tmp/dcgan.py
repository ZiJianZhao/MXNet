import mxnet as mx
import numpy as np 
from sklearn.datasets import fetch_mldata
import logging
import cv2
from datetime import datetime

def make_dcgan_sym(ngf, ndf, nc, no_bias=True, fix_gamma=True, eps=1e-5+1e-12):
    BatchNorm = mx.sym.BatchNorm
    rand = mx.sym.Variable('rand')

    g1 = mx.sym.Deconvolution(rand, name='g1', kernel=(4,4), num_filter=ngf*8, no_bias=no_bias)
    gbn1 = BatchNorm(g1, name='gbn1', fix_gamma=fix_gamma, eps=eps)
    gact1 = mx.sym.Activation(gbn1, name='gact1', act_type='relu')


    g2 = mx.sym.Deconvolution(gact1, name='g2', kernel=(4,4), stride=(2,2), pad=(1,1), num_filter=ngf*4, no_bias=no_bias)
    gbn2 = BatchNorm(g2, name='gbn2', fix_gamma=fix_gamma, eps=eps)
    gact2 = mx.sym.Activation(gbn2, name='gact2', act_type='relu')

    g3 = mx.sym.Deconvolution(gact2, name='g3', kernel=(4,4), stride=(2,2), pad=(1,1), num_filter=ngf*2, no_bias=no_bias)
    gbn3 = BatchNorm(g3, name='gbn3', fix_gamma=fix_gamma, eps=eps)
    gact3 = mx.sym.Activation(gbn3, name='gact3', act_type='relu')

    g4 = mx.sym.Deconvolution(gact3, name='g5', kernel=(4,4), stride=(2,2), pad=(1,1), num_filter=ngf, no_bias=no_bias)
    gbn4 = BatchNorm(g4, name='gbn4', fix_gamma=fix_gamma, eps=eps)
    gact4 = mx.sym.Activation(gbn4, name='gact4', act_type='relu')

    g5 = mx.sym.Deconvolution(gact4, name='g5', kernel=(4,4), stride=(2,2), pad=(1,1), num_filter=nc, no_bias=no_bias)
    gout = mx.sym.Activation(g5, name='gout', act_type='tanh')

    data = mx.sym.Variable('data')
    label = mx.sym.Variable('label')

    d1 = mx.sym.Convolution(data, name='d1', kernel=(4,4), stride=(2,2), pad=(1,1), num_filter=ndf, no_bias=no_bias)
    act1 = mx.sym.LeakyReLU(d1, name='dact1', act_type='leaky', slope=0.2)

    d2 =mx.sym.Convolution(dact1, name='d2', kernel=(4,4), stride=(2,2), pad=(1,1), num_filter=ndf*2, no_bias=no_bias)
    dbn2 = BatchNorm(d2, name='dbn2', fix_gamma=fix_gamma, eps=eps)
    dact2 = mx.sym.LeakyReLU(dbn2, name='dact2', act_type='leaky', slope=0.2)

    d3 = mx.sym.Convolution(dact2, name='d3', kernel=(4,4), stride=(2,2), pad=(1,1), num_filter=ndf*4, no_bias=no_bias)
    dbn3 = BatchNorm(d3, name='dbn3', fix_gamma=fix_gamma, eps=eps)
    dact3 = mx.sym.LeakyReLU(dbn3, name='dact3', act_type='leaky', slope=0.2)

    d4 = mx.sym.Convolution(dact3, name='d4', kernel=(4,4), stride=(2,2), pad=(1,1), num_filter=ndf*8, no_bias=no_bias)
    dbn4 = BatchNorm(d4, name='dbn4', fix_gamma=fix_gamma, eps=eps)
    dact4 = mx.sym.LeakyReLU(dbn4, name='dact4', act_type='leaky', slope=0.2)

    d5 = mx.sym.Convolution(dact4, name='d5', kernel=(4,4), num_filter=1, no_bias=no_bias)
    d5 = mx.sym.Flatten(d5)

    dloss = mx.sym.LogisticRegressionOutput(data=d5, label=label, name='dloss')
    return gout, dloss

def get_mnist():
    mnist = fetch_mldata('MNIST original')
    np.random.seed(1234)
    p = np.random.permutation(mnist.data.shape[0])
    X = mnist.data[p]
    X = X.reshape((70000, 28, 28))

    X = np.asarray([cv2.resize(x, (64, 64)) for x in X])

    X = X.astype(np.float32)/(255.0/2) - 1.0 
    X = X.reshape((70000, 1, 64, 64))
    X = np.tile(X, (1,3,1,1))
    X_train = X[:60000]
    X_test = X[60000:]

    return X_train, X_test

class RandIter(mx.io.DataIter):
    def __init__(self, batch_size, ndim):
        self.batch_size = batch_size
        self.ndim = ndim
        self.provide_data = [('rand', (batch_size, ndim, 1, 1))]
        self.provide_label = []

    def iter_next(self):
        return True

    def getdata(self):
        return [mx.random.normal(0, 1.0, shape=(self.batch_size, self.ndim, 1, 1))]

class ImagenetIter(mx.io.DataIter):
    def __init__(self, path, batch_size, data_shape):
        self.internal = mx.io.ImageRecordIter(
            path_imgrec = path,
            data_shape = data_shape,
            batch_size = batch_size,
            rand_crop = True,
            rand_mirror = True,
            max_crop_size = 256,
            min_crop_size = 192)
        self.provide_data = [('data', (batch_size,)+data_shape)]
        self.provide_label = []

    def reset(self):
        self.internal.reset()

    def iter_next(self):
        return self.internal.iter_next()

    def getdata(self):
        data = self.internal.getdata()
        data = data * (2.0/255.0)
        data -= 1
        return [data]

def fill_buf(buf, i, img, shape):
    n = buf.shape[0]/shape[1]
    m = buf.shape[1]/shape[0]
    sx = (i%m) * shape[0]
    sy = (i/m) * shape[1]
    buf[sy:sy+shape[1], sx:sx+shape[0], :] = img

def visual(title, X):
    assert len(X.shape) == 4
    X = X.transpose((0, 2, 3, 1))
    X = np.clip((X+1.0)*(255.0/2.0), 0, 255).astype(np.uint8)
    n = np.ceil(np.sqrt(X.shape[0]))
    buff = np.zeros((n*X.shape[1], n*X.shape[2], X.shape[3]), dtype=np.uint8)
    for i, img in enumerate(X):
        fill_buf(buff, i, img, X.shape[1:3])
    buff = cv2.cvtColor(buff, cv2.COLOR_BGR2RGB)
    cv2.imshow(title, buff)
    cv2.waitKey(1)

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    #==========================setting====================
    dataset = 'mnist'
    imagnet_path = './train.rec'
    ndf = 64
    ngf = 64
    nc = 3
    batch_size = 64
    Z = 100
    lr = 0.0002
    beta1 = 0.5
    ctx = mx.gpu(0)
    check_point = False
    symG, symD = make_dcgan_sym(ngf, ndf, nc)
    