import mxnet as mx 

class InitializerWithWord2vec(mx.init.Initializer):
    def __init__(self, enc_word2idx, dec_word2idx, scale):
        super(Customized, self).__init__()
        self.scale = scale
        self.enc_word2idx = enc_word2idx
        self.dec_word2idx = dec_word2idx
        self.word2vec = self.load_word2vec()
        self.enc_embed = self.get_embed_with_word2vec(self.enc_word2idx, self.word2vec)
        self.dec_embed = self.get_embed_with_word2vec(self.dec_word2idx, self.word2vec)

    def load_word2vec(self, file = 'stc_corpus.vector'):
        mydic = {}
        with codecs.open(file, encoding = 'utf-8') as f:
            f.readline()
            while True:
                str = f.readline()
                if str == '':
                    break;
                else:
                    lis = str.strip().split(' ')
                    tmp = [float(word) for word in lis[1:]]
                    mydic[lis[0]] = np.array(tmp)
        return mydic

    def get_embed_with_word2vec(self, word2idx, word2vec):
        embed_weight = []
        lis = sorted(word2idx.iteritems(), key=lambda d:d[1], reverse = False)
        total = len(word2idx)
        num = 0
        np.random.seed(1)
        for word, idx in lis:
            if word in word2vec:
                embed_weight.append(word2vec[word])
                num += 1
            else:
                vector = np.random.uniform(-0.05, 0.05, 400)
                embed_weight.append(vector)
        logging.info('total: %d, idx: %d' % (total, num))
        embed_weight = np.array(embed_weight)
        return embed_weight

    def _init_weight(self, name, arr):
        if name.endswith('enc_embed_weight'):
            enc_embed_weight = mx.nd.array(self.enc_embed)
            logging.info('enc_embed_weight with word2vec')
            enc_embed_weight.copyto(arr)
        elif name.endswith('dec_embed_weight'):
            dec_embed_weight = mx.nd.array(self.dec_embed)
            logging.info('enc_embed_weight with word2vec')
            dec_embed_weight.copyto(arr)
        else:
            mx.random.uniform(-self.scale, self.scale, out=arr)

class InitializerWithPretrainedParameter(mx.init.Initializer):
    def __init__(self, params_dir = 'optimal_params', params_prefix = 'couplet', epoch = 20, scale = 0.05):
        super(CustomizedInitializer, self).__init__()
        self.scale = scale

    def _init_weight(self, name, arr):
        if name.endswith('embed_weight'):
            embed_weight.copyto(arr)
        else:
            mx.random.uniform(-self.scale, self.scale, out=arr)