import mxnet as mx 
import numpy as np 

class PerplexityWithoutExp(mx.metric.EvalMetric):
    """Calculate perplexity

    Parameters
    ----------
    ignore_label : int or None
        index of invalid label to ignore when
        counting. usually should be -1. Include
        all entries if None.
    """
    def __init__(self, ignore_label):
        super(PerplexityWithoutExp, self).__init__('PerplexityWithoutExp')
        self.ignore_label = ignore_label

    def update(self, labels, preds):
        assert len(labels) == len(preds)
        loss = 0.
        num = 0
        probs = []
        for label, pred in zip(labels, preds):
            assert label.size == pred.size/pred.shape[-1], \
                "shape mismatch: %s vs. %s"%(label.shape, pred.shape)
            label = label.as_in_context(pred.context).astype(dtype='int32').reshape((label.size,))
            pred = mx.ndarray.batch_take(pred, label)
            probs.append(pred)

        for label, prob in zip(labels, probs):
            prob = prob.asnumpy()
            if self.ignore_label is not None:
                ignore = label.asnumpy().flatten() == self.ignore_label
                prob = prob*(1-ignore) + ignore
                num += prob.size - ignore.sum()
            else:
                num += prob.size
            loss += -np.log(np.maximum(1e-10, prob)).sum()
        #self.sum_metric += numpy.exp(loss / num)
        #self.num_inst += 1
        self.sum_metric += loss 
        self.num_inst += num