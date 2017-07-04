Main file:

train.py: main file, containing train mode and test mode.

=========================================================================

Model files:

seq2seq.py: basic sequence to sequence model file.

global_attention.py: the global attention model file.

focus_attention.py: the focus atention model file, especially for couplet.

=========================================================================

Data processing and iterator files:

enc_dec_iter.py: containing read dict and text2id functions and iterator class.

=========================================================================

Support files:

eval_and_visual.py: containing functions for caluculate bleu score and visual attention weights.

metric.py: Define Preplexity without Exp since the mxnet-self-contained ppl is not accurate.
