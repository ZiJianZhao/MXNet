RNN Example
===========
This folder contains RNN examples using low level symbol interface.

There are two forms to construct the unroll lstm network: 
 1. use "data\0", "mask\0", "label\0" to represent the input and output at each step: StepIO
 2. use "data", "mask", "label" to represent the input and output, that is to say, the inputs and outputs of all steps are in just one symbol. We can use some tricks to seperate them for each step in the program: WrapIO

There are no essential differences between these two forms. We just need to modify the data_io file so that the names in the provide_data are same as the names used in the lstm network.
However, There are several differences: 
The first form is more complicated to write but I believe it is intuitive to construct the seq2seq attention model.
And the second form is easier to write but I think it is less intuitive to construct the seq2seq attention model.

Performance Note:
More ```MXNET_GPU_WORKER_NTHREADS``` may lead to better performance. For setting ```MXNET_GPU_WORKER_NTHREADS```, please refer to [Environment Variables](https://mxnet.readthedocs.org/en/latest/env_var.html).
