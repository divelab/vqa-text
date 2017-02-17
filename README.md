# Learning Convolutional Text Representations for Visual Question Answering

This is the code for our paper submitted to KDD 2017, "Learning Convolutional Text Representations for Visual Question Answering". We used it to explore different text representation methods in VQA. The reference code is [vqa-mcb](https://github.com/akirafukui/vqa-mcb).

## Instructions

To replicate our results, do the following prerequisites as in [vqa-mcb](https://github.com/akirafukui/vqa-mcb):

- Compile the `feature/20160617_cb_softattention` branch of [this fork of Caffe](https://github.com/akirafukui/caffe/). This branch contains Yang Gao’s Compact Bilinear layers ([dedicated repo](https://github.com/gy20073/compact_bilinear_pooling), [paper](https://arxiv.org/abs/1511.06062)) released under the [BDD license](https://github.com/gy20073/compact_bilinear_pooling/blob/master/caffe-20160312/LICENSE_BDD), and Ronghang Hu’s Soft Attention layers ([paper](https://arxiv.org/abs/1511.03745)) released under BSD 2-clause.
- Download the [pre-trained ResNet-152 model](https://github.com/KaimingHe/deep-residual-networks).
- Download the [VQA tools](https://github.com/VT-vision-lab/VQA).
- Download the [VQA real-image dataset](http://visualqa.org/download.html).
- Do the [data preprocessing](https://github.com/akirafukui/vqa-mcb/tree/master/preprocess).

**Note:** As explained in our paper, we did not use any additional data such as "GloVe" and "Visual Genome".

To train and test a model, edit the corresponding `config.py` and `qlstm_solver.prototxt` files.

**Note:** Unlike [vqa-mcb](https://github.com/akirafukui/vqa-mcb), in our experiments, different methods require different data provider layers. Use `vqa_data_provider_layer.py` and `visualize_tools.py` in the same folder.

In `config.py`, set `GPU_ID` and `VALIDATE_INTERVAL` (training iterations) properly.

**Note:** As stated in our paper, we trained only on the training set, and tested on the validation set. The code has been modified to do training and testing automatically if you set `VALIDATE_INTERVAL` to the number of iterations for training. The pre-set number is what we used in our results. In our experiments, we split the original training set into new training set and validation set, and used early stopping to determine this number. Then we used this code to train our model on all training data.

In `qlstm_solver.prototxt`, set `snapshot` and `snapshot_prefix`  correctly.

Now just run `python train_xxx.py`. Training can take some time. Snapshots are saved according to the settings in `qlstm_solver.prototxt`. To stop training, just hit `Control + C`.