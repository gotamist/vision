# Image Captioning

Image captioning is an exciting area that brings together Computer
Vision and Natural language processing (or slicing it another way,
brings together CNNs and RNNs).

This is a supervised learning task where we can train on one of the
datasets best suited for this task, the Microsoft Common Objects in
Context dataset ([MS COCO](http://cocodataset.org/#home)) which
contains images and multiple human generated captions describing
them. The dataset can be explored
[here](http://cocodataset.org/#explore), pulling up images containing
various common objects that you can specify from a list.  Here, the
2014 version of the train, test and validation images (and
corresponding annotation files is used).

## The notebooks

There are 4 jupyter notebooks here.

- `0_Dataset` is to get a feel of the dataset.

- `1_Preliminaries` shows the transforms applied to the image a
pre-processing before they are fed into the encoder.  The other
function of the data_loader class is to split up the data into batches
for training.

- `2_Training` is for choosing the hyperparameters , defining the loss
criterion , choosing the optimizer and then running the training and
storing the resulting model after each training epoch.  At the bottom,
I show model validation, for which I have written up a modified class
in `val_data_loader`.  The predicted captions are compared with true
annotations available in the test set.  Finally a variety of metrics
are computed to determine the quality of the translation using the [MS
COCO Caption Evaluation
package](https://github.com/salaniz/pycocoevalcap). The metric are
Bleu_1,..., Bleu_4, METEOR, ROUGE_l and CIDEr.

- `3_Inference` is for inference on the test set, demonstrating the
  action of the methods `sample()` and `beam_sample()` (use of the
  beam search).  Captions are generated on example images.  To show
  that the model does make mistakes, some examples are shown where the
  generated caption is not good.  

## Architecture

An architecture of a CNN to encode images and then a sequence of
RNN, dense layers and softmax to decode into sentences is ideal for
training them. I have added dropout for regularization. Inference and
validation can be done with a greedy algorithm over the outputs of the
CNN or a beam search, which is also implemented. The network is
implemented with PyTorch and is specified in the file `model.py`. 


This project was done for the Udacity Computer Vision Nanodegree
program.  Thanks to Udacity for the problem and for the starter code.
Most of my contributions to the code are in `2_Training.ipynb`,
`3_Inference.ipynb`, `model.py` and `val_data_loader.py`.