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
package](https://github.com/salaniz/pycocoevalcap). The metrics are
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

## Instructions

To get it running on your machine, some data needs to be downloaded like this.

First, clone the [cocapi
repository](https://github.com/cocodataset/cocoapi) like this
```
git clone https://github.com/cocodataset/cocoapi.git 
```
Then setup the coco API (also described in the readme [here](https://github.com/cocodataset/cocoapi)) 
```
cd cocoapi/PythonAPI
make
cd ..
```

Finally, download the 2014 data from [here](http://cocodataset.org/#download). 
​
* Under **Annotations**, download:
  * **2014 Train/Val annotations [241MB]** (extract captions_train2014.json and captions_val2014.json, and place at locations cocoapi/annotations/captions_train2014.json and cocoapi/annotations/captions_val2014.json, respectively)  
  * **2014 Testing Image info [1MB]** (extract image_info_test2014.json and place at location cocoapi/annotations/image_info_test2014.json)

* Under **Images**, download:
  * **2014 Train images [83K/13GB]** (extract the train2014 folder and place at location cocoapi/images/train2014/)
  * **2014 Val images [41K/6GB]** (extract the val2014 folder and place at location cocoapi/images/val2014/)
  * **2014 Test images [41K/6GB]** (extract the test2014 folder and place at location cocoapi/images/test2014/)

## Acknowledgements
This project was done for the Udacity Computer Vision Nanodegree
program.  Thanks to Udacity for the problem and for the starter code.
Most of my contributions to the code are in `2_Training.ipynb`,
`3_Inference.ipynb`, `model.py` and `val_data_loader.py`.