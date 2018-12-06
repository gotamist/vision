# Semi-supervised learning using GANs

This notebook shows how a GAN can be used to do semi-supervised
training, where a large dataset is available for training, but only a
small amount of it is labeled.  The problem here is to classify
handwritten digits as one of 10 digits.  

This is done by making the discriminator train to classify
two problems - to tell real images from those generated by the
discriminator, and if the image is identified as real, to classify it
as one of ten digits.

Here, the discriminator is what is retained as valuable (the generator
can be tossed after training), in contrast to the more common use of
GANs as generative models, where the valuable end-product of the
training is the generator (and the discriminator can be discarded).

## The dataset

The dataset is obtained from the [SVHN dataset
](http://ufldl.stanford.edu/housenumbers/).  The training set has
73,257 images of street view house numbers, only 1,000 of them are
labelled (or rather, in this notebook, we mask the labels of the
rest.  See `Dataset.__init__()`). The test set has 26,032 images.

We divide the dataset into mini-batches for training (see `batches()` ).

## Network architecture

### Generator

The generator G is no different from a regular DCGAN generator that
learns the distribution of the training set.  Here, it is a 3-layer
CNN with batch normalization and a ReLU activation function.  Note the
use of `tf.layers.conv2d_transpose()`. G takes a random input vector
of a certain size ( say 100) and produces an image that matches the
size of the images in the dataset.

### Discriminator

The discriminator uses 6 convolutional layers.  Dropout is added for
regularization.  Batch normalization is used in all layers except the
very last one.  BN is not used in the last layer because of the use of
feature-matching loss later.  

An extra class, `fake` is added and a `softmax` is used over the 11
classes to get a prediction. Further explanation of feature matching,
how and why the `extra_class` value is set, are in comments in the
notebook.

## Loss definition and optimization

The optimizer is an [Adam optimizer](https://arxiv.org/abs/1412.6980).
The optimization is done by simultaneously minimizing the
discriminator loss `d_loss` and the generator loss `g_loss` at each
step of training.

The `d_loss` consists of three components which are summed together -
a softmax cross entropy loss on the images that are classified into
one of the 10 classes, and sigmoid cross entropy losses on the images
for the real/fake classification (the fake ones are compared with zero
and the real ones are compared with 1).

The generator loss is the "feature matching" loss from Tim Salimans
*et al* (see reference below) where the idea is to minimize the
absolute difference between the mean of various features (channels)
that are obtained when the real images and fake images are passed
through the discriminator.

As in the case of generative models, as the discriminator loss is
minimized, it gets better at telling fakes from the real images and
since the generator loss is being minimized simultaneously, it gets
better at faking the images.  Here, the extra training of the
discriminator is helping it get much better on the classification
problem, for which it has only a little data.

## References

The notebook is an implementation of the idea described in the 2016
paper [Improved Techniques for training
GANs](https://arxiv.org/pdf/1606.03498.pdf) by Tim Salimans, Ian
Goodfellow and co-workers and improves on the accuracy of a previous
attempt at using DCGANs for semi-supervised learning by Kingma et al
in 2014 [Semi-supervised learning with Deep Generative
Models](https://arxiv.org/pdf/1406.5298.pdf).  The guidance for the
implementation and some starter code is from Ian Goodfellow.