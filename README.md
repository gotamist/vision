# Computer vision

All of the projects in this repository are applications of deep neural
networks to vision problems.  This is a brief summary.  More details
can be found within each project's directory.

## Facial keypoint detection

Here, we look at a collection of data from the [YouTube Faces
database](https://www.cs.tau.ac.il/~wolf/ytfaces/), which has 5,770
color images of faces at various angles, all extracted from YouTube
videos.  In this supervised learning problem, each image in the
training set is labelled with 68 __facial keypoints__ (corners of eyes,
tip of the nose, corners of the mouth, etc).  The trained network is
then used to identify the keypoints on test images.

Finally, this is combined with a __face detection__ [Haar
cascade](https://docs.opencv.org/3.3.0/d7/d8b/tutorial_py_face_detection.html)
to extract multiple faces from images and then identify the keypoints
in each photo.  Applications can include identifying emotions, such
whether a person is smiling or frowning by observing the relative
position of various keypoints.

This is carried out using __PyTorch__.

## Image captioning

The dataset here is the Microsoft Common Objects in Context dataset
([MS COCO](http://cocodataset.org/#home)) which contains images and
multiple human generated captions describing them.  An architecture of
a CNN to __encode images__ and then a sequence of RNN, dense layers and
softmax to __decode into sentences__ is ideal for training them.  I have
added dropout for regularization.  __Inference and validation__ can be
done with a greedy algorithm over the outputs of the CNN or a __beam
search__, which is also implemented.  The network is implemented with
__PyTorch__.

## SLAM - Simultaneous Localization And Mapping

Autonomous vehicles and robots need to infer the position of objects
in their environment as well as their own position within the
environment by using the imprecise measurements that they have of
distances to landmarks taken from different locations as well as
estimates of the distance moved by the robot between its different
positions (poses).  There are combined with __Kalman filtering__ to get a
good __estimate of the poses and the landmarks__ via the SLAM algorithm.

## Semi-supervised learning using GANs

This project shows how a GAN can be used to do semi-supervised
training, where a large dataset is available for training, but only a
small amount of it is labeled. The problem here is to classify digits
from the [Street View House Numbers (SVHN)
dataset](http://ufldl.stanford.edu/housenumbers) as one of 10 digits.

This is done by making the discriminator train to classify two
problems - to tell real images from those generated by the
discriminator, and if the image is identified as real, to classify it
as one of ten digits. More details inside the directory.

## The machine where the models are trained

Finally, I'm including the details of the [multi-GPU
machine](https://github.com/gotamist/other_machine_learning/tree/master/deep-learning-machine)
that I built. The computations were partially performed on this
machine. Some of them were also run on a cloud (AWS).

### Notes
In cases where a project was done for a Udacity nanodegree program, a
README in within the project folder will identify it as such.
