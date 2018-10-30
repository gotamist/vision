# Computer vision

All of the projects in this repository are applications of deep neural
networks to vision problems.  This is a brief summary.  More details
can be found within each project's directory.

## Facial recognition

Here, we look at a collection of data from the [YouTube Faces
database](https://www.cs.tau.ac.il/~wolf/ytfaces/), which has 5,770
color images of faces at various angles, all extracted from YouTube
videos.  In this supervised learning problem, each image in the
training set is labelled with 68 facial keypoints (corners of eyes,
tip of the nose, corners of the mouth, etc).  The trained network is
then used to identify the 68 keypoints on test images.

Finally, this is combined with a face detection [Haar
cascade](https://docs.opencv.org/3.3.0/d7/d8b/tutorial_py_face_detection.html)
to extract multiple faces from images and then identify the keypoints
in each photo.  Applications can include identifying whether a person
is smiling/forwning by observing the relative position of various
keypoints.

This is carried out using PyTorch.

## Image captioning

The dataset here is the Microsoft Common Objects in Context dataset
([MS COC0](http://cocodataset.org/#home)) which contains images and
multiple human generated captions describing them.  An architecture of
a CNN to encode images and then a sequence of RNN, dense layers and
softmax to decode into sentences is ideal for training them.  I have
added dropout for regularization.  Inference and validation can be
done with a greedy algorithm over the outputs of the CNN or a beam
search, which is also implemented.  The network is implemented with
PyTorch.

In cases where a project was done for a Udacity nanodegree program, a
README in within the project folder will identify it as such.