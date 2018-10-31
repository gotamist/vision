# Facial Keypoint Detection

This is a supervised training task for detecting facial keypoints (68
of them for each face, such as corners of eyes and mouths, ridge and
tip of nose) when shown an image with a human face in it.  The dataset
for the facee is the [YouTube Faces
DB](https://www.cs.tau.ac.il/~wolf/ytfaces/) which has 5,770 color
images of faces at various angles, all extracted from YouTube videos.

The main network architecture is defined in models.py.  The notebook
2_Define_the_Network_Architecture.ipynb is where the Neural
network is trained after defining the loss and the optimization
parameters.  Then it is tested on a test set. At the end, there is
also a section to visualize what various trained layers in the network
actually do (edge detection, for example).

In 3_Facial_Keypoint_Detection_Complete_Pipeline.ipynb, [Haar
cascades](https://docs.opencv.org/3.3.0/d7/d8b/tutorial_py_face_detection.html)
are employed to detect any number of faces in images.  Then, the
previously trained CNN is used to detect the keypoints for each face
that is detected by the Haar cascades in the image.

The netowrk architectures are defined in models.py.  One is the
architecture that I've construcuted.  Here, I have four convolutional
layers.  The output is then flattened and fed to two fully connected
layers.  These layers have dropout between them and I've kept the
dropout ratio constant.  The activation function is ReLU and I've used
He initialization [Kaiming He et al,
2015](https://arxiv.org/pdf/1502.01852v1.pdf) to mitigate the
vanishing/exploding gradients problem.

I've also included the architecture in the
[paper](https://arxiv.org/pdf/1710.00977.pdf) by *Naimish et
al*. Here, the six dropout layers have progressively increasing
dropout ratios, maxpool layers are used and the initialization is
`kaiming_uniform` instead of `kaiming_normal`.  Otherwise, the
architecture is similar to the one above.

Changes were made in files 2_...and 3_... to allow them to train the
model on multiple GPUs in parallel.

This project was done as part of the Udacity Computer Vision
nanodegree program.

