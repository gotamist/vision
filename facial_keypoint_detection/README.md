This is a supervised training task for detecting facial keypoints (68
of them for each face).

The main network architecture is defined in models.py.  The notebook
2_Define_the_Network_Architecture.ipynb is where the data is Neural
network is trained after defining the loss and the optimization
parameters.  Then it si tested on a test set. At the end, there is
also a section to visualize what various trained layers in the network
actually do (edge detection, for example).

And in 3_Facial_Keypoint_Detection_Complete_Pipeline.ipynb, Haar
cascades are employed to detect any number of faces in images.  Then,
the previously trained CNN is used to detect the keypoints for each
face that is detected by the Haar Casscades.
