## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):


    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        # Input is 1,224,224
        self.conv1 = nn.Conv2d(1, 32, 5) # if stride=1, the size is (32,220,220)
        I.kaiming_normal(self.conv1.weight)
        self.conv1_bn = nn.BatchNorm2d(32)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        # maxpool layer
        # pool with kernel_size=2, stride=2
        self.pool = nn.MaxPool2d(2, 2) #now it will be (32,110,110)
        self.conv2 = nn.Conv2d(32, 64, 5, stride=2) # now it is (64, 53,53 )
        I.kaiming_normal_(self.conv2.weight)
        self.conv2_bn = nn.BatchNorm2d(64)
        # After one more pool layer, it will be (64,26,26)
        self.conv3 = nn.Conv2d(64, 128, 3, stride=1) # now it is (128, 24,24 ).  After one more pooling, it should be (128,12,12)
        I.kaiming_normal_(self.conv3.weight)
        self.conv3_bn = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, 5, stride=2)# now it is (256, 4,4 ).  After one more pooling, it should be (256,2,2)
        I.kaiming_normal_(self.conv4.weight)
        self.conv4_bn = nn.BatchNorm2d(256)
        self.fc1 = nn.Linear( 256*2*2, 512 )
        I.kaiming_normal_(self.fc1.weight)
        # dropout with p=0.5
        self.drop_layer = nn.Dropout(p=0.2)
        # finally, create 68*2 output channels (x and y for each of the 68 keytpoints)
        self.fc2 = nn.Linear(512, 256)
        I.kaiming_normal_(self.fc2.weight)
        self.fc3 = nn.Linear(256, 68*2)
        I.kaiming_normal_(self.fc3.weight)
        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv1_bn(self.conv1(x)))) # (32,110,110)
        x = self.drop_layer(x)
#        print(x.shape)
        x = self.pool(F.relu(self.conv2_bn(self.conv2(x)))) # (64,26,26)
        x = self.drop_layer(x)
#        print(x.shape)
        x = self.pool(F.relu(self.conv3_bn(self.conv3(x)))) # (128,12,12)
        x = self.drop_layer(x)
#        print(x.shape)
        x = self.pool(F.relu(self.conv4_bn(self.conv4(x)))) # (256,2,2)
        x = self.drop_layer(x)
#        print(x.shape)
        # prep for linear layer by flattening 
        x = x.view(x.size(0), -1)
        
        # two linear layers with dropout in between
        x = F.relu(self.fc1(x))
#        print(x.shape)
        x = self.drop_layer(x)
#        print(x.shape)
        x = F.relu(self.fc2(x))
#        print(x.shape)
        x = self.drop_layer(x)
#        print(x.shape)
        x = self.fc3(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x

