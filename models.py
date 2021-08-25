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
        
        # input = 224 , out = (224-5) + 1 = 220 (32 , 220 , 220)
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.bn1   = nn.BatchNorm2d(32)
        # input = 220 , out = 110 , (32 , 110 , 110)
        self.pool1 = nn.MaxPool2d(2,2)
        self.drop1 = nn.Dropout(p=0.1)
        
        # input = 110 , out = (110-4) + 1 = 107 (64 , 107 , 107)
        self.conv2 = nn.Conv2d(32, 64, 4)
        self.bn2   = nn.BatchNorm2d(64)
        # input = 107 , out = 53 (53 , 64 , 64)
        self.pool2 = nn.MaxPool2d(2,2)
        self.drop2 = nn.Dropout(p=0.2)
        
        # input = 53 , out = (53-3)+1 = 51 ( 128 , 51 , 51)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.bn3   = nn.BatchNorm2d(128)
        # out = (128 , 25 , 25)
        self.pool3 = nn.MaxPool2d(2,2)
        self.drop3 = nn.Dropout(p=0.3)
        
        # input = 25 , out = (25-2)+1 = 24 (256 , 24 , 24)
        self.conv4 = nn.Conv2d(128, 256, 2)
        self.bn4   = nn.BatchNorm2d(256)
        #out = (256 , 12 , 12)
        self.pool4 = nn.MaxPool2d(2,2)
        self.drop4 = nn.Dropout(p=0.4)
        
        # input = 12 , out = (12-2) + 1 = 11 (512 , 11 , 11)
        self.conv5 = nn.Conv2d(256, 512, 2)
        self.bn5   = nn.BatchNorm2d(512)
        # out = (512 , 5 , 5)
        self.pool5 = nn.MaxPool2d(2,2)
        self.drop5 = nn.Dropout(p=0.4)
        
        self.fc1      = nn.Linear(12800, 1000)
        self.bn_fc1   = nn.BatchNorm1d(1000)
        self.drop_fc1 = nn.Dropout(p=0.3)
        
        self.fc2      = nn.Linear(1000, 500)
        self.bn_fc2   = nn.BatchNorm1d(500)
        self.drop_fc2 = nn.Dropout(p=0.2)
        
        self.fc3      = nn.Linear(500, 136)
        
        I.xavier_uniform(self.fc1.weight.data)
        I.xavier_uniform(self.fc2.weight.data)
        I.xavier_uniform(self.fc3.weight.data)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        
        # Pass it through CNN
        x = self.drop1(self.pool1(F.relu(self.bn1(self.conv1(x)))))
        x = self.drop2(self.pool2(F.relu(self.bn2(self.conv2(x)))))
        x = self.drop3(self.pool3(F.relu(self.bn3(self.conv3(x)))))
        x = self.drop4(self.pool4(F.relu(self.bn4(self.conv4(x)))))
        x = self.drop5(self.pool5(F.relu(self.bn5(self.conv5(x)))))
        
        # flatten
        x = x.view(x.size(0),-1)
                   
        # Pass it through FNN
        x = self.drop_fc1(F.relu(self.bn_fc1(self.fc1(x))))
        x = self.drop_fc2(F.relu(self.bn_fc2(self.fc2(x))))
        x = self.fc3(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
