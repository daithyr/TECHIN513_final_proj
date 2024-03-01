import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import transforms


# %%
class NN(nn.Module):
    def __init__(self, arr=[]):
        super(NN, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(30 * 30 * 3, 128)
        self.fc2 = nn.Linear(128, 5)

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# %%
class SimpleCNN(nn.Module):
    def __init__(self, arr=[]):
        super(SimpleCNN, self).__init__()
        self.conv_layer = nn.Conv2d(3, 8, 3)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(1568, 5)

    def forward(self, x):
        """
        Question 2

        fill this forward function for data flow
        """
        x = self.pool(F.relu(self.conv_layer(x)))
        x = x.view(-1, 1568)  # Flatten the output for fully connected layer
        x = self.fc1(x)
        return x


# %%
basic_transformer = transforms.Compose([transforms.ToTensor()])

"""
Question 3

Add color normalization to the transformer. For simplicity, let us use 0.5 for mean
      and 0.5 for standard deviation for each color channel.
"""

norm_transformer = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])



# %%
class DeepCNN(nn.Module):
    def __init__(self, arr=[]):
        super(DeepCNN, self).__init__()
        """
        Question 4

        TODO: setup the structure of the network
        """
    
        layers = []
        in_channels = 3  # Initial number of input channels
        for item in arr:
            if isinstance(item, int):
                layers.append(nn.Conv2d(in_channels, item, kernel_size=3, padding=0))
                layers.append(nn.ReLU())
                in_channels = item
            elif item == "pool":
                layers.append(nn.MaxPool2d(kernel_size=2))
        self.features = nn.Sequential(*layers)
        # Calculate the size of the tensor after passing through convolutional layers
        self.fc_input_size = 32 * 12 * 12  # Calculated based on the output size of the last conv layer
        self.fc = nn.Linear(self.fc_input_size, 5)  # Output size is 5 for 5 classes

    def forward(self, x):
        """
        Question 4

        setup the flow of data (tensor)
        """
        x = self.features(x)
        x = x.view(-1, self.fc_input_size)  # Flatten the output tensor
        x = self.fc(x)
        return x


# %%
"""
Question 5


    change the aug_transformer to a tranformer with random horizontal flip
    and random affine transformation

    1. It should randomly flip the image horizontally with probability 50%
    2. It should apply random affine transformation to the image, which randomly rotate the image 
        within 5 degrees, and shear the image within 10 degrees.
    3. It should include color normalization after data augmentation. Similar to question 3.
"""

"""Add random data augmentation to the transformer"""
aug_transformer = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),  # Randomly flip the image horizontally with probability 50%
    transforms.RandomAffine(degrees=5, shear=10),  # Apply random affine transformation (rotation within 5 degrees, shear within 10 degrees)
    transforms.ToTensor(),  # Convert image to tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize color channels
])


