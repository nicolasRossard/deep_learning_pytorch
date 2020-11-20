import torch
import torchvision
import torchvision.transforms as transforms
 
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.optim as optim
  
torch.set_grad_enabled(True)
torch.set_printoptions(linewidth =120)
   
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
 
from sklearn.metrics import confusion_matrix
#from plotcm import plot_confusion_matrix
  
import pdb
   
#Note that pdb is the Python debugger

# Legend
## B: Batch
## D: Channels
## H: Height
## W: Witdh
## K: Number of filters
## F: Size of filters
## S: Stride
## P: Zero-padding
## [B, D, H, W]: paramaters about the picture before/after layer
class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        #Init CONV1 K=6, F=3, S=1, P=1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=1)
        
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=3, stride=1, padding=1)
                                     
        self.fc1 = nn.Linear(in_features=12*20*20, out_features=2400)
        self.fc2 = nn.Linear(in_features=2400, out_features=1200)
        self.out = nn.Linear(in_features=1200, out_features=2)

    def forward(self, t):
        # (1) input layer
        t = t
        # (2) hidden conv layer
        # [1,3,80,80] => [1,6,80,80]
        t = self.conv1(t)
        t = F.relu(t)
        

        # (3) hidden pool layer: F=2,S=2
        # [1,6,80,80] => [1,6,40,40]
        t = F.max_pool2d(t, kernel_size=2, stride=2)
        # (4) hidden conv layer
        # [1,3,80,80] => [1,12,40,40]
        t = self.conv2(t)
        t = F.relu(t)


        # (5) hidden pool layer: F=2, S=2
        # [1,12,40,40] => [1,12,20,20]
        t = F.max_pool2d(t, kernel_size=2, stride=2)
                      
        # (6) hidden linear layer
        t = t.reshape(-1, 12 * 20 * 20)
        t = self.fc1(t)
        t = F.relu(t)
                        
        # (7) hidden linear layer
        t = self.fc2(t)
        t = F.relu(t)
                         
        # (8) output layer
        t = self.out(t)
        #t = F.softmax(t, dim=1)
                                                                                                                                                          
        return t
