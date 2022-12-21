# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 12:41:55 2022

@author: Bestom
"""


#Cnn.py
import torch
from torch import nn
import numpy as np
import cv2

from SingleEmbedding import get_single_embedding

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(          # 1*28*28
                in_channels=1,  # 输入为单层图像
                out_channels=16,  # 卷积成16层    16*28*28
                kernel_size=5,  # 卷积壳5x5
                stride=1,  # 步长，每次移动1步
                padding=2,  # 边缘层，给图像边缘增加像素值为0的框      
            ),
            nn.ReLU(),  # 激活函数
            nn.MaxPool2d(kernel_size=2),  # 池化层，将图像长宽减少一半  16*14*14
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
                                                                    #32*14*14
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 2),
            nn.ReLU(),
            #nn.MaxPool2d(2),
                                                                    # 64*14*14
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 16, 3, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
                                                                    #16*7*7
        )

        self.out = nn.Linear(16*9*9, 960)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output


def save_model(net, path):
    torch.save(net, path)


def load_model(path):
    net = torch.load(path)
    return net

def predict(model, file):
    spect = get_single_embedding(file)
    spect = np.abs(spect)
    spect = cv2.resize(spect, (28, 28))
    data = torch.Tensor(spect)
    data = data.unsqueeze(0)
    data = data.unsqueeze(0)

    output = model(data)
    confidence, pred_y = torch.max(output, 1)
    print("识别结果为：",pred_y.numpy())
    
    output2,index=output.topk(k=900, dim=1, largest=True, sorted=True)
    print("是",output2)
    
    return output2,index