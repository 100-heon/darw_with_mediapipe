import os
import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
from collections import deque

class CNNHandGestureModel(nn.Module):
    def __init__(self):
        super(CNNHandGestureModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))
        self.fc1 = nn.Linear(64 * 5 * 2, 128)
        self.fc2 = nn.Linear(128, 6)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 64 * 5 * 2)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x