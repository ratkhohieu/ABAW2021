from abc import ABC

import torch.nn as nn
import torch
from torchvision import models


class r2plus1d_18(nn.Module, ABC):
    def __init__(self, num_class=3):
        super(r2plus1d_18, self).__init__()
        self.r2plus1d = models.video.r2plus1d_18(pretrained=True)
        self.r2plus1d.fc = nn.Linear(in_features=self.r2plus1d.fc.in_features, out_features=num_class)

    def forward(self, x):
        return self.r2plus1d(x)
