from abc import ABC

import torch
from torch import nn
from models import *
from torchvision import models

from .resnet import resnet50, resnet18
from .densenet import densenet169, densenet121, densenet161


class TwoStreamAuralVisualModel(nn.Module, ABC):
    def __init__(self, input_face=256, input_audio=64):
        super(TwoStreamAuralVisualModel, self).__init__()
        self.video_model = resnet50(num_classes=input_face, pretrained=True)
        self.audio_model = resnet18(num_classes=input_audio, pretrained=True)
        self.fc = nn.Sequential(nn.Dropout(0.0),
                                nn.Linear(in_features=input_face + input_audio, out_features=3))
        self.modes = ['face', 'audio']

    def forward(self, face, audio):
        audio_model_features = self.audio_model(audio)
        video_model_features = self.video_model(face)

        features = torch.cat([audio_model_features, video_model_features], dim=1)
        out = self.fc(features)

        return out


class Identity(nn.Module, ABC):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Resnet_Multitask(nn.Module, ABC):
    def __init__(self, pretrained):
        super(Resnet_Multitask, self).__init__()
        self.backbone = models.resnet50(pretrained=pretrained)
        # self.fc = nn.Linear(in_features=self.backbone.classifier.in_features, out_features=20)
        self.fc1 = nn.Linear(in_features=self.backbone.fc.in_features, out_features=3)
        self.fc2 = nn.Linear(in_features=self.backbone.fc.in_features, out_features=7)

        self.backbone.fc = Identity()

    def forward(self, x):
        x = self.backbone(x)
        out1 = self.fc1(x)
        out2 = self.fc2(x)

        return out1, out2


class Densenet_Multitask(nn.Module, ABC):
    def __init__(self, pretrained, get_backbone=False):
        super(Densenet_Multitask, self).__init__()
        self.backbone = densenet169(pretrained=pretrained)
        self.fc1 = nn.Linear(in_features=self.backbone.classifier.in_features, out_features=3)
        self.fc2 = nn.Linear(in_features=self.backbone.classifier.in_features, out_features=7)

        self.backbone.classifier = Identity()
        self.get_backbone = get_backbone

    def forward(self, x):
        x = self.backbone(x)
        out1 = self.fc1(x)
        out2 = self.fc2(x)
        if self.get_backbone:
            return x
        return out1, out2



class Densenet_Multitask_CCD(nn.Module, ABC):
    def __init__(self, pretrained):
        super(Densenet_Multitask_CCD, self).__init__()

        self.backbone = densenet169(pretrained=pretrained)

        self.fc1 = nn.Linear(in_features=self.backbone.classifier.in_features, out_features=3)

        self.fc2 = nn.Linear(in_features=self.backbone.classifier.in_features, out_features=7)

        # self.fc1_aff = nn.Linear(in_features=self.backbone.classifier.in_features, out_features=2)
        # self.fc2_aff = nn.Linear(in_features=self.backbone.classifier.in_features, out_features=7)
        # self.fc3_aff = nn.Linear(in_features=self.backbone.classifier.in_features, out_features=136)
        #
        # self.fc1_afe = nn.Linear(in_features=self.backbone.classifier.in_features, out_features=2)

        self.backbone.classifier = Identity()

    # def forward_affectnet(self, x):
    #     x = self.backbone(x)
    #     out1 = self.fc1_aff(x)
    #     out2 = self.fc2_aff(x)
    #     out3 = self.fc3_aff(x)
    #     return out1, out2, out3

    def forward_kerc2020(self, x):
        x = self.backbone(x)
        out1 = self.fc1(x)
        return out1
    #
    # def forward_afew(self, x):
    #     x = self.backbone(x)
    #     out1 = self.fc1_afe(x)
    #     return out1

    def forward_kerc2019(self, x):
        x = self.backbone(x)
        out2 = self.fc2(x)
        return out2


class Densenet_Multitask_AFF(nn.Module, ABC):
    def __init__(self, pretrained):
        super(Densenet_Multitask_AFF, self).__init__()
        self.backbone = models.densenet169(pretrained=pretrained)

        self.fc1 = nn.Linear(in_features=self.backbone.classifier.in_features, out_features=2)
        self.fc2 = nn.Linear(in_features=self.backbone.classifier.in_features, out_features=7)
        self.fc3 = nn.Linear(in_features=self.backbone.classifier.in_features, out_features=136)

        self.backbone.classifier = Identity()

    def forward(self, x):
        x = self.backbone(x)
        out1 = self.fc1(x)
        out2 = self.fc2(x)
        out3 = self.fc3(x)
        return out1, out2, out3