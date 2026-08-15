import torch
import torch.nn as nn
from torchvision import models


class OrdinalKneeClassifier(nn.Module):

    def __init__(
        self,
        num_classes=5,
        pretrained=True,
        dropout=0.3,
        freeze_backbone=True
    ):

        super().__init__()

        weights = (
            models.ResNet50_Weights.DEFAULT
            if pretrained
            else None
        )

        self.backbone = models.resnet50(
            weights=weights
        )

        if freeze_backbone:

            for parameter in self.backbone.parameters():
                parameter.requires_grad = False

        input_features = self.backbone.fc.in_features

        self.backbone.fc = nn.Sequential(

            nn.Linear(
                input_features,
                512
            ),

            nn.BatchNorm1d(512),

            nn.ReLU(inplace=True),

            nn.Dropout(dropout),

            nn.Linear(
                512,
                num_classes - 1
            )
        )

        self.num_classes = num_classes

    def forward(self, x):

        return self.backbone(x)

    def unfreeze_last_layers(self):

        for parameter in self.backbone.layer3.parameters():
            parameter.requires_grad = True

        for parameter in self.backbone.layer4.parameters():
            parameter.requires_grad = True

        for parameter in self.backbone.fc.parameters():
            parameter.requires_grad = True

    def trainable_parameters(self):

        return [
            parameter
            for parameter in self.parameters()
            if parameter.requires_grad
        ]