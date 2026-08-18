import torch
import torch.nn as nn
from torchvision.models import (
    resnet50,
    ResNet50_Weights
)


class OrdinalResNet50(nn.Module):

    def __init__(
        self,
        num_classes=5,
        pretrained=True
    ):

        super().__init__()

        if pretrained:

            weights = ResNet50_Weights.IMAGENET1K_V2

        else:

            weights = None

        self.backbone = resnet50(
            weights=weights
        )

        in_features = (
            self.backbone.fc.in_features
        )

        # 4 ordinal thresholds for 5 grades:
        #
        # Grade >= 1
        # Grade >= 2
        # Grade >= 3
        # Grade >= 4
        #
        self.backbone.fc = nn.Sequential(

            nn.Dropout(
                p=0.30
            ),

            nn.Linear(
                in_features,
                512
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.Dropout(
                p=0.20
            ),

            nn.Linear(
                512,
                num_classes - 1
            )
        )

    def forward(self, x):

        return self.backbone(x)

    def freeze_backbone(self):

        for parameter in (
            self.backbone.parameters()
        ):

            parameter.requires_grad = False

        for parameter in (
            self.backbone.fc.parameters()
        ):

            parameter.requires_grad = True

    def unfreeze_layers_2_to_4(self):

        for parameter in (
            self.backbone.layer2.parameters()
        ):

            parameter.requires_grad = True

        for parameter in (
            self.backbone.layer3.parameters()
        ):

            parameter.requires_grad = True

        for parameter in (
            self.backbone.layer4.parameters()
        ):

            parameter.requires_grad = True

        for parameter in (
            self.backbone.fc.parameters()
        ):

            parameter.requires_grad = True