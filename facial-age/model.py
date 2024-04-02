from torchvision import models
from torch import nn


def get_model(num_classes, pretrained=False):
    if pretrained:
        model = models.efficientnet_b2(
            weights=None
        )
    else:
        model = models.efficientnet_b2(
            weights='DEFAULT'
        )
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features=1408, out_features=num_classes)
    )
    return model
