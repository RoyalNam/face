from torchvision import models


def get_model(num_keypoints):
    model = models.detection.keypointrcnn_resnet50_fpn(
        weights_backbone='DEFAULT',
        num_keypoints=num_keypoints,
        num_classes=2
    )
    return model
