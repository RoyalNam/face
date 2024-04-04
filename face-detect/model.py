from torchvision import models


def get_model(num_classes, pretrain=True):
    if pretrain:
        model = models.detection.fasterrcnn_mobilenet_v3_large_fpn(
            weights='DEFAULT',
        )
        print(model)
    else:
        model = models.detection.fasterrcnn_mobilenet_v3_large_fpn(
            weights='None'
        )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = models.detection.faster_rcnn.FastRCNNPredictor(
        in_features, num_classes
    )
    return model
