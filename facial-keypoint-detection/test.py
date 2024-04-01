import torch
import torchvision
from torchvision import models, transforms
from PIL import Image
from training.visualzation import visualize_landmark
from IPython.display import display
model = models.detection.keypointrcnn_resnet50_fpn(
    weights=None,
    weights_backbone=None,
    num_keypoints=14,
    num_classes=2
)
model.load_state_dict(
    torch.load('facial_landmark_detection.pth', map_location=torch.device('cpu'))
)

img = Image.open('me.jpg').convert('RGB')
transform = transforms.Compose([
    transforms.ToTensor(),
])
img=transform(img)

model.eval()
pred = model([img])

(visualize_landmark(img, pred[0]))


print('end')