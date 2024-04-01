from PIL import ImageDraw
from torchvision.transforms import ToPILImage


def visualize_landmark(img, targets):
    img_pil = ToPILImage()(img)
    draw = ImageDraw.Draw(img_pil)
    kps = targets['keypoints']
    boxes = targets['boxes']
    for kp in kps[0]:
        x, y, _ = kp
        draw.ellipse([x-1, y-1, x+1, y+1], fill='red')
    for box in boxes:
        x_min, y_min, x_max, y_max = box
        draw.rectangle([x_min, y_min, x_max, y_max], outline='green')
    return img_pil
