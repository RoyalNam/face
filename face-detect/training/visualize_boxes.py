from torchvision.transforms import ToPILImage
from PIL import ImageDraw


def visualize_boxes(img, target):
    img_pil = ToPILImage()(img)
    draw = ImageDraw.Draw(img_pil)
    boxes = target['boxes']
    for box in boxes:
        x0, y0, x1, y1 = box
        draw.rectangle([x0, y0, x1, y1], outline='green')
    return img_pil
