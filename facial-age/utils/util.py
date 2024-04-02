import os.path
import sys

import yaml
from PIL import ImageDraw
from torchvision import transforms
import json
import logging


logging_str = "[%(asctime)s: %(levelname)s: %(module)s: %(message)s]"
log_dir = "logs"
log_filepath = os.path.join(log_dir, 'running_logs.logs')
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format=logging_str,

    handlers=[
        logging.FileHandler(log_filepath),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("FacialAgeDetectionLogger")


def visualize_img(img, target, predict):
    img_pil = transforms.ToPILImage()(img)
    draw = ImageDraw.Draw(img_pil)
    draw.text((10, 10), f"True: {target}\nPred: {predict}", fill=(255, 255, 255))
    return img_pil


def save_json(filename, data):
    with open(filename, 'wt') as f:
        json.dump(data, f)


def read_yaml(filename):
    with open(filename, 'r') as f:
        return yaml.safe_load(f)
