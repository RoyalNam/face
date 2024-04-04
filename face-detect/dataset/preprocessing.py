import os
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset


class FaceDetectionDataset(Dataset):
    def __init__(self, imgs_dir, object_regions_path, img_size):
        self.imgs_dir = imgs_dir
        self.object_regions_df = pd.read_csv(object_regions_path)
        self.img_size = img_size
        self.img_paths = os.listdir(imgs_dir)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_filename = self.img_paths[idx]
        img_path = os.path.join(self.imgs_dir, img_filename)
        img = Image.open(img_path).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
        ])
        img = transform(img)

        filtered_df = self.object_regions_df[self.object_regions_df['image_name'] == img_filename]
        boxes = filtered_df.apply(
            lambda row: [
                row['x0'] * self.img_size / row['width'],
                row['y0'] * self.img_size / row['height'],
                row['x1'] * self.img_size / row['width'],
                row['y1'] * self.img_size / row['height']
            ], axis=1
        ).tolist()
        labels = [1] * len(filtered_df)
        area = (filtered_df['x1'] * filtered_df['x0']) * (filtered_df['y1'] - filtered_df['y0'])
        iscrowd = torch.ones((len(boxes),), dtype=torch.int64) if len(boxes) > 1 else torch.zeros((len(boxes),), dtype=torch.int64)
        img_id = torch.tensor([idx])

        target = {
            'labels': torch.tensor(labels, dtype=torch.int64),
            'boxes': torch.tensor(boxes, dtype=torch.float32),
            'area': torch.tensor(area.tolist(), dtype=torch.float32),
            'iscrowd': iscrowd,
            'img_id': img_id
        }
        return img, target
