import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
import pandas as pd


class FacialLandmarksDetectionDataset(Dataset):
    def __init__(self, images_dir, annotations_path, img_size=128):
        self.images_dir = images_dir
        self.img_size = img_size
        self.img_paths = os.listdir(images_dir)
        self.annotations = self.get_data(annotations_path)
        
    def get_data(self, annotations_path):
        df = pd.read_csv(annotations_path)
        filenames = df.iloc[:, 0].values
        annotations = df.iloc[:, 1:].values.reshape(-1, 68, 2)

        data_dict = {}
        for filename, annotation in zip(filenames, annotations):
            data_dict[filename] = annotation

        return data_dict
    
    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_filename = self.img_paths[idx]
        img_path = os.path.join(self.images_dir, img_filename)
        annotations = self.annotations[img_filename]
        
        img = Image.open(img_path).convert('RGB')
        w, h = img.width, img.height
        transform = transforms.Compose([
            # transforms.RandomHorizon(),
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor()
        ])
        img = transform(img)
        
        kps = torch.tensor(annotations, dtype=torch.float32)
        kps[:, 0] *= self.img_size / w
        kps[:, 1] *= self.img_size / h
        visibility = torch.ones_like(kps[:, 0])
        kps_full = torch.cat((kps, visibility.unsqueeze(1)), dim=1)

        min_x = min(kps[:, 0])
        max_x = max(kps[:, 0])
        min_y = min(kps[:, 1])
        max_y = max(kps[:, 1])

        boxes = torch.tensor([[min_x, min_y, max_x, max_y]], dtype=torch.float32)
        labels = torch.tensor([1])
        targets = {
            'boxes': boxes,
            'labels': labels,
            'keypoints': kps_full.unsqueeze(0)
        }
        return img, targets
