import torch
from torch.utils.data import Dataset
import pandas as pd
import os
from PIL import Image
from config import CONFIG


class CustomDataset(Dataset):
    def __init__(self, image_dir, attr_path, transform=None):
        super().__init__()
        self.image_dir = image_dir
        self.all_attr_df = pd.read_csv(attr_path)
        self.transform = transform

    def __len__(self):
        return len(self.all_attr_df)

    def __getitem__(self, idx):
        filename = self.all_attr_df.iloc[idx, 0]
        img_path = os.path.join(self.image_dir, filename)
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        attr = self.all_attr_df.loc[idx, CONFIG.SELECTED_ATTRS].replace(-1, 0).astype(float)
        attr = torch.tensor(attr.values, dtype=torch.float32)

        return img, attr
