import os
from PIL import Image
from torch.utils.data import Dataset


class MapDataset(Dataset):
    def __init__(self, inputs_dir, target_dir, transform=None):
        self.inputs_dir = inputs_dir
        self.targets_dir = target_dir
        self.input_paths = os.listdir(inputs_dir)
        self.transform = transform

    def __len__(self):
        return len(self.input_paths)

    def __getitem__(self, idx):
        filename = self.input_paths[idx]
        input_path = os.path.join(self.inputs_dir, filename)
        target_path = os.path.join(self.targets_dir, filename)

        input_img = Image.open(input_path).convert('RGB')
        target_img = Image.open(target_path).convert('RGB')
        if self.transform:
            input_img = self.transform(input_img)
            target_img = self.transform(target_img)

        return input_img, target_img
