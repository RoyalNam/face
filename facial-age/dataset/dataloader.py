import torch
from utils.util import logger
from torchvision import datasets
from torch.utils.data import DataLoader


def get_loader(root_dir, transform, batch_size=64):
    dataset = datasets.ImageFolder(root_dir, transform=transform)
    classes_name = dataset.classes
    logger.info(f'Total number of samples in dataset: {len(dataset)}')

    class_counts = torch.bincount(torch.tensor(dataset.targets))
    max_class_count = torch.max(class_counts)

    class_weights = max_class_count / class_counts.float()
    sample_weights = [class_weights[target] for target in dataset.targets]

    sampler = torch.utils.data.WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    logger.info(f'Total classes name: {len(classes_name)}')

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  
        sampler=sampler 
    )
    logger.info(f'Number of batches in dataloader: {len(dataloader)}')

    return dataloader, classes_name
