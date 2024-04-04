from torch.utils.data import DataLoader


def get_dataloader(dataset, batch_size, shuffle=True, num_workers=1, collate_fn=None):
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    return dataloader
