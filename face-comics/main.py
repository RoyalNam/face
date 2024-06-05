import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
import os
from logger import logger
from utils import load_json, save_checkpoint, load_checkpoint
from dataset.data_ingestion import download_and_extract
from dataset.data_preprocessing import MapDataset
from models.generator import Generator
from models.discriminator import Discriminator
from trainer import train_epoch


prefix = 'https://drive.google.com/uc?/export=download&id='
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def main():
    logger.info('>>>>>> start <<<<<<')
    config = load_json('config.json')
    data_ingestion_cf = config['data_ingestion']
    file_id = data_ingestion_cf['source_url'].split('/')[-2]
    os.makedirs(data_ingestion_cf['root_dir'], exist_ok=True)
    download_and_extract(prefix+file_id, data_ingestion_cf['output_path'], data_ingestion_cf['root_dir'])

    dataloader_cf = config['dataloader']
    transform = transforms.Compose([
        transforms.Resize((dataloader_cf['img_size'], dataloader_cf['img_size'])),
        transforms.ToTensor(),
    ])
    dataset = MapDataset(
        inputs_dir=dataloader_cf['input_dir'],
        target_dir=dataloader_cf['target_dir'],
        transform=transform
    )
    logger.info(f'Total number of samples in dataset: {len(dataset)}')
    dataloader = DataLoader(
        dataset,
        batch_size=dataloader_cf['batch_size'],
        shuffle=dataloader_cf['shuffle'],
        num_workers=dataloader_cf['num_workers']
    )
    logger.info(f'Number of batches i dataloader: {len(dataloader)}')

    gen = Generator(in_channels=3)
    disc = Discriminator(in_channels=3)
    gen.to(device)
    disc.to(device)
    opt_disc = torch.optim.Adam(disc.parameters(), lr=config['optimizer']['disc_lr'], betas=(0.5, 0.999))
    opt_gen = torch.optim.Adam(gen.parameters(), lr=config['optimizer']['gen_lr'], betas=(0.5, 0.999))

    BCE = nn.BCEWithLogitsLoss()
    L1_LOSS = nn.L1Loss()
    step = 0

    training_cf = config['training']
    if os.path.exists(training_cf['checkpoint_disc']):
        load_checkpoint(training_cf['checkpoint_disc'], disc, opt_disc, device)
    if os.path.exists(training_cf['checkpoint_gen']):
        load_checkpoint(training_cf['checkpoint_gen'], gen, opt_gen, device)

    writer_real = SummaryWriter('logs/real')
    writer_fake = SummaryWriter('logs/fake')
    g_scaler = torch.cuda.amp.GradScaler() if device == 'cuda' else None
    d_scaler = torch.cuda.amp.GradScaler() if device == 'cuda' else None

    for epoch in tqdm(range(training_cf['num_epochs'])):
        step = train_epoch(
            gen, disc, dataloader, BCE, L1_LOSS, opt_gen, opt_disc,
            device, writer_real, writer_fake, step, training_cf['l1_lambda'],
            d_scaler, g_scaler, log_step=100
        )
    torch.save(gen.state_dict(), 'face-comics-gen.pth')
    save_checkpoint(gen, opt_gen, training_cf['checkpoint_gen'])
    save_checkpoint(disc, opt_disc, training_cf['checkpoint_disc'])

    logger.info('>>>>>> end <<<<<<')


if __name__ == '__main__':
    main()
