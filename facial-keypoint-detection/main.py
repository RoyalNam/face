import torch
import os
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils import load_json, collate_fn, logger, start_tensorboard, stop_tensorboard, save_model
from dataset.dataset import FacialLandmarksDetectionDataset
from dataset.data_ingestion import download_and_extract
from training.train import train_epoch
from model.model import get_model
from tqdm import tqdm


def main():
    try:
        print("===== start ======")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        params = load_json('params.json')

        url = "https://drive.google.com/file/d/1X6bIAhpL_kAhYO5b5COgI-sizh3csqBk/view?usp=sharing"
        file_id = url.split('/')[-2]
        prefix = 'https://drive.google.com/uc?/export=download&id='
        output_path = 'data/dataset.zip'
        extract_dir = 'data'
        os.makedirs(extract_dir, exist_ok=True)
        download_and_extract(prefix + file_id, output_path, extract_dir)
        logger.info(f'Download data from {url} into the {output_path}')
        logger.info(f'Successfully extracted {output_path} to {extract_dir}')

        dataset = FacialLandmarksDetectionDataset(
            images_dir='data/new_data/training',
            annotations_path='data/new_data/training.csv',
            img_size=params['img_size']
        )
        logger.info(f"Total number of samples in dataset: {len(dataset)}")

        loader = DataLoader(
            dataset,
            batch_size=params['batch_size'],
            shuffle=params['shuffle'],
            num_workers=params['num_workers'],
            collate_fn=collate_fn
        )
        logger.info(f"Number of batches in loader: {len(loader)}")

        model = get_model(params['num_keypoints'])
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=params['lr'],
            total_steps=params['num_epochs'] * len(loader)
        )
        scaler = torch.cuda.amp.GradScaler() if device == 'cuda' else None
        step = 0
        writer_true = SummaryWriter(f'logs/true')
        writer_pred = SummaryWriter(f'logs/pred')

        # Start TensorBoard
        tb_process = start_tensorboard('logs')

        for epoch in tqdm(range(params['num_epochs'])):
            train_loss, step = train_epoch(
                model=model,
                loader=loader,
                optimizer=optimizer,
                scaler=scaler,
                lr_scheduler=lr_scheduler,
                device=device,
                writer_true=writer_true,
                writer_pred=writer_pred,
                step=step,
                epoch=epoch
            )
            logger.info(f'Epoch: [{epoch+1}/{params["num_epochs"]}] | train_loss: {train_loss}')

        save_model(model.state_dict(), 'facial_landmark_detection.pth')
        # Stop TensorBoard
        stop_tensorboard(tb_process)
        print("===== end ======")

    except Exception as e:
        logger.error(f'Unexpected error: {e}')
        raise e


if __name__ == '__main__':
    main()
