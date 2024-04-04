import os
import torch.cuda
from utils.logger import logger
from utils.utils import load_json, collate_fn
from dataset.data_ingestion import download_and_extract
from dataset.preprocessing import FaceDetectionDataset
from dataset.dataloader import get_dataloader
from model import get_model
from training.train import Trainer
from torch.utils.tensorboard import SummaryWriter


prefix = 'https://drive.google.com/uc?/export=download&id='
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def main():
    logger.info('>>>>>> start <<<<<<')
    params = load_json('params.json')
    save_filepath = 'face-detect.pth'

    source_url = 'https://drive.google.com/file/d/1QkrHHwlg4SNcJWTeTXA2YGH5es6E_DcT/view?usp=sharing'
    file_id = source_url.split('/')[-2]
    extract_dir = 'data'
    output_path = 'data/dataset.zip'
    os.makedirs(extract_dir, exist_ok=True)
    download_and_extract(prefix+file_id, output_path, extract_dir)

    dataset = FaceDetectionDataset(
        imgs_dir='data/images',
        object_regions_path='data/faces.csv',
        img_size=params['img_size']
    )
    logger.info(f'Total number of samples in dataset: {len(dataset)}')
    dataloader = get_dataloader(
        dataset,
        batch_size=params['batch_size'],
        shuffle=params['shuffle'],
        collate_fn=collate_fn
    )
    logger.info(f'Number of batches in dataloader: {len(dataloader)}')

    pretrained = True
    model = get_model(params['num_classes'], pretrained)
    model.to(device)
    if not pretrained:
        model.load_state_dict(
            torch.load('face-detect.pth', map_location=torch.device(device))
        )
    logger.info(f'load model successfully')
    optimizer = torch.optim.SGD(model.parameters(), lr=params['lr'], momentum=0.9, weight_decay=0.0005)
    scaler = torch.cuda.amp.GradScaler() if device == 'cuda' else None
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=10,
        gamma=0.98,
        verbose=False
    )
    writer = SummaryWriter('logs/train')
    training = Trainer(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        writer=writer,
        device=device,
        save_filepath=save_filepath,
        scaler=scaler,
        num_epochs=params['num_epochs']
    )
    training.main()

    logger.info(f'>>>>> end <<<<<<')


if __name__ == '__main__':
    main()
