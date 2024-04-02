import torch
from torchvision import transforms
from dataset.dataloader import get_loader
from dataset.data_ingestion import download_and_extract
from utils.util import read_yaml, logger
from model import get_model
from training.train import Trainer
from torch.utils.tensorboard import SummaryWriter
import os

prefix = 'https://drive.google.com/uc?/export=download&id='
device = 'cuda' if torch.cuda.is_available() else 'cpu'

try:
    print('>>>>>>> start <<<<<<<<<')
    params = read_yaml('params.yaml')

    data_url = 'https://drive.google.com/file/d/1M8rU_zWtixKXS_df0wCgxkSEDSDO4xN8/view?usp=sharing'
    file_id = data_url.split('/')[-2]
    output_path = 'data/dataset.zip'
    extract_dir = 'data/'
    os.makedirs(extract_dir, exist_ok=True)
    download_and_extract(prefix+file_id, output_path, extract_dir)

    img_size = params['img_size']
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomCrop((img_size-32)),
        transforms.ToTensor(),
    ])

    train_dataloader, classes_name = get_loader(
        root_dir='data/face_age',
        transform=transform,
        batch_size=params['batch_size'],
    )
    pretrained=False
    model = get_model(len(classes_name), pretrained)
    if pretrained:
        model.load_state_dict(torch.load('age-detect.pth', map_location=torch.device(device)))
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])

    scaler = torch.cuda.amp.GradScaler() if device == 'cuda' else None
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=25, gamma=0.98, verbose=False
    )

    writer = SummaryWriter(f'logs/train')

    training = Trainer(
        model,
        train_dataloader,
        criterion,
        optimizer,
        device,
        lr_scheduler,
        writer,
        params['num_epochs'],
        scaler
    )
    training.main_loop()
    print('>>>>>>> end <<<<<<<<<')
except Exception as e:
    logger.error(f'Unexpected error: {e}')
    raise e
