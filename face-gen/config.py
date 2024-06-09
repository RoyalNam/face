import torch


class Config:
    # data ingestion
    SOURCE_URL = ''
    EXTRACT_DIR = 'data'
    OUTPUT_FILEPATH = 'data/dataset.zip'
    # dataloader
    ROOT_DIR = ''
    BATCH_SIZE = 32
    IMAGE_SIZE = 128
    IMAGE_CHANNELS = 3

    # model
    Z_DIM = 100
    FEATURES_G = 64
    FEATURES_D = 64
    LAMBDA_GP = 10
    # training
    N_EPOCHS = 5
    G_LR = 1e-4
    D_LR = 1e-4
    CRITIC_ITERATIONS = 5
    PRETRAINED = False
    BETA1 = 0.0
    BETA2 = 0.999
    FIXED_NOISE = torch.randn((8, Z_DIM, 1, 1))


CONFIG = Config()
