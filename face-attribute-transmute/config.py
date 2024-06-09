class Config:
    # Data config
    ROOT_DIR = ''
    ATTR_PATH = ''

    # Model config
    SELECTED_ATTRS = ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young']
    C_DIM = len(SELECTED_ATTRS)
    CROP_SIZE = 178 # as celeba
    IMG_SIZE = 128
    G_CONV_DIM = 64
    D_CONV_DIM = 64
    N_RES = 6
    N_STRIDED = 6
    LAMBDA_CLS = 1
    LAMBDA_REC = 10
    LAMBDA_GP = 10

    # Training config
    START_EPOCH = 0
    N_EPOCHS = 10
    BATCH_SIZE = 64
    NUM_WORKERS = 1
    PRETRAINED = False
    G_LR = 1e-4
    D_LR = 1e-4
    N_CRITIC = 5
    BETA1 = 0.5
    BETA2 = 0.999

    # Logging
    LOG_STEP = 10


CONFIG = Config()