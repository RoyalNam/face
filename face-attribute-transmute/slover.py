import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torch import nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from tqdm.auto import tqdm
from IPython.display import display
from config import CONFIG
from dataset.dataset import CustomDataset
from model.generator import Generator
from model.discriminator import Discriminator
from utils import *


class Solver:
    def __init__(self):
        self.device = self.get_device()
        self.loader = self.get_data_loader()
        self.gen, self.disc, self.opt_gen, self.opt_disc = self.initialize_models_and_optimizers(self.device)

        if CONFIG.PRETRAINED:
            self.load_pretrained()
        else:
            self.initialize_weights()

        self.criterion_cycle, self.criterion_cls = self.get_loss_functions()
        self.scaler = GradScaler()
        self.writer = SummaryWriter()

        self.scheduler_G, self.scheduler_D = self.get_schedulers(self.opt_gen, self.opt_disc)
        self.x_fixed, self.c_fixed = self.get_fixed_samples(self.loader, self.device)

    def get_device(self):
        return 'cuda' if torch.cuda.is_available() else 'cpu'

    def get_data_loader(self, train_transforms):
        train_transforms = transforms.Compose([
            transforms.CenterCrop(CONFIG.CROP_SIZE),
            transforms.Resize(CONFIG.IMG_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        dataset = CustomDataset(
            CONFIG.ROOT_DIR,
            CONFIG.ATTR_PATH,
            transform=train_transforms
        )
        loader = DataLoader(
            dataset,
            batch_size=CONFIG.BATCH_SIZE,
            shuffle=True,
            num_workers=CONFIG.NUM_WORKERS
        )
        print(f'length of loader: {len(loader)}')
        return loader

    def initialize_models_and_optimizers(self, device):
        gen = Generator(CONFIG.G_CONV_DIM, CONFIG.C_DIM, CONFIG.N_RES)
        disc = Discriminator(CONFIG.IMG_SIZE, CONFIG.D_CONV_DIM, CONFIG.C_DIM, CONFIG.N_STRIDED)
        gen = gen.to(device)
        disc = disc.to(device)
        opt_gen = torch.optim.Adam(gen.parameters(), lr=CONFIG.G_LR, betas=(CONFIG.BETA1, CONFIG.BETA2))
        opt_disc = torch.optim.Adam(disc.parameters(), lr=CONFIG.D_LR, betas=(CONFIG.BETA1, CONFIG.BETA2))
        return gen, disc, opt_gen, opt_disc

    def load_pretrained(self):
        load_checkpoint(self.gen, self.opt_gen, 'gen.pth')
        load_checkpoint(self.disc, self.opt_disc, 'disc.pth')

    def initialize_weights(self):
        self.gen.apply(weights_init)
        self.disc.apply(weights_init)
        print('Initialized weights')

    def get_loss_functions(self):
        criterion_cycle = nn.L1Loss()
        criterion_cls = nn.BCEWithLogitsLoss()
        return criterion_cycle, criterion_cls

    def get_schedulers(self, opt_gen, opt_disc):
        lambda_lr = lambda epoch: 0.95 ** epoch
        scheduler_G = torch.optim.lr_scheduler.LambdaLR(opt_gen, lr_lambda=lambda_lr)
        scheduler_D = torch.optim.lr_scheduler.LambdaLR(opt_disc, lr_lambda=lambda_lr)
        return scheduler_G, scheduler_D

    def get_fixed_samples(self, loader, device):
        data_iter = iter(loader)
        x_fixed, c_fixed = next(data_iter)
        x_fixed = x_fixed.to(device)
        c_fixed = c_fixed.to(device)
        return x_fixed, c_fixed

    def train(self):
        for epoch in range(CONFIG.START_EPOCH, CONFIG.N_EPOCHS):
            loop = tqdm(self.loader, desc=f'Epoch {epoch + 1}')
            for i, (real_x, real_label) in enumerate(loop):
                # Gen fake labels randomly
                rand_idx = torch.randperm(real_label.size(0))
                fake_label = real_label[rand_idx]

                real_x = real_x.to(self.device)
                real_label = real_label.to(self.device)
                fake_label = fake_label.to(self.device)

                # Train the discriminator
                with autocast():
                    out_src, out_cls = self.disc(real_x)
                    d_loss_real = - torch.mean(out_src)
                    d_loss_cls = self.criterion_cls(out_cls, real_label) / out_cls.size(0)

                    x_fake = self.gen(real_x, fake_label)
                    out_src, _ = self.disc(x_fake.detach())
                    d_loss_fake = torch.mean(out_src)

                    gp = gradient_penalty(self.disc, real_x, x_fake, self.device)

                    d_loss = d_loss_real + d_loss_fake + d_loss_cls * CONFIG.LAMBDA_CLS + gp * CONFIG.LAMBDA_GP

                self.opt_disc.zero_grad()
                self.scaler.scale(d_loss).backward()
                self.scaler.step(self.opt_disc)
                self.scaler.update()

                # Logging
                loss = {
                    'D/loss_real': d_loss_real.item(),
                    'D/loss_fake': d_loss_fake.item(),
                    'D/loss_cls': d_loss_cls.item(),
                    'D/loss_gp': gp.item()
                }

                # Train the generator
                if (i + 1) % CONFIG.N_CRITIC == 0:
                    with autocast():
                        x_fake = self.gen(real_x, fake_label)
                        out_src, out_cls = self.disc(x_fake)
                        g_loss_fake = - torch.mean(out_src)
                        g_loss_cls = self.criterion_cls(out_cls, fake_label) / out_cls.size(0)

                        x_reconst = self.gen(x_fake, real_label)
                        g_loss_rec = self.criterion_cycle(x_reconst, real_x)

                        g_loss = g_loss_fake + g_loss_cls * CONFIG.LAMBDA_CLS + g_loss_rec * CONFIG.LAMBDA_REC

                    self.opt_gen.zero_grad()
                    self.scaler.scale(g_loss).backward()
                    self.scaler.step(self.opt_gen)
                    self.scaler.update()

                    # Logging
                    loss.update({
                        'G/loss_fake': g_loss_fake.item(),
                        'G/loss_cls': g_loss_cls.item(),
                        'G/loss_rec': g_loss_rec.item()
                    })

                # Display imgs
                if (i + 1) % (len(self.loader) // 5) == 0:
                    with torch.no_grad():
                        x_fake = self.gen(self.x_fixed[:6], self.c_fixed[:6])
                        imgs = make_grid(x_fake, normalize=True)
                        display(transforms.ToPILImage()(imgs))

                # Logging
                if (i + 1) % CONFIG.LOG_STEP == 0:
                    for tag, val in loss.items():
                        self.writer.add_scalar(tag, val, epoch * len(self.loader) + i)

                    loop.set_postfix(
                        d_loss=d_loss.item(),
                        g_loss=g_loss.item(),
                        lr=self.opt_gen.param_groups[0]['lr']
                    )

            # Save checkpoint
            save_checkpoint(self.gen, self.opt_gen, 'gen.pth')
            save_checkpoint(self.disc, self.opt_disc, 'disc.pth')
            # Update lr
            self.scheduler_G.step()
            self.scheduler_D.step()
