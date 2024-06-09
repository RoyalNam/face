import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from utils import *
from config import CONFIG
from model.generator import Generator
from model.discriminator import Discriminator
from tqdm.auto import tqdm
from IPython import display
from torchvision.utils import make_grid


device = 'cuda' if torch.cuda.is_available() else 'cpu'


def main():
    download_and_extract(CONFIG.SOURCE_URL, CONFIG.OUTPUT_FILEPATH, CONFIG.EXTRACT_DIR)

    transform_ = transforms.Compose([
        transforms.Resize(CONFIG.IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5 for _ in range(CONFIG.IMAGE_CHANNELS)],
            [0.5 for _ in range(CONFIG.IMAGE_CHANNELS)],
        )
    ])
    dataset = ImageFolder(
        CONFIG.ROOT_DIR,
        transform=transform_
    )
    print(f'len dataset: {len(dataset)}')

    loader = DataLoader(
        dataset,
        batch_size=CONFIG.BATCH_SIZE,
        shuffle=True,
        drop_last=True
    )
    print(f'len loader: {len(loader)}')

    gen = Generator(CONFIG.Z_DIM, CONFIG.IMAGE_CHANNELS, CONFIG.FEATURES_G).to(device)
    disc = Discriminator(CONFIG.IMAGE_CHANNELS, CONFIG.FEATURES_D).to(device)

    opt_gen = torch.optim.Adam(
        gen.parameters(), lr=CONFIG.G_LR, betas=(CONFIG.BETA1, CONFIG.BETA2)
    )
    opt_disc = torch.optim.Adam(
        disc.parameters(), lr=CONFIG.D_LR, betas=(CONFIG.BETA1, CONFIG.BETA2)
    )

    if CONFIG.PRETRAINED:
        load_checkpoint(gen, opt_gen, device, 'gen.pth')
        load_checkpoint(disc, opt_disc, device, 'disc.pth')
    else:
        gen.apply(weights_init)
        disc.apply(weights_init)
        print('Weights initial!')

    # Training
    for epoch in tqdm(range(CONFIG.N_EPOCHS)):
        gen.train()
        disc.train()

        loop = tqdm(loader, desc=f'Epoch {epoch + 1}')
        for batch_idx, (real, _) in enumerate(loop):
            real = real.to(device)

            # Train Disc: max E[disc(real)] - E[disc(fake)]
            for _ in range(CONFIG.CRITIC_ITERATIONS):
                noise = torch.randn((CONFIG.BATCH_SIZE, CONFIG.Z_DIM, 1, 1)).to(device)
                fake = gen(noise)
                D_real = disc(real)
                D_fake = disc(fake.detach())
                gp = gradient_penalty(disc, real, fake, device=device)
                loss_D = -(torch.mean(D_real) - torch.mean(D_fake)) + CONFIG.LAMBDA_GP * gp

                disc.zero_grad()
                loss_D.backward(retain_graph=True)
                opt_disc.step()

            # Train Generator: max E[disc(gen_fake)] <-> min -E[disc(gen_fake)]
            gen_fake = disc(fake)
            loss_G = -torch.mean(gen_fake)

            gen.zero_grad()
            loss_G.backward()
            opt_gen.step()
            loop.set_postfix(loss_D=loss_D.item(), loss_G=loss_G.item())
            loop.update()

            if batch_idx > 0 and batch_idx % (len(loader) // 3 - 1) == 0:
                with torch.no_grad():
                    fake = gen(CONFIG.FIXED_NOISE) * 0.5 + 0.5
                    grid_fake = make_grid(fake)
                    display(transforms.ToPILImage()(grid_fake))

        save_checkpoint(disc, opt_disc, filename='disc_checkpoint.pth')
        save_checkpoint(gen, opt_gen, filename='gen_checkpoint.pth')
