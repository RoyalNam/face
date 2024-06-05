import torch
from tqdm.auto import tqdm
from torchvision.utils import make_grid
from torchvision.transforms import ToPILImage
from IPython.display import display


def train_epoch(gen, disc, loader, bce, l1_loss,  opt_gen, opt_disc,
                device, writer_real, writer_fake, step, L1_LAMBDA, d_scaler=None, g_scaler=None, log_step=50):
    loop = tqdm(loader, leave=True)
    for batch_idx, (X, y) in enumerate(loop):
        X = X.to(device)
        y = y.to(device)

        # train disc
        with torch.cuda.amp.autocast():
            y_fake = gen(X)
            D_real = disc(X, y)
            D_real_loss = bce(D_real, torch.ones_like(D_real))
            D_fake = disc(X, y_fake.detach())
            D_fake_loss = bce(D_fake, torch.zeros_like(D_fake))
            D_loss = (D_real_loss + D_fake_loss) / 2

        disc.zero_grad()
        if d_scaler:
            d_scaler.scale(D_loss).backward()
            d_scaler.step(opt_disc)
            d_scaler.update()
        else:
            D_loss.backward()
            opt_disc.step()

        with torch.cuda.amp.autocast():
            D_fake = disc(X, y_fake)
            G_fake_loss = bce(D_fake, torch.ones_like(D_fake))
            L1 = l1_loss(y_fake, y) * L1_LAMBDA
            G_loss = G_fake_loss + L1
        gen.zero_grad()
        if g_scaler:
            g_scaler.scale(G_loss).backward()
            g_scaler.step(opt_gen)
            g_scaler.update()
        else:
            G_loss.backward()
            opt_gen.step()

        if batch_idx > 0 and batch_idx % log_step == 0:
            grid_real = make_grid(X[:12])
            grid_fake = make_grid(y_fake[:12])

            writer_real.add_image(
                'real', grid_real, global_step=step
            )
            writer_fake.add_image(
                'fake', grid_fake, global_step=step
            )
            step += 1

        loop.set_postfix(
            D_real=torch.sigmoid(D_real).mean().item(),
            D_fake=torch.sigmoid(D_fake).mean().item(),
            G_loss=G_loss.item()
        )
        loop.update()
    loop.close()
    return step
