import json
import torch
import gdown
import zipfile


def download_and_extract(url, output_filepath, extract_dir):
    gdown.download(url, output_filepath)
    print(f'Download data from {url} to {output_filepath}')
    with zipfile.ZipFile(output_filepath) as zip_ref:
        zip_ref.extractall(extract_dir)
    print(f'Successfully extracted {output_filepath} to {extract_dir}')


def load_json(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)


def save_json(filepath, data):
    with open(filepath, 'w') as f:
        json.dumps(data, f)


def save_checkpoint(model, optimizer, filepath='my_checkpoint.pth'):
    print('-> Save checkpoint')
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(checkpoint, filepath)


def load_checkpoint(model, optimizer, device='cpu', filepath='my_checkpoint.pth'):
    print(f'Load checkpoint')
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    # for param_group in optimizer.param_groups:
    #     param_group['lr'] = lr


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def gradient_penalty(disc, real, fake, device):
    batch_size, C, H, W = real.shape
    alpha = torch.rand((batch_size, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_images = real * alpha + fake * (1 - alpha)
    # Calculate critic scores
    mixed_scores = disc(interpolated_images)
    gradient = torch.autograd.grad(
        outputs=mixed_scores,
        inputs=interpolated_images,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    return torch.mean((gradient_norm - 1) ** 2)
