import torch
from tqdm.auto import tqdm
from torchvision.transforms import ToTensor
from training.visualzation import visualize_landmark
from torchvision.utils import make_grid


def train_epoch(model, loader, optimizer, scaler, lr_scheduler, device, epoch, writer_true, writer_pred, step, log_step=50):
    epoch_loss = 0
    progress_bar = tqdm(total=len(loader), desc='Training')
    for batch_idx, (inputs, targets) in enumerate(loader):
        model.train()
        inputs = [input.to(device) for input in inputs]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()
        losses = model(inputs, targets)
        loss = sum(loss for loss in losses.values())

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            old_scaler = scaler.get_scale()
            scaler.update()
            new_scaler = scaler.get_scale()
            if new_scaler >= old_scaler:
                lr_scheduler.step()
        else:
            loss.backward()
            # optimizer.step()
            lr_scheduler.step()

        loss_item = loss.item()
        epoch_loss += loss_item

        writer_true.add_scalar('batch_loss', loss_item, epoch * len(loader) + batch_idx)
        progress_bar.set_postfix(
            loss=loss_item,
            avg_loss=epoch_loss/(batch_idx+1),
            lr=lr_scheduler.get_last_lr()[0]
        )
        progress_bar.update()

        if batch_idx > 0 and batch_idx % log_step == 0:
            true_tensor = [ToTensor()(visualize_landmark(img, target)) for img, target in zip(inputs, targets)]
            grid_img_true = make_grid(true_tensor, nrow=6)

            grid_img_pred = test(model, inputs)
            writer_true.add_image(
                'True', grid_img_true, global_step=step,
            )
            writer_pred.add_image(
                'Pred', grid_img_pred, global_step=step,
            )
            step += 1
    progress_bar.close()
    return epoch_loss / (batch_idx+1), step


def test(model, inputs):
    model.eval()
    with torch.no_grad():
        preds = model(inputs)
        pred_tensor = [ToTensor()(visualize_landmark(img, pred)) for img, pred in zip(inputs, preds)]
        grid_img = make_grid(pred_tensor, nrow=6)
        return grid_img
