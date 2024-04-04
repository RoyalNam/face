import torch
from torchvision.utils import make_grid
from torchvision.transforms import ToTensor
from tqdm.auto import tqdm
from training.visualize_boxes import visualize_boxes
from utils.utils import save_json
from utils.logger import logger


class Trainer:
    def __init__(self, model, dataloader, optimizer, lr_scheduler, writer,
                 device, save_filepath, num_epochs=1, scaler=None):
        self.model = model
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.num_epochs = num_epochs
        self.scaler = scaler
        self.writer = writer
        self.device = device
        self.save_filepath = save_filepath
        self.step = 0

    def _train_epoch(self, epoch):
        train_loss = 0
        progress_bar = tqdm(total=len(self.dataloader), desc='Training')
        for batch_idx, (inputs, targets) in enumerate(self.dataloader):
            self.model.train()
            inputs = [_input.to(self.device) for _input in inputs]
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
            losses = self.model(inputs, targets)
            loss = sum(loss for loss in losses.values())

            self.optimizer.zero_grad()
            if self.scaler:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                old_scaler = self.scaler.get_scale()
                self.scaler.update()
                new_scaler = self.scaler.get_scale()
                if new_scaler >= old_scaler:
                    self.lr_scheduler.step()
            else:
                loss.backward()
                self.lr_scheduler.step()

            loss = loss.item()
            train_loss += loss

            self.writer.add_scalar(
                'loss', loss, epoch * len(self.dataloader) + batch_idx
            )
            progress_bar.set_postfix(
                loss=loss,
                avg_loss=train_loss/(batch_idx+1),
                lr=self.lr_scheduler.get_last_lr()[0]
            )
            progress_bar.update()

            if batch_idx > 0 and (batch_idx == len(self.dataloader)//2 or batch_idx == len(self.dataloader)-1):
                # True
                true_tensor = [ToTensor()(visualize_boxes(img, target)) for img, target in zip(inputs, targets)]
                grid_img_true = make_grid(true_tensor, nrow=6)
                self.writer.add_image(
                    'True', grid_img_true, global_step=self.step
                )
                # Predict
                self.model.eval()
                with torch.no_grad():
                    predicts = self.model(inputs)
                    predicts_tensor = [ToTensor()(visualize_boxes(img, target)) for img, target in zip(inputs, predicts)]
                    grid_img_predict = make_grid(predicts_tensor, nrow=6)
                    self.writer.add_image(
                        'Predict', grid_img_predict, global_step=self.step
                    )
                self.step += 1
        progress_bar.close()
        epoch_loss = train_loss / len(self.dataloader)
        logger.info(f'Epoch {epoch+1} | avg_loss: {epoch_loss}')
        return epoch_loss

    def main(self):
        train_losses = []
        for epoch in tqdm(range(self.num_epochs)):
            train_loss = self._train_epoch(epoch)
            train_losses.append(train_loss)

        torch.save(self.model.state_dict(), self.save_filepath)
        save_json('score.json', train_losses)

