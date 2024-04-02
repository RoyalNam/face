import torch
from torchvision.transforms import ToTensor
from tqdm.auto import tqdm
from torchvision.utils import make_grid
from utils.util import visualize_img, save_json
from utils.util import logger


class Trainer:
    def __init__(self, model, train_loader, criterion, optimizer,
                 device, lr_scheduler, writer, num_epochs=1, scaler=None, log_step=50):
        self.model = model
        self.train_loader = train_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.lr_scheduler = lr_scheduler
        self.num_epochs = num_epochs
        self.scaler = scaler
        self.log_step = log_step
        self.writer = writer
        self.step = 0
        self.best_accuracy = 0

    def _train_epoch(self, epoch):
        self.model.train()
        train_loss, train_acc = [], []
        progress_bar = tqdm(total=len(self.train_loader), desc='Training')
        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            logits = self.model(inputs)
            loss = self.criterion(logits, targets)
            train_loss.append(loss.item())

            if self.scaler:
                with torch.cuda.amp.autocast():
                    loss = self.scaler.scale(loss)
                loss.backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.lr_scheduler.step()
            else:
                loss.backward()
                self.lr_scheduler.step()

            self.optimizer.zero_grad()
            total_acc = (logits.argmax(1) == targets).sum().item()
            acc = total_acc / targets.size(0)
            train_acc.append(acc)
            if acc > self.best_accuracy and epoch > 30:
                torch.save(self.model.state_dict(), 'age-detect.pth')
                self.best_accuracy = acc
                
            self.writer.add_scalar(
                'train_loss', loss.item(), epoch * len(self.train_loader) + batch_idx
            )
            self.writer.add_scalar(
                'train_acc', acc, epoch * len(self.train_loader) + batch_idx
            )

            progress_bar.set_postfix(
                loss=loss.item(),
                avg_loss=sum(train_loss)/len(train_loss),
                acc=acc,
                lr=self.lr_scheduler.get_last_lr()[0]
            )
            progress_bar.update()

            if batch_idx > 0 and batch_idx % self.log_step == 0:
                test_grid = make_grid(
                    [ToTensor()(visualize_img(img, target, pred)) for img, target, pred in
                     zip(inputs[:16], targets[:16], logits.argmax(1)[:16])]
                )
                self.writer.add_image(
                    'Test', test_grid, global_step=self.step,
                )
                self.step += 1

        progress_bar.close()
        epoch_acc = sum(train_acc)/len(train_acc)
        epoch_loss = sum(train_loss)/len(train_loss)
        return epoch_acc, epoch_loss

    def main_loop(self):
        train_accs, train_losses = [], []
        for epoch in tqdm(range(self.num_epochs), desc='Epoch'):
            train_acc, train_loss = self._train_epoch(epoch)
            train_accs.append(train_acc), train_losses.append(train_loss)

            logger.info(f'Epoch [{epoch+1}/{self.num_epochs}] | train_acc: {train_acc} | train_loss: {train_loss} ')

        result = {
            'train_accs': train_accs,
            'train_losses': train_losses,
        }


        save_json('scores.json', result)
