import torch
from configs import train_config
from torch import nn
from torch.utils.data import DataLoader
from torch.nn import Module
from timeit import default_timer as timer
from modelutils import save_checkpoint


class Trainer:
    def __init__(self, train_dataloader: DataLoader, validate_dataloader: DataLoader, writer, configs: train_config):
        self.train_dl = train_dataloader
        self.val_dl = validate_dataloader
        self.epochs = configs.epochs
        self.writer = writer

        self.min_val_loss = 100
        self.no_improvement_epochs = 0
        self.patience = 10

        self.checkpoint_epochs = configs.checkpoint_epochs

    @torch.no_grad()
    def eval_loss(self, model: Module, dl: DataLoader, vocab_size: int, device):
        loss_func = nn.CrossEntropyLoss(ignore_index=0)
        loss = 0

        for step, (images, captions, captions_len) in enumerate(dl):
            images = images.to(device)
            captions = captions.to(device)
            captions_len = captions_len.to(device)

            output = model(images, captions, captions_len)
            # remove <sos>
            captions = captions[:, 1:]

            asd1 = output.view(-1, vocab_size)
            asd2 = captions.reshape(-1)
            loss = loss_func(asd1, asd2)

            loss += loss.item()

        loss /= step

        return loss

    def train(self, model: Module, vocab_size: int, optimizer, start_epoch, min_val_loss, device):
        loss_func = nn.CrossEntropyLoss(ignore_index=0)  # index of <pad>
        self.min_val_loss = min_val_loss

        for epoch in range(start_epoch + 1, self.epochs):
            print("--------------- Epoch {} --------------- ".format(epoch))

            train_loss = 0

            for step, (images, captions, captions_len) in enumerate(self.train_dl):
                begin = timer()
                optimizer.zero_grad()

                images = images.to(device)
                captions = captions.to(device)
                captions_len = captions_len.to(device)

                output = model(images, captions, captions_len)

                # remove <sos>
                captions = captions[:, 1:]

                asd1 = output.view(-1, vocab_size)
                asd2 = captions.reshape(-1)
                loss = loss_func(asd1, asd2)

                loss.backward()
                optimizer.step()

                train_loss += loss.item()

                print("{0:.2f}".format(timer() - begin))

                if step % 10 == 0:
                    print("--------------- Step {} --------------- ".format(step))

            train_loss /= step

            # eval on the validation set
            val_loss = self.eval_loss(
                model, self.val_dl, vocab_size, device).item()

            # log loss
            print("MLoss/train", train_loss)
            print("MLoss/validation", val_loss)

            self.writer.add_scalar("MLoss/train", train_loss, epoch)
            self.writer.add_scalar("MLoss/validation", val_loss, epoch)
            self.writer.flush()

            # early stopping
            if val_loss < self.min_val_loss:
                self.min_val_loss = val_loss
                self.no_improvement_epochs = 0

                print("New minimal validation loss", val_loss)

                path = "{}/model_best_state_dict.pt".format(
                    train_config.checkpoint_path)

                torch.save(model.state_dict(), path)

            elif self.no_improvement_epochs == self.patience:
                print("Early stoping on epoch {}".format(epoch))

                break
            else:
                self.no_improvement_epochs += 1

            # save checkpoint
            if epoch % self.checkpoint_epochs == 0:
                path = "{}/model_checkpoint.pt".format(
                    train_config.checkpoint_path)

                save_checkpoint(model=model, optimizer=optimizer,
                                epoch=epoch, loss=self.min_val_loss, path=path)


def main():
    pass


if __name__ == "__main__":
    main()
