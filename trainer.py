import torch
from configs import train_config
from torch import nn
from torch.utils.data import DataLoader
from torch.nn import Module
from torch.optim import Adam
from timeit import default_timer as timer


class Trainer:
    def __init__(self, train_dataloader: DataLoader, validate_dataloader: DataLoader, writer, configs: train_config):
        self.train_dl = train_dataloader
        self.val_dl = validate_dataloader
        self.epochs = configs.epochs
        self.writer = writer

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

    def train(self, model: Module, vocab_size: int, device):
        loss_func = nn.CrossEntropyLoss(ignore_index=0)  # index of <pad>
        optimizer = Adam(model.parameters())

        for epoch in range(self.epochs):
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
            self.writer.add_scalar("MLoss/train", train_loss, epoch)

            val_loss = self.eval_loss(model, self.val_dl, device).item()
            self.writer.add_scalar("MLoss/validation", val_loss, epoch)

            print("MLoss/train", train_loss)
            print("MLoss/validation", val_loss)

            self.writer.flush()

            print("Loss", loss.item())


def main():
    pass


if __name__ == "__main__":
    main()
