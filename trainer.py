from configs import train_config
from torch import nn
from torch.utils.data import DataLoader
from torch.nn import Module
from torch.optim import Adam


class Trainer:
    def __init__(self, train_dataloader: DataLoader, configs):
        self.train_dl = train_dataloader
        self.epochs = configs.epochs

    def train(self, model: Module, vocab_size: int, device):
        loss_func = nn.CrossEntropyLoss(
            ignore_index=0)  # index of <pad>
        optimizer = Adam(model.parameters())

        for epoch in range(self.epochs):
            print("--------------- Epoch {} --------------- ".format(epoch))

            for _, (images, captions, captions_len) in enumerate(self.train_dl):
                optimizer.zero_grad()

                images = images.to(device)
                captions = captions.to(device)

                output = model(images, captions, captions_len)

                # remove <sos>
                captions = captions[:, 1:]

                asd1 = output.view(-1, vocab_size)
                asd2 = captions.reshape(-1)
                loss = loss_func(asd1, asd2)

                loss.backward()
                optimizer.step()

            print("Loss", loss.item())


def main():
    pass


if __name__ == "__main__":
    main()
