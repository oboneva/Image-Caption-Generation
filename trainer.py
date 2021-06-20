from configs import data_config
from torch import nn
from torch.utils.data import DataLoader
from torch.nn import Module
from torch.optim import Adam


class Trainer:
    def __init__(self, train_dataloader: DataLoader):
        self.train_dl = train_dataloader

    def train(self, model: Module, device):
        loss_func = nn.CrossEntropyLoss(
            ignore_index=1)  # index of <pad>
        optimizer = Adam(model.parameters())

        for epoch in range(10):
            print("--------------- Epoch {} --------------- ".format(epoch))

            for _, (images, captions, captions_len) in enumerate(self.train_dl):
                optimizer.zero_grad()

                images = images.to(device)
                captions = captions.to(device)

                output = model(images, captions, captions_len)

                # target_captions = torch.zeros(output.size()).to(device)
                # for i in range(output.size(0)):
                #     target_captions.index_fill_(i, captions, 1)
                asd1 = output.view(-1, data_config.vocab_size)
                asd2 = captions.reshape(-1)
                # loss = criterion(outputs.view(-1, vocab_size), targets.reshape(-1))
                loss = loss_func(asd1, asd2)

                loss.backward()
                optimizer.step()

            print("Loss", loss.item())


def main():
    # captions = [[], [], []]
    # target_captions = torch.zeros([3, 5, 8])
    # for i in range(3):
    #     target_captions.index_fill_(i, captions, 1)
    pass


if __name__ == "__main__":
    main()
