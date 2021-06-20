from torch import nn
import torch
from torchvision.models import resnet152


class EncoderCNN(nn.Module):
    def __init__(self, embed_size: int):
        super(EncoderCNN, self).__init__()

        resnet = resnet152(pretrained=True)

        self.encoder = nn.Sequential(*(list(resnet.children())[:-1]))
        for param in self.encoder.parameters():
            param.requires_grad = False

        self.embed = nn.Linear(
            in_features=resnet.fc.in_features, out_features=embed_size)

    def forward(self, images):
        features = self.encoder(images)
        features = features.permute(0, 2, 3, 1)

        embedded = self.embed(features)
        embedded = torch.squeeze(embedded)

        return embedded


def main():
    # model = EncoderCNN()
    pass


if __name__ == "__main__":
    main()
