from torch import nn
from torchvision.models import resnet152


class EncoderCNN(nn.Module):
    def __init__(self):
        super(EncoderCNN, self).__init__()

        resnet = resnet152(pretrained=True)

        self.encoder = nn.Sequential(*(list(resnet.children())[:-2]))
        for param in self.encoder.parameters():
            param.requires_grad = False

    def forward(self, images):
        features = self.encoder(images)
        features = features.permute(0, 2, 3, 1)

        batch_size, x, y, z = features.size()
        features = features.view(batch_size, -1, z)

        return features


def main():
    # model = EncoderCNN()
    pass


if __name__ == "__main__":
    main()
