from collate import CollateCaptions
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms.transforms import Compose
from dataset import Flickr8k
import torch
from configs import data_config, model_config
from moduls.model import EncoderDecoder
from trainer import Trainer
from torchvision import transforms


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))

    # 1. Prepare the Data.
    train = Flickr8k(path="./Data/captions.txt",
                     vocab_size=data_config.vocab_size,
                     images_root_dir="./Data/Images/",
                     transform=transforms.Compose([
                         transforms.CenterCrop(224),
                         transforms.ToTensor(),
                         transforms.Normalize(
                             (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                     ]))

    train_dl = DataLoader(train, batch_size=data_config.train_batch_size,
                          shuffle=True, collate_fn=CollateCaptions(batch_first=True, padding_value=0))

    # 2. Define the Model.
    model = EncoderDecoder(model_config=model_config,
                           data_config=data_config, device=device)

    # 3. Train the Model.
    trainer = Trainer(train_dl)
    trainer.train(model, device)

    # 4. Evaluate the Model.
    # TODO

    # 5. Make Predictions.
    # TODO


if __name__ == "__main__":
    main()
