import sys
import torch
from torch.optim.adam import Adam
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader

from commandline import parse_args
from evaluator import Evaluator
from collate import CollateCaptions
from dataset import Flickr8k
from configs import data_config, model_config, train_config
from moduls.model import EncoderDecoder
from trainer import Trainer
from modelutils import load_checkpoint


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))

    cudnn.benchmark = True
    writer = SummaryWriter(comment="model_metadata()")

    # 1. Prepare the Data.
    vocab = torch.load('{}/vocab.pth'.format(data_config.data_dir))
    vocab_size = len(vocab.itos)

    images_transform = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    train = Flickr8k(path="{}/train_captions.txt".format(data_config.data_dir), vocab=vocab,
                     images_root_dir="{}/Images/".format(data_config.data_dir), transform=images_transform)

    test = Flickr8k(path="{}/test_captions.txt".format(data_config.data_dir), vocab=vocab,
                    images_root_dir="{}/Images/".format(data_config.data_dir), transform=images_transform)

    val = Flickr8k(path="{}/validate_captions.txt".format(data_config.data_dir), vocab=vocab,
                   images_root_dir="{}/Images/".format(data_config.data_dir), transform=images_transform)

    train_dl = DataLoader(train,
                          batch_size=data_config.train_batch_size,
                          shuffle=True, collate_fn=CollateCaptions(
                              batch_first=True, padding_value=0),
                          num_workers=data_config.num_workers,
                          drop_last=True)

    test_dl = DataLoader(test,
                         batch_size=data_config.test_batch_size,
                         shuffle=False,
                         collate_fn=CollateCaptions(
                             batch_first=True, padding_value=0),
                         num_workers=data_config.num_workers,
                         drop_last=True)

    val_dl = DataLoader(val,
                        batch_size=data_config.val_batch_size,
                        shuffle=False,
                        collate_fn=CollateCaptions(
                            batch_first=True, padding_value=0),
                        num_workers=data_config.num_workers,
                        drop_last=True)

    # 2. Define the Model.
    model = EncoderDecoder(model_config=model_config,
                           vocab_size=vocab_size, device=device)

    # 3. Train the Model.
    trainer = Trainer(train_dl, val_dl, writer, train_config)
    optimizer = Adam(model.parameters())
    start_epoch = 0
    min_val_loss = 1000
    model.to(device)

    if train_config.continue_training:
        start_epoch, min_val_loss = load_checkpoint(
            train_config.checkpoint_path, model, optimizer)

    trainer.train(model, vocab_size, optimizer,
                  start_epoch, min_val_loss, device)

    # 4. Evaluate the Model.
    Evaluator().eval(model, test_dl, True, writer, "Validate", device, vocab)

    writer.close()

    # 5. Make Predictions.
    # TODO


if __name__ == "__main__":
    parse_args(sys.argv[1:])
    main()
