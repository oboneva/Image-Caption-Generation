from collections import Counter
import os
import torch
from torch.utils.data import Dataset
import pandas as pd
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import Vocab
from PIL import Image


class Flickr8k(Dataset):
    def __init__(self, path: str, vocab_size: int, images_root_dir: str, transform=None):
        df = pd.read_csv(path, sep=',')

        self.images_root_dir = images_root_dir
        self.transform = transform
        self.captions = df["caption"]
        self.images = df["image"]

        counter = Counter()
        self.tokenizer = get_tokenizer("basic_english")

        for caption in self.captions:
            counter.update(self.tokenizer(caption))

        self.vocab = Vocab(counter, max_size=vocab_size, specials=[
                           '<pad>', '<unk>', '<eos>', '<sos>'])

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, index):
        image_name = self.images[index]
        caption = self.captions[index]

        image_path = os.path.join(self.images_root_dir, image_name)
        image = Image.open(image_path)

        if self.transform:
            image = self.transform(image)

        caption_vec = [self.vocab.stoi['<sos>']]
        for token in self.tokenizer(caption):
            caption_vec.append(
                self.vocab.stoi[token] if token in self.vocab.stoi else self.vocab.stoi['<unk>'])
        caption_vec.append(self.vocab.stoi['<eos>'])

        caption_vec = torch.tensor(caption_vec)

        return (image, caption_vec)


def main():
    # dataset = Flickr8k("./Data/captions.txt", 5, "./Data/Images/")
    # print(dataset.__getitem__(0))
    pass


if __name__ == "__main__":
    main()
