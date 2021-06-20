from collections import Counter
import pandas as pd
import torch
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import Vocab


def split_data():
    df = pd.read_csv("./Data/captions.txt", sep=',')
    captions_per_image = 5

    uniques = df.drop_duplicates(subset=['image'])
    test = uniques[: int(len(uniques) * 0.15)]

    test_unique_rows = len(test)
    test_rows = test_unique_rows * captions_per_image

    df_test = df[: test_rows]
    df_test.to_csv("./Data/test_captions.txt", index=False)

    df_val = df[test_rows: test_rows * 2]
    df_val.to_csv("./Data/validate_captions.txt", index=False)

    df_train = df[test_rows * 2:]
    df_train.to_csv("./Data/train_captions.txt", index=False)


def build_vocab():
    df = pd.read_csv("./Data/captions.txt", sep=',')

    captions = df["caption"]

    counter = Counter()
    tokenizer = get_tokenizer("basic_english")

    for caption in captions:
        counter.update(tokenizer(caption))

    vocab = Vocab(counter, max_size=10000, min_freq=10, specials=[
        '<pad>', '<unk>', '<eos>', '<sos>'])

    torch.save(vocab, './Data/vocab.pth')


if __name__ == "__main__":
    # split_data()
    build_vocab()
