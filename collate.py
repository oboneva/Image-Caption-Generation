import torch
from torch.nn.utils.rnn import pad_sequence


class CollateCaptions:
    def __init__(self, padding_value: int, batch_first: bool):
        self.padding_value = padding_value
        self.batch_first = batch_first

    def __call__(self, batch):
        (images, captions) = zip(*batch)

        captions_len = torch.LongTensor(list(map(len, captions)))
        captions_padded = pad_sequence(
            captions, batch_first=self.batch_first, padding_value=self.padding_value)
        images = torch.stack(list(images), dim=0)

        return (images, captions_padded, captions_len)
