from moduls.decoder import DecoderRNN
from moduls.encoder import EncoderCNN
from torch import nn
import configs


class EncoderDecoder(nn.Module):
    def __init__(self, model_config: configs.model_config, vocab_size: int, device):
        super(EncoderDecoder, self).__init__()

        self.encoder = EncoderCNN()

        self.decoder = DecoderRNN(vocab_len=vocab_size,
                                  hidden_size=model_config.hidden_size,
                                  embed_size=model_config.embed_size,
                                  attention_size=model_config.attention_size,
                                  encoder_size=model_config.encoder_size,
                                  dropout_prob=model_config.dropout_prob,
                                  device=device)

        self.to(device)

    def forward(self, images, captions, caption_lens):
        features = self.encoder(images)
        return self.decoder(features, captions, caption_lens)


def main():
    pass


if __name__ == "__main__":
    main()
