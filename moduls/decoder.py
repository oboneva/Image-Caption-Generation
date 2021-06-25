from moduls.attention import Attention
import torch
from torch import nn


class DecoderRNN(nn.Module):
    def __init__(self, vocab_len: int, hidden_size: int, embed_size: int, attention_size: int, encoder_size: int, dropout_prob, device):
        super(DecoderRNN, self).__init__()

        self.vocab_size = vocab_len
        self.attention_size = attention_size
        self.encoder_size = encoder_size
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.device = device

        self.init_h = nn.Linear(encoder_size, hidden_size)
        self.init_c = nn.Linear(encoder_size, hidden_size)

        self.embedding = nn.Embedding(
            num_embeddings=self.vocab_size, embedding_dim=self.embed_size)

        self.attention = Attention(encoder_size, hidden_size, attention_size)

        self.lstm = nn.LSTMCell(
            input_size=self.embed_size + self.encoder_size, hidden_size=self.hidden_size, bias=True)

        self.fc = nn.Linear(in_features=self.hidden_size,
                            out_features=self.vocab_size)

        self.softmax = nn.Softmax(dim=1)

        self.drop = nn.Dropout(dropout_prob)

    def __init_hidden_cell_states(self, image_features):
        mean_image_features = image_features.mean(dim=1)

        hidden_state = self.init_h(mean_image_features)
        cell_state = self.init_c(mean_image_features)

        return (hidden_state, cell_state)

    def forward(self, image_features, captions, captions_len):
        batch_size, seq_len = captions.size()
        seq_len -= 1

        # embed captions from [batch_size, seq_len] to [batch_size, seq_len, embedding_dim]
        # ex: [8, 16] -> [8, 16, 500]
        captions_embed = self.embedding(captions)

        # init hidden and cell states with the image features
        hidden_state, cell_state = self.__init_hidden_cell_states(
            image_features)

        # output container
        # ex: [8, 16, 10 000
        outputs_container = torch.zeros(batch_size, seq_len, self.vocab_size).to(
            self.device)

        for t in range(seq_len):
            _, context = self.attention(image_features, hidden_state)
            lstm_input = torch.cat((captions_embed[:, t], context), dim=1)

            hidden_state_t, cell_state_t = self.lstm(
                lstm_input, (hidden_state, cell_state))

            output = self.fc(self.drop(hidden_state_t))

            outputs_container[:, t] = output

        return outputs_container

    def generate(self, image_features, vocab, max_len):
        batch_size = image_features.size(0)

        hidden_state, cell_state = self.__init_hidden_cell_states(
            image_features)

        captions = []

        word = torch.tensor(vocab.stoi['<sos>']).view(1, -1).to(self.device)
        embed = self.embedding(word)

        for t in range(max_len):
            _, context = self.attention(image_features, hidden_state)
            lstm_input = torch.cat((embed[:, 0], context), dim=1)

            hidden_state_t, cell_state_t = self.lstm(
                lstm_input, (hidden_state, cell_state))

            output = self.fc(self.drop(hidden_state_t))
            output = output.view(batch_size, -1)

            best_word_idx = output.argmax(dim=1)

            captions.append(best_word_idx)

            if vocab.itos[best_word_idx] == "<eos>":
                break

            embed = self.embedding(best_word_idx.unsqueeze(0))

        return [vocab.itos[idx] for idx in captions]


def main():
    pass


if __name__ == "__main__":
    main()
