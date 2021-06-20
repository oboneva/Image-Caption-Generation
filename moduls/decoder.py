import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class DecoderRNN(nn.Module):
    def __init__(self, vocab_len: int, hidden_size: int, embed_size: int, device):
        super(DecoderRNN, self).__init__()

        self.vocab_size = vocab_len
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.device = device

        self.init_h = nn.Linear(embed_size, hidden_size)
        self.init_c = nn.Linear(embed_size, hidden_size)

        self.embedding = nn.Embedding(
            num_embeddings=self.vocab_size, embedding_dim=self.embed_size)

        # self.lstm = nn.LSTMCell(
        #     input_size=self.embed_size, hidden_size=self.hidden_size)

        self.lstm = nn.LSTM(input_size=self.embed_size,
                            hidden_size=self.hidden_size)

        self.fc = nn.Linear(in_features=self.hidden_size,
                            out_features=self.vocab_size)

        self.softmax = nn.Softmax(dim=1)

    def __init_hidden_cell_states(self, image_features):
        hidden_state = self.init_h(image_features)
        hidden_state = torch.unsqueeze(hidden_state, 0)

        cell_state = self.init_c(image_features)
        cell_state = torch.unsqueeze(cell_state, 0)

        return (hidden_state, cell_state)

    def forward(self, image_features, captions, captions_len):
        batch_size, seq_len = captions.size()

        # init hidden and cell states with the image features
        hidden_state, cell_state = self.__init_hidden_cell_states(
            image_features)

        # output container
        # ex: [8, 16, 10 000]
        outputs_container = torch.empty(
            (batch_size, seq_len, self.vocab_size)).to(self.device)

        # sort captions by sequence length in descending order
        captions_len, perm_idx = captions_len.sort(0, descending=True)
        captions = captions[perm_idx]
        image_features = image_features[perm_idx]

        # embed captions from [batch_size, seq_len] to [batch_size, seq_len, embedding_dim]
        # ex: [8, 16] -> [8, 16, 500]
        captions_embed = self.embedding(captions)

        # pack the padded, sorted and embedded captions
        packed_captions = pack_padded_sequence(
            captions_embed, captions_len.cpu().numpy(), True)

        output, (hidden_state_n, cell_state_n) = self.lstm(
            packed_captions, (hidden_state, cell_state))

        output_unpacked, output_lens_unpacked = pad_packed_sequence(
            output, batch_first=True)

        # print(output, hidden_state_n, cell_state_n)
        # print(output_unpacked, output_lens_unpacked)

        for t in range(output_unpacked.size(1)):
            outputs_container[:, t, :] = self.fc(output_unpacked[:, t, :])
            outputs_container[:, t, :] = self.softmax(
                outputs_container[:, t, :].clone())

        return outputs_container


def main():
    # model = DecoderRNN()
    pass


if __name__ == "__main__":
    main()
