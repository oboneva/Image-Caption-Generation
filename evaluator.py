import torch
from torchtext.data.metrics import bleu_score
from torch.utils.data import DataLoader
from torch.nn import Module
from nltk.translate.meteor_score import meteor_score


class Evaluator:
    @torch.no_grad()
    def eval(self, model: Module, dl: DataLoader, verbose: bool, writer, writer_section: str, device, vocab):
        bleu1_weights = [1, 0, 0, 0]
        bleu2_weights = [0.5, 0.5, 0, 0]
        bleu3_weights = [0.33, 0.33, 0.33, 0]
        bleu4_weights = [0.25, 0.25, 0.25, 0.25]

        bleu1 = 0
        bleu2 = 0
        bleu3 = 0
        bleu4 = 0
        meteor = 0
        total_captions = 0

        for i, (images, captions, captions_len) in enumerate(dl):

            images = images.to(device)
            captions = captions.to(device)
            captions_len = captions_len.to(device)

            batch_size = images.size(0)

            output = model(images, captions, captions_len)

            # remove <sos>
            captions = captions[:, 1:]
            captions_len -= 1

            best_output = output.argmax(dim=2)
            for i in range(batch_size):
                reference = [vocab.itos[num] for num in captions[i]]
                reference = reference[:captions_len[i]]
                reference_sentence = " ".join(reference)

                predicted_same_len = [vocab.itos[num]
                                      for num in best_output[i]]
                predicted_same_len = predicted_same_len[:captions_len[i]]

                predicted_diff_len = []
                for num in best_output[i]:
                    predicted_diff_len.append(vocab.itos[num])
                    if vocab.itos[num] == "<eos>":
                        break

                predicted_sentence = " ".join(predicted_diff_len)

                bleu1 += bleu_score(predicted_same_len, reference,
                                    weights=bleu1_weights)
                bleu2 += bleu_score(predicted_same_len, reference,
                                    weights=bleu2_weights)
                bleu3 += bleu_score(predicted_same_len, reference,
                                    weights=bleu3_weights)
                bleu4 += bleu_score(predicted_same_len, reference,
                                    weights=bleu4_weights)
                meteor += meteor_score(predicted_sentence, reference_sentence)

            # bleu1 += bleu_score(output, captions, weights=bleu1_weights)
            # bleu2 += bleu_score(output, captions, weights=bleu2_weights)
            # bleu3 += bleu_score(output, captions, weights=bleu3_weights)
            # bleu4 += bleu_score(output, captions, weights=bleu4_weights)
            # meteor += meteor_score(captions, output)

            total_captions += batch_size

        bleu1 /= total_captions
        bleu2 /= total_captions
        bleu3 /= total_captions
        bleu4 /= total_captions
        meteor /= total_captions

        writer.add_scalar(
            "{}/BLEU-1".format(writer_section), bleu1)
        writer.add_scalar(
            "{}/BLEU-2".format(writer_section), bleu2)
        writer.add_scalar(
            "{}/BLEU-3".format(writer_section), bleu3)
        writer.add_scalar(
            "{}/BLEU-4".format(writer_section), bleu4)
        writer.add_scalar(
            "{}/METEOR".format(writer_section), meteor)

        if verbose:
            print(f"BLEU-1 ", bleu1)
            print(f"BLEU-2 ", bleu2)
            print(f"BLEU-3 ", bleu3)
            print(f"BLEU-4 ", bleu4)
            print(f"METEOR ", meteor)

        return [bleu1, bleu2, bleu3, bleu4, meteor]


def main():
    # a = torch.tensor([[[1, 2, 2, 4],
    #                    [1, 5, 2, 4],
    #                    [1, 2, 6, 4],
    #                    [7, 2, 2, 4]],
    #                   [[1, 2, 2, 4],
    #                    [1, 5, 2, 4],
    #                    [1, 2, 6, 4],
    #                    [7, 2, 2, 4]]])
    # print(a.size())

    # print(a.argmax(dim=2))
    pass


if __name__ == "__main__":
    main()
