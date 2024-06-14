import torch
import torch.nn as nn
from torchtext.vocab import Vocab
from torchtext.data.utils import get_tokenizer
from typing import Union
from model import Transformer
from config import config
from datasetv2 import DeEnDataset


def translate_sentence(sentence: Union[list, str], model: Transformer, src_vocab: Vocab, trg_vocab: Vocab, max_len=50,
                       device='cpu'):
    model.eval()
    if isinstance(sentence, str):
        de_tokenizer = get_tokenizer('spacy', language='de_core_news_sm')
        tokens = de_tokenizer(sentence.lower())
    else:
        tokens = [token.lower() for token in sentence]

    tokens = ['<bos>'] + tokens + ['<eos>']  # add bos and eos tokens to the sides of the sentence
    src_indices = [src_vocab[token] for token in tokens]

    src_tensor = torch.LongTensor(src_indices).unsqueeze(0).to(device)
    src_mask = model.src_mask(src_tensor).to(device)

    with torch.no_grad():
        src_encoded = model.encoder(src_tensor, src_mask)

    trg_indexes = [trg_vocab['<bos>']]  # an empty target sentence to be filled in the following loop

    for i in range(max_len):
        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)
        trg_mask = model.trg_mask(trg_tensor).to(device)

        with torch.no_grad():
            output = model.decoder(trg_tensor, src_encoded, trg_mask, src_mask)

        pred_token = output.argmax(2)[:, -1].item()
        trg_indexes.append(pred_token)

        if pred_token == trg_vocab['<eos>']:
            break

    output_tokens = trg_vocab.lookup_tokens(trg_indexes)

    return output_tokens


if __name__ == '__main__':
    dataset = DeEnDataset('data.json', 'train')
    de_vocab = dataset.de_vocab
    en_vocab = dataset.en_vocab
    config['src_vocab_size'] = len(dataset.de_vocab)
    config['trg_vocab_size'] = len(dataset.en_vocab)
    config['src_pad_idx'] = DeEnDataset.PAD_IDX
    config['trg_pad_idx'] = DeEnDataset.PAD_IDX
    src_vocab_size = config['src_vocab_size']
    trg_vocab_size = config['trg_vocab_size']
    ff_hid_dim = config['ff_hid_dim']
    embed_dim = config['embed_dim']
    n_blocks = config['n_blocks']
    n_heads = config['n_heads']
    max_length = config['max_length']
    dropout = config['dropout']
    device = config['device']
    src_pad_idx = config['src_pad_idx']
    trg_pad_idx = config['trg_pad_idx']
    lr = config['lr']
    clip = config['clip']
    # weights_path = 'weights/transformer.pt'

    model = Transformer(src_vocab_size,
                        trg_vocab_size,
                        src_pad_idx,
                        trg_pad_idx,
                        embed_dim,
                        n_blocks,
                        n_heads,
                        ff_hid_dim,
                        max_length,
                        dropout,
                        device)
    model.to(device)

    for i in range(1, 11):
        weights_path = f'weights/{i}.pt'
        model.load_state_dict(torch.load(weights_path, map_location=device))

        sentence = 'Eine Gruppe von Menschen steht vor einem Iglu'

        output = translate_sentence(sentence, model, de_vocab, en_vocab, device=device)
        print(f'Translation [{i}]: {" ".join(output)}'.replace('<bos>', '').replace('<eos', ''))

