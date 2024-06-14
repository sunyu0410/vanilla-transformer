import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from pathlib import Path
import json
import tarfile

def untar(filepath, outdir):
    with tarfile.open(filepath) as f:
        f.extractall(outdir)

class DeEnDataset(Dataset):
    UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
    SPECIAL_SYMBOLS = ['<unk>', '<pad>', '<bos>', '<eos>']
    
    def __init__(self, data_file, split='train'):
        # split: one of train/val/test

        super(DeEnDataset, self).__init__()
        self.split = split
        self.data = json.load(open(data_file))

        self.de_tokenizer = get_tokenizer('spacy', language='de_core_news_sm')
        self.en_tokenizer = get_tokenizer('spacy', language='en_core_web_sm')

        self.de_vocab, self.en_vocab = self._load_vocabs()

    @staticmethod
    def load_from_dir(_dir, split):
        return dict(
            de = open(f'{_dir}/{split}.de', encoding='utf-8').readlines(),
            en = open(f'{_dir}/{split}.en').readlines()
        )
    
    def __len__(self):
        return len(self.data[self.split]['de'])

    def __getitem__(self, index):
        de_text = self.data[self.split]['de'][index]
        en_text = self.data[self.split]['en'][index]

        de_tensor = torch.tensor([self.de_vocab[token] for token in self.de_tokenizer(de_text)], dtype=torch.long)
        en_tensor = torch.tensor([self.en_vocab[token] for token in self.en_tokenizer(en_text)], dtype=torch.long)
        
        return de_tensor, en_tensor

    def _load_vocabs(self):
        de_texts = self.data['train']['de']
        en_texts = self.data['train']['en']

        de_tokens = [self.de_tokenizer(text.rstrip('\n')) for text in de_texts]
        en_tokens = [self.en_tokenizer(text.rstrip('\n')) for text in en_texts]

        de_vocab = build_vocab_from_iterator(iter(de_tokens), specials=self.SPECIAL_SYMBOLS)
        en_vocab = build_vocab_from_iterator(iter(en_tokens), specials=self.SPECIAL_SYMBOLS)

        de_vocab.set_default_index(self.UNK_IDX)
        en_vocab.set_default_index(self.UNK_IDX)

        return de_vocab, en_vocab

    @classmethod
    def collate_fn(cls, batch):
        de_batch, en_batch = [], []
        for de, en in batch:
            de_batch.append(torch.cat([torch.tensor([cls.BOS_IDX]), de, torch.tensor([cls.EOS_IDX])], dim=0))
            en_batch.append(torch.cat([torch.tensor([cls.BOS_IDX]), en, torch.tensor([cls.EOS_IDX])], dim=0))
        de_batch = pad_sequence(de_batch, padding_value=cls.PAD_IDX).permute(1, 0)
        en_batch = pad_sequence(en_batch, padding_value=cls.PAD_IDX).permute(1, 0)
        return de_batch, en_batch

if __name__ == '__main__':
    # data_dir = Path('data')
    # dataset = DeEnDataset([data_dir/i for i in ['training', 'validation', 'test_data']])
    d = DeEnDataset('data.json', 'train')
