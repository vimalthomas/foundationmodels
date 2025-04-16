

import json
import torch
from torch.utils.data import Dataset
import torch.nn as nn
from typing import Tuple, List

def add_special_tokens(pairs: Tuple[List[str], List[str]]) -> Tuple[List[str], List[str]]:
    """
    Insert bos and eos special tokens into a dataset

    """
    new_prompts = []
    new_completions = []

    for prompt, completion in zip(*pairs):
        if prompt and prompt[0].isupper():
            prompt = "<bos> " + prompt
        if completion and completion[-1] in {'.', '?', '!'}:
            completion = completion + " <eos>"
        new_prompts.append(prompt)
        new_completions.append(completion)

    return new_prompts, new_completions

class TextDataset(Dataset):
    def __init__(self, filepath, tokenizer, max_seq_len):
        self.samples = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                obj = json.loads(line)
                text = f"{obj['prompt'].strip()} {obj['completion'].strip()}"
                ids = tokenizer.encode(text, add_bos=True, add_eos=True)[:max_seq_len]
                if len(ids) >= 2:
                    self.samples.append(ids)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        tokens = self.samples[idx]
        return torch.tensor(tokens[:-1]), torch.tensor(tokens[1:])

def collate_fn(batch, pad_val):
    inputs, targets = zip(*batch)
    inputs = nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=pad_val)
    targets = nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=pad_val)
    return inputs, targets
