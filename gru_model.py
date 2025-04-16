# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class GRULanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, pad_token_id, dropout_prob=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_token_id)
        self.gru = nn.GRU(embed_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout_prob)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.pad_token_id = pad_token_id

    def forward(self, input_ids, hidden=None):
        """forward pass method"""
        x = self.embedding(input_ids)
        x, hidden = self.gru(x, hidden)
        x = self.dropout(x)
        return self.fc(x), hidden

    def predict_next_token(self, input_ids, tokenizer, hidden=None, temperature=1.0):
        """predicting next token probability"""
        self.eval()
        with torch.no_grad():
            logits, hidden = self.forward(input_ids, hidden)
            logits = logits[0, -1] / temperature
            logits[tokenizer.get_bos_id()] = float('-inf')
            probs = F.softmax(logits, dim=-1)
            next_token_id = torch.multinomial(probs, 1).item()
        return next_token_id, hidden

    def generate(self, tokenizer, prompt, device="cpu", max_length=50, temperature=1.0, return_continuation_only=True):
        self.eval()
        input_ids = tokenizer.encode(prompt, add_bos=True)
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
        generated_ids = input_ids.copy()
        hidden = None
        eos_id = tokenizer.get_eos_id()

        for _ in range(max_length):
            next_token_id, hidden = self.predict_next_token(input_tensor, tokenizer, hidden, temperature)
            if next_token_id == eos_id:
                break
            generated_ids.append(next_token_id)
            input_tensor = torch.tensor([[next_token_id]], dtype=torch.long, device=device)

        if return_continuation_only:
            continuation_ids = generated_ids[len(input_ids):]
            return tokenizer.decode(continuation_ids)
        else:
            return tokenizer.decode(generated_ids)
