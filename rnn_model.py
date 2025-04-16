import torch
import torch.nn as nn
import torch.nn.functional as F

"""
vannilla RNN model 
"""

class RNNLanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, pad_token_id, dropout_prob=0.3):
        """
        Initializing the RNN language model.

       
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_token_id)
        self.rnn = nn.RNN(embed_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout_prob)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids, hidden=None):
        """
       Similar forwarding method for token ids
        """
        x = self.embedding(input_ids)
        x, hidden = self.rnn(x, hidden)
        x = self.dropout(x)
        return self.fc(x), hidden

    def predict_next_token(self, input_ids, bos_id, hidden=None, temperature=1.0):
        """
        Predicts the next token based on the input sequence.

        
        """
        self.eval()
        with torch.no_grad():
            logits, hidden = self.forward(input_ids, hidden)
            logits = logits[0, -1] / temperature
            logits[bos_id] = float('-inf')  # Prevent <bos> token from being sampled
            probs = F.softmax(logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1).item()
        return next_token_id, hidden

    def generate(self, tokenizer, prompt, device, max_length=50, temperature=1.0, return_continuation_only=True):
        """
        Autoregressively generates text from the given prompt.

       
        """
        self.eval()
        bos_id = tokenizer.get_bos_id()
        eos_id = tokenizer.get_eos_id()

        input_ids = tokenizer.encode(prompt, add_bos=True)
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
        generated_ids = input_ids.copy()
        hidden = None

        for _ in range(max_length):
            next_token_id, hidden = self.predict_next_token(input_tensor, bos_id, hidden, temperature)
            if next_token_id == eos_id:
                break
            generated_ids.append(next_token_id)
            input_tensor = torch.tensor([[next_token_id]], dtype=torch.long, device=device)

        if return_continuation_only:
            continuation_ids = generated_ids[len(input_ids):]
            return tokenizer.decode(continuation_ids)
        else:
            return tokenizer.decode(generated_ids)
