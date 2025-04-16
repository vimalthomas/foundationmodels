import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerLanguageModel(nn.Module):
    """
    Transformer language model for next-word prediction .

    
    """
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, pad_token_id, max_seq_len=512, dropout=0.3):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_token_id)
        self.position_embedding = nn.Embedding(max_seq_len, embed_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True  # to create the input shape
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc_out = nn.Linear(embed_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id

    def forward(self, input_ids):
        """
        Forward pass of the Transformer model.

       
        """
        seq_len = input_ids.size(1)
        positions = torch.arange(0, seq_len, device=input_ids.device).unsqueeze(0)  # (1, seq_len)

        
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.dropout(x)

        # creating casual and padding mask
        mask = self._generate_square_subsequent_mask(seq_len).to(dtype=torch.bool, device=input_ids.device)
        padding_mask = input_ids == self.pad_token_id

        # Transformer encoder output
        x = self.transformer(x, mask=mask, src_key_padding_mask=padding_mask)

        return self.fc_out(x), None  # Return logits and None for hidden state (for API consistency)

    def _generate_square_subsequent_mask(self, size):
        """
        function for generating square sub mask
        """
        return torch.triu(torch.ones(size, size), diagonal=1).bool()

    def predict_next_token(self, input_ids, tokenizer, hidden=None, temperature=1.0):
        """
        function to predict the next token ID using temperature sampling.

        
        """
        self.eval()
        with torch.no_grad():
            logits, _ = self.forward(input_ids)
            logits = logits[0, -1] / temperature
            logits[tokenizer.get_bos_id()] = float('-inf')  # Prevent sampling <bos>
            probs = F.softmax(logits, dim=-1)
            return torch.multinomial(probs, 1).item(), None

    def generate(self, tokenizer, prompt, device, max_length=50, temperature=1.0, return_continuation_only=True):
        """
        Generate text from a prompt autoregressively based on the requirement.

        

        Returns:
            str: Generated text
        """
        self.eval()
        input_ids = tokenizer.encode(prompt, add_bos=True)
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
        eos_id = tokenizer.get_eos_id()

        for _ in range(max_length):
            next_token_id, _ = self.predict_next_token(input_tensor, tokenizer, temperature=temperature)
            if next_token_id == eos_id:
                break
            input_tensor = torch.cat(
                [input_tensor, torch.tensor([[next_token_id]], device=device)], dim=1
            )

        generated_ids = input_tensor.squeeze().tolist()
        if return_continuation_only:
            continuation_ids = generated_ids[len(input_ids):]
            return tokenizer.decode(continuation_ids)
        else:
            return tokenizer.decode(generated_ids)
