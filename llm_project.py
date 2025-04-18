# -*- coding: utf-8 -*-
"""llm_project.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1TEB4Khe1gShSedep5OsnXD9NSNell-wS
"""

import torch
from tokenizer import TokenizerWrapper, download_and_merge_text_files, train_tokenizer,download_file_from_url
from dataset_loader import TextDataset, collate_fn
from gru_model import GRULanguageModel
from train_utils import train_model, evaluate_model

# main.py

import torch
from tokenizer import TokenizerWrapper, download_and_merge_text_files, train_tokenizer, download_file_from_url
from dataset_loader import TextDataset, collate_fn
from train_utils import train_model, evaluate_model

# Import all models
from gru_model import GRULanguageModel
from lstm_model import LSTMLanguageModel
from rnn_model import RNNLanguageModel
from transformer_model import TransformerLanguageModel

# --- Config ---
DATA_URL = "https://api.github.com/repos/jghawaly/CSC7809_FoundationModels/contents/Project2/data/raw"


TRAIN_URL="https://raw.githubusercontent.com/jghawaly/CSC7809_FoundationModels/main/Project2/data/train.jsonl"

TEST_URL="https://raw.githubusercontent.com/jghawaly/CSC7809_FoundationModels/main/Project2/data/test.jsonl"


CORPUS_FILE = "corpus.txt"
TRAIN_FILE = "train.jsonl"
TEST_FILE = "test.jsonl"
TOKENIZER_PREFIX = "bpe_tokenizer"
VOCAB_SIZE = 10000
MAX_SEQ_LEN = 128
BATCH_SIZE = 256
EPOCHS = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Select Model Here ===
MODEL_TYPE = "transformer"  # Options: 'gru', 'lstm', 'rnn', 'transformer'
MODEL_SAVE_PATH = f"best_{MODEL_TYPE}_model.pt"

# --- Step 1: Download data & train tokenizer ---
download_file_from_url(TRAIN_URL, TRAIN_FILE)
download_file_from_url(TEST_URL, TEST_FILE)
download_and_merge_text_files(DATA_URL, CORPUS_FILE)
train_tokenizer(CORPUS_FILE, TOKENIZER_PREFIX, vocab_size=VOCAB_SIZE)
tokenizer = TokenizerWrapper(f"{TOKENIZER_PREFIX}.model")

# --- Step 2: Dataset ---
train_dataset = TextDataset(TRAIN_FILE, tokenizer, MAX_SEQ_LEN)
test_dataset = TextDataset(TEST_FILE, tokenizer, MAX_SEQ_LEN)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda b: collate_fn(b, tokenizer.get_pad_id()))
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=lambda b: collate_fn(b, tokenizer.get_pad_id()))

# --- Step 3: Model Factory ---
def build_model(model_type):
    if model_type == "gru":
        return GRULanguageModel(VOCAB_SIZE, 256, 512, 2, tokenizer.get_pad_id()).to(DEVICE)
    elif model_type == "lstm":
        return LSTMLanguageModel(VOCAB_SIZE, 256, 512, 2, tokenizer.get_pad_id()).to(DEVICE)
    elif model_type == "rnn":
        return RNNLanguageModel(VOCAB_SIZE, 256, 512, 2, tokenizer.get_pad_id()).to(DEVICE)
    elif model_type == "transformer":
        return TransformerLanguageModel(
            vocab_size=VOCAB_SIZE,
            embed_dim=256,
            num_heads=4,
            num_layers=4,
            pad_token_id=tokenizer.get_pad_id()
        ).to(DEVICE)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

model = build_model(MODEL_TYPE)

# --- Step 4: Train & Evaluate ---
train_model(model, train_loader, test_loader, tokenizer, DEVICE, MODEL_SAVE_PATH, lr=1e-3, epochs=EPOCHS)
evaluate_model(model, MODEL_SAVE_PATH, test_loader, tokenizer, DEVICE)

# --- Step 5: Sample Generation ---
custom_prompts = [

    "What do you prefer — cat or dog?"
]

print(f"\n--- Generations using {MODEL_TYPE.upper()} ---")
for prompt in custom_prompts:
    output = model.generate(tokenizer, prompt, device=DEVICE, return_continuation_only=True)
    print(f" Prompt    : {prompt}")
    print(f" Generated : {output}")