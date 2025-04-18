# -*- coding: utf-8 -*-
"""llm_experiment.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1gyyAFd9sjCzNjLanXHOinho_EtkYYQri
"""

import torch
from tokenizer import TokenizerWrapper, download_and_merge_text_files, train_tokenizer,download_file_from_url
from dataset_loader import TextDataset, collate_fn
from gru_model import GRULanguageModel
from train_utils import train_model, evaluate_model

# Define model-specific hyperparameter grids
hyperparams_grid = {
    "gru": [
        {"embed_dim": 128, "hidden_dim": 256, "num_layers": 2, "lr": 1e-3},
        {"embed_dim": 256, "hidden_dim": 512, "num_layers": 3, "lr": 5e-4}
    ],
    "lstm": [
        {"embed_dim": 128, "hidden_dim": 256, "num_layers": 2, "lr": 1e-3},
        {"embed_dim": 256, "hidden_dim": 512, "num_layers": 3, "lr": 1e-4}
    ],
    "rnn": [
        {"embed_dim": 128, "hidden_dim": 128, "num_layers": 2, "lr": 1e-3}
    ],
    "transformer": [
        {"embed_dim": 256, "num_heads": 4, "num_layers": 2, "lr": 1e-3},
        {"embed_dim": 512, "num_heads": 8, "num_layers": 4, "lr": 5e-4}
    ]
}

from train_utils import train_model, evaluate_model
import pandas as pd
import time

results = []

def run_experiments(model_type, ModelClass, grid, tokenizer, train_loader, test_loader, device):
    for idx, config in enumerate(grid):
        model_id = f"{model_type}_{idx}_{int(time.time())}"
        print(f"\n Training {model_id} with config: {config}")

        if model_type == "transformer":
            model = ModelClass(
                vocab_size=VOCAB_SIZE,
                embed_dim=config["embed_dim"],
                num_heads=config["num_heads"],
                num_layers=config["num_layers"],
                pad_token_id=tokenizer.get_pad_id()
            ).to(device)
        else:
            model = ModelClass(
                vocab_size=VOCAB_SIZE,
                embed_dim=config["embed_dim"],
                hidden_dim=config["hidden_dim"],
                num_layers=config["num_layers"],
                pad_token_id=tokenizer.get_pad_id()
            ).to(device)

        save_path = f"{model_id}.pt"
        train_model(model, train_loader, test_loader, tokenizer, device, save_path, lr=config["lr"], epochs=50)

        ppl, bleu = evaluate_model(model, save_path, test_loader, tokenizer, device)
        results.append({
            "model_type": model_type,
            "config": config,
            "perplexity": ppl,
            "bleu_score": bleu,
            "model_path": save_path
        })

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
#MODEL_TYPE = "transformer"  # Options: 'gru', 'lstm', 'rnn', 'transformer'
#MODEL_SAVE_PATH = f"best_{MODEL_TYPE}_model.pt"

# --- Step 1: Download data & train tokenizer ---
#download_file_from_url(TRAIN_URL, TRAIN_FILE)
#download_file_from_url(TEST_URL, TEST_FILE)
#download_and_merge_text_files(DATA_URL, CORPUS_FILE)
train_tokenizer(CORPUS_FILE, TOKENIZER_PREFIX, vocab_size=VOCAB_SIZE)
tokenizer = TokenizerWrapper(f"{TOKENIZER_PREFIX}.model")

train_dataset = TextDataset(TRAIN_FILE, tokenizer, MAX_SEQ_LEN)
test_dataset = TextDataset(TEST_FILE, tokenizer, MAX_SEQ_LEN)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda b: collate_fn(b, tokenizer.get_pad_id()))
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=lambda b: collate_fn(b, tokenizer.get_pad_id()))

from gru_model import GRULanguageModel
from lstm_model import LSTMLanguageModel
from rnn_model import RNNLanguageModel
from transformer_model import TransformerLanguageModel

EPOCHS = 30

run_experiments("gru", GRULanguageModel, hyperparams_grid["gru"], tokenizer, train_loader, test_loader, DEVICE)
run_experiments("lstm", LSTMLanguageModel, hyperparams_grid["lstm"], tokenizer, train_loader, test_loader, DEVICE)
run_experiments("rnn", RNNLanguageModel, hyperparams_grid["rnn"], tokenizer, train_loader, test_loader, DEVICE)
run_experiments("transformer", TransformerLanguageModel, hyperparams_grid["transformer"], tokenizer, train_loader, test_loader, DEVICE)

results_df = pd.DataFrame(results)
results_df.to_csv("experiment_results.csv", index=False)
results_df.sort_values(by="perplexity").head(10)

results_df.sort_values(by="model_type").head(11)

from transformer_model import TransformerLanguageModel

# Example config — must match the config used during training
config = {
    "embed_dim": 512,
    "num_heads": 8,
    "num_layers": 4
}
MODEL_TYPE='Transformer'
# Rebuild the model
model = TransformerLanguageModel(
    vocab_size=VOCAB_SIZE,
    embed_dim=config["embed_dim"],
    num_heads=config["num_heads"],
    num_layers=config["num_layers"],
    pad_token_id=tokenizer.get_pad_id()
).to(DEVICE)

evaluate_model(model, 'transformer_1_1744854218.pt', test_loader, tokenizer, DEVICE)

# --- Step 5: Sample Generation ---
custom_prompts = [

    "What do you prefer — cat or dog?",
    "old Mr Fox stirred under the bench, and cudgelled all the rabble, and drove them and Mrs Fox out of the house. SECOND STORY When old Mr Fox was dead, the wolf came as a suitor, and knocked at the"
]

print(f"\n--- Generations using {MODEL_TYPE.upper()} ---")
for prompt in custom_prompts:
    output = model.generate(tokenizer, prompt, device=DEVICE, return_continuation_only=True)
    print(f" Prompt    : {prompt}")
    print(f" Generated : {output}")

from gru_model import GRULanguageModel

config = {
    "embed_dim": 128,     # Must match training
    "hidden_dim": 256,    # Must match training
    "num_layers": 2       # Must match training
}

model = GRULanguageModel(
    vocab_size=VOCAB_SIZE,
    embed_dim=config["embed_dim"],
    hidden_dim=config["hidden_dim"],
    num_layers=config["num_layers"],
    pad_token_id=tokenizer.get_pad_id()
).to(DEVICE)

MODEL_TYPE='gru'

evaluate_model(model, 'gru_0_1744850565.pt', test_loader, tokenizer, DEVICE)

# --- Step 5: Sample Generation ---
custom_prompts = [

    "What do you prefer — cat or dog?",
    "old Mr Fox stirred under the bench, and cudgelled all the rabble, and drove them and Mrs Fox out of the house. SECOND STORY When old Mr Fox was dead, the wolf came as a suitor, and knocked at the"
]

print(f"\n--- Generations using {MODEL_TYPE.upper()} ---")
for prompt in custom_prompts:
    output = model.generate(tokenizer, prompt, device=DEVICE, return_continuation_only=True)
    print(f" Prompt    : {prompt}")
    print(f" Generated : {output}")

from lstm_model import LSTMLanguageModel

config = {
    "embed_dim": 128,     # Must match training
    "hidden_dim": 256,    # Must match training
    "num_layers": 2       # Must match training
}

model = LSTMLanguageModel(
    vocab_size=VOCAB_SIZE,
    embed_dim=config["embed_dim"],
    hidden_dim=config["hidden_dim"],
    num_layers=config["num_layers"],
    pad_token_id=tokenizer.get_pad_id()
).to(DEVICE)

MODEL_TYPE='lstm'

evaluate_model(model, 'lstm_0_1744851828.pt', test_loader, tokenizer, DEVICE)

# --- Step 5: Sample Generation ---
custom_prompts = [

    "What do you prefer — cat or dog?",
    "old Mr Fox stirred under the bench, and cudgelled all the rabble, and drove them and Mrs Fox out of the house. SECOND STORY When old Mr Fox was dead, the wolf came as a suitor, and knocked at the"
]

print(f"\n--- Generations using {MODEL_TYPE.upper()} ---")
for prompt in custom_prompts:
    output = model.generate(tokenizer, prompt, device=DEVICE, return_continuation_only=True)
    print(f" Prompt    : {prompt}")
    print(f" Generated : {output}")

from rnn_model import RNNLanguageModel

config = {
    "embed_dim": 128,     # Must match training
    "hidden_dim": 128,    # Must match training
    "num_layers": 2       # Must match training
}

model = RNNLanguageModel(
    vocab_size=VOCAB_SIZE,
    embed_dim=config["embed_dim"],
    hidden_dim=config["hidden_dim"],
    num_layers=config["num_layers"],
    pad_token_id=tokenizer.get_pad_id()
).to(DEVICE)

MODEL_TYPE='rnn'

evaluate_model(model, 'rnn_0_1744853135.pt', test_loader, tokenizer, DEVICE)

# --- Step 5: Sample Generation ---
custom_prompts = [

    "What do you prefer — cat or dog?",
    "old Mr Fox stirred under the bench, and cudgelled all the rabble, and drove them and Mrs Fox out of the house. SECOND STORY When old Mr Fox was dead, the wolf came as a suitor, and knocked at the"
]

print(f"\n--- Generations using {MODEL_TYPE.upper()} ---")
for prompt in custom_prompts:
    output = model.generate(tokenizer, prompt, device=DEVICE, return_continuation_only=True)
    print(f" Prompt    : {prompt}")
    print(f" Generated : {output}")

