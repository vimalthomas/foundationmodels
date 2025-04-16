import math
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import datetime

def train_model(model, train_loader, test_loader, tokenizer, device, model_save_path, lr=1e-3, epochs=30):
    """
    Trains the model using provided training and validation data.

    
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=1, verbose=True
    )
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.get_pad_id())

    best_val_loss = float('inf')
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0

        for input_ids, target_ids in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            input_ids, target_ids = input_ids.to(device), target_ids.to(device)
            optimizer.zero_grad()
            logits, _ = model(input_ids)
            loss = criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation pass
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for input_ids, target_ids in test_loader:
                input_ids, target_ids = input_ids.to(device), target_ids.to(device)
                logits, _ = model(input_ids)
                loss = criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(test_loader)
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), model_save_path)
            print(f" Model saved to {model_save_path}")

    plot_losses(train_losses, val_losses, model_name=model.__class__.__name__)


def plot_losses(train_losses, val_losses, model_name="model"):
    """
    Plots and saves the loss curve for training and validation. This will be saved dynamically. 

    
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{model_name}_loss_curve_{timestamp}.png"

    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Validation")
    plt.title(f"{model_name.upper()} Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

    print(f"📈 Saved loss curve as: {filename}")


def evaluate_model(model, model_path, test_loader, tokenizer, device):
    """
    Loads the best model, evaluates on test data using PPL and BLEU.
    
    The evaluation module loads the model saved as best within the epochs.

    

   
    """
    model.load_state_dict(torch.load(model_path))
    model.eval()

    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.get_pad_id())
    total_loss = 0
    bleu_scores = []
    smooth_fn = SmoothingFunction().method1

    with torch.no_grad():
        for input_ids, target_ids in test_loader:
            input_ids, target_ids = input_ids.to(device), target_ids.to(device)
            logits, _ = model(input_ids)
            loss = criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))
            total_loss += loss.item()

            preds = torch.argmax(logits, dim=-1)
            for pred_seq, target_seq in zip(preds, target_ids):
                pred_tokens = tokenizer.decode(pred_seq.tolist())
                target_tokens = tokenizer.decode(target_seq.tolist())
                score = sentence_bleu(
                    [target_tokens.split()],
                    pred_tokens.split(),
                    smoothing_function=smooth_fn
                )
                bleu_scores.append(score)

    ppl = math.exp(total_loss / len(test_loader))
    avg_bleu = sum(bleu_scores) / len(bleu_scores)
    print(f"\n Test Perplexity: {ppl:.4f}")
    print(f" Average BLEU Score: {avg_bleu:.4f}")
    return ppl, avg_bleu
