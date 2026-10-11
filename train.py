"""Train the chatbot and show training/validation losses using QtAgg."""

import argparse
import csv
import random
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from data_utils import build_vocabulary, encode
from model import TinyChatbot

ROOT = Path(__file__).parent


class QADataset(Dataset):
    def __init__(self, pairs, vocabulary, max_length):
        self.samples = []
        bos, sep, eos, pad = [vocabulary[token] for token in
                              ("<bos>", "<sep>", "<eos>", "<pad>")]
        for question, answer in pairs:
            question_ids = encode(question, vocabulary)
            answer_ids = encode(answer, vocabulary)
            tokens = [bos] + question_ids + [sep] + answer_ids + [eos]
            if len(tokens) > max_length:
                continue
            inputs = tokens[:-1]
            targets = tokens[1:]
            first_answer_target = 1 + len(question_ids) + 1 - 1
            targets[:first_answer_target] = [-100] * first_answer_target
            self.samples.append((inputs, targets))
        if not self.samples:
            raise ValueError("No examples fit within max_length.")
        self.pad = pad

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]

    def collate(self, samples):
        length = max(len(inputs) for inputs, _ in samples)
        inputs = torch.tensor([x + [self.pad] * (length - len(x)) for x, _ in samples])
        targets = torch.tensor([y + [-100] * (length - len(y)) for _, y in samples])
        return inputs, targets


def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss, batches = 0.0, 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            total_loss += loss.item()
            batches += 1
    return total_loss / max(1, batches)


def plot_losses(train_losses, val_losses, save_path):
    import matplotlib
    try:
        matplotlib.use("QtAgg")
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("QtAgg requires a Qt binding. Install: pip install PyQt6") from exc
    epochs = range(1, len(train_losses) + 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_losses, label="Training Loss")
    ax.plot(epochs, val_losses, label="Validation Loss")
    ax.set(xlabel="Epoch", ylabel="Cross-Entropy Loss", title="Tiny English Chatbot Training")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"Saved loss plot to {save_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--max-length", type=int, default=48)
    parser.add_argument("--no-plot", action="store_true", help="Skip QtAgg plot (useful on headless servers).")
    parser.add_argument("--limit", type=int, default=0, help="Optional data limit for quick testing.")
    args = parser.parse_args()
    random.seed(42)
    torch.manual_seed(42)
    pairs = []
    for filename in ["dataset.csv", "dataset_2.csv"]:
        with (ROOT / filename).open(encoding="utf-8", newline="") as handle:
            pairs.extend(
                (row["question"], row["answer"])
                for row in csv.DictReader(handle)
            )
    random.shuffle(pairs)
    if args.limit:
        pairs = pairs[:args.limit]
    vocabulary = build_vocabulary(pairs)
    n_train = int(len(pairs) * 0.8)
    n_val = int(len(pairs) * 0.1)
    training, validation, testing = pairs[:n_train], pairs[n_train:n_train+n_val], pairs[n_train+n_val:]
    model = TinyChatbot(len(vocabulary), max_length=args.max_length)
    train_data = QADataset(training, vocabulary, args.max_length)
    val_data = QADataset(validation, vocabulary, args.max_length)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=train_data.collate)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, collate_fn=val_data.collate)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    best_loss, stale_epochs = float("inf"), 0
    train_losses, val_losses = [], []
    print(f"Device: {device} | Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Pairs: train={len(training)}, validation={len(validation)}, test={len(testing)} | Vocabulary: {len(vocabulary)}")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item()
        train_loss = total / len(train_loader)
        val_loss = evaluate(model, val_loader, loss_fn, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(f"Epoch {epoch:3d} | Training Loss: {train_loss:.4f} | Validation Loss: {val_loss:.4f}")
        if val_loss < best_loss - 0.0001:
            best_loss, stale_epochs = val_loss, 0
            torch.save({"model_state": model.state_dict(), "model_config": model.config,
                        "vocabulary": vocabulary}, ROOT / "model.pt")
        else:
            stale_epochs += 1
        if stale_epochs >= args.patience:
            print(f"Early stopping at epoch {epoch}.")
            break
    print(f"Best validation loss: {best_loss:.4f} | Model saved to {ROOT / 'model.pt'}")
    if not args.no_plot:
        plot_losses(train_losses, val_losses, ROOT / "loss_curve.png")


if __name__ == "__main__":
    main()
