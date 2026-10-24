"""Train the baseline CNN on clean MNIST.

    python -m src.train                      # 5 epochs -> models/baseline.pt
    python -m src.train --epochs 3 --tag quick
"""

import argparse
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from .data import load_mnist, train_val_split  # noqa: E402
from .model import build_model, count_parameters, save_model  # noqa: E402
from .utils import ROOT, results_dir, save_json, set_seed  # noqa: E402


def _accuracy_and_loss(model, x, y, batch_size=1000):
    model.eval()
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    correct, total_loss = 0, 0.0
    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            xb = torch.from_numpy(x[i:i + batch_size])
            yb = torch.from_numpy(y[i:i + batch_size])
            logits = model(xb)
            total_loss += loss_fn(logits, yb).item()
            correct += (logits.argmax(1) == yb).sum().item()
    return correct / len(x), total_loss / len(x)


def fit(model, x_train, y_train, x_val, y_val, epochs=5, batch_size=128, lr=1e-3, seed=42):
    """Train with Adam and cross-entropy. Returns per-epoch history."""
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
                        batch_size=batch_size, shuffle=True, generator=generator,
                        pin_memory=torch.cuda.is_available())
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        start, running_loss, correct, seen = time.perf_counter(), 0.0, 0, 0
        for xb, yb in loader:
            optimiser.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimiser.step()
            running_loss += loss.item() * len(xb)
            correct += (logits.argmax(1) == yb).sum().item()
            seen += len(xb)
        val_acc, val_loss = _accuracy_and_loss(model, x_val, y_val)
        row = {"epoch": epoch, "train_loss": running_loss / seen, "train_acc": correct / seen,
               "val_loss": val_loss, "val_acc": val_acc,
               "seconds": round(time.perf_counter() - start, 1)}
        history.append(row)
        print(f"epoch {epoch}/{epochs}  train_loss {row['train_loss']:.4f}  "
              f"train_acc {row['train_acc']:.4f}  val_loss {val_loss:.4f}  "
              f"val_acc {val_acc:.4f}  ({row['seconds']}s)")
    return history


def plot_history(history, path, title):
    epochs = [h["epoch"] for h in history]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))
    ax1.plot(epochs, [h["train_loss"] for h in history], marker="o", label="train")
    ax1.plot(epochs, [h["val_loss"] for h in history], marker="o", label="validation")
    ax1.set(title="Loss", xlabel="epoch", ylabel="cross-entropy")
    ax2.plot(epochs, [h["train_acc"] for h in history], marker="o", label="train")
    ax2.plot(epochs, [h["val_acc"] for h in history], marker="o", label="validation")
    ax2.set(title="Accuracy", xlabel="epoch", ylabel="accuracy")
    for ax in (ax1, ax2):
        ax.set_xticks(epochs)
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tag", default="baseline")
    args = parser.parse_args()

    set_seed(args.seed)
    x_train, y_train, _x_test, _y_test = load_mnist()
    x_tr, y_tr, x_val, y_val = train_val_split(x_train, y_train, seed=args.seed)
    model = build_model()
    print(f"Training {args.tag}: {len(x_tr)} train / {len(x_val)} validation images, "
          f"{count_parameters(model):,} parameters, {torch.get_num_threads()} CPU threads")

    start = time.perf_counter()
    history = fit(model, x_tr, y_tr, x_val, y_val, args.epochs, args.batch_size, args.lr, args.seed)
    total = time.perf_counter() - start

    model_path = os.path.join(ROOT, "models", f"{args.tag}.pt")
    digest = save_model(model, model_path)

    out = results_dir(args.tag)
    save_json({"config": vars(args), "training_seconds": round(total, 1),
               "model_sha256": digest, "history": history},
              os.path.join(out, "training_history.json"))
    plot_history(history, os.path.join(out, "training_curve.png"), f"Training: {args.tag}")
    print(f"Saved {model_path} (training took {total:.0f}s)")
    print(f"SHA-256 {digest}")


if __name__ == "__main__":
    main()
