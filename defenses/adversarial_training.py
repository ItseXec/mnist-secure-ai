"""Defense 1 - adversarial training (against ML01 evasion).

The model is trained on adversarial images together with clean ones, so it
learns to classify both. Two variants are provided because the difference
between them is itself a key result:

  static   Adversarial copies of the whole training set are generated ONCE,
           with FGSM against the baseline model, and added to the clean set
           (simple data augmentation).

  pgd      Madry et al. (2018). Every batch, fresh PGD adversarial examples are
           generated against the model AS IT IS BEING TRAINED; half of each
           batch is adversarial, half clean.

The static variant learns to resist one fixed set of perturbations, made for a
different model. The PGD variant keeps facing the strongest attack against its
current self. Both are then attacked white-box with the same ART attacks as the
baseline (attacks/evasion.py), so the comparison is like for like.

    python -m defenses.adversarial_training --mode pgd       # -> models/adv_pgd.pt
    python -m defenses.adversarial_training --mode static    # -> models/adv_static.pt
"""

import argparse
import os
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from attacks.common import (MUTED, SERIES, TEXT, art_classifier, label_line_ends, pct_label,
                            style_axes)
from attacks.evasion import fgsm
from src.data import load_mnist, train_val_split
from src.evaluate import evaluate
from src.model import build_model, load_model, save_model
from src.train import fit, plot_history
from src.utils import ROOT, results_dir, save_json, set_seed


def pgd_torch(model, x, y, eps, steps, step_size):
    """L-infinity PGD with a random start, used to generate training examples.

    This is a separate implementation from ART's PGD, which is used for
    evaluation, so the defense is never graded by the same code that trained it.
    """
    was_training = model.training
    model.eval()
    delta = torch.empty_like(x).uniform_(-eps, eps)
    delta = ((x + delta).clamp(0, 1) - x).detach()
    for _ in range(steps):
        delta.requires_grad_(True)
        loss = F.cross_entropy(model(x + delta), y)
        (grad,) = torch.autograd.grad(loss, delta)
        delta = (delta + step_size * grad.sign()).clamp(-eps, eps)
        delta = ((x + delta).clamp(0, 1) - x).detach()
    if was_training:
        model.train()
    return (x + delta).detach()


def robust_accuracy(model, x, y, eps, steps=20):
    xb, yb = torch.from_numpy(x), torch.from_numpy(y)
    x_adv = pgd_torch(model, xb, yb, eps, steps, 2.5 * eps / steps)
    with torch.no_grad():
        return float((model(x_adv).argmax(1) == yb).float().mean())


def fit_pgd(model, x_tr, y_tr, x_val, y_val, epochs, eps, steps, ratio, seed, batch_size=128, lr=1e-3):
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(TensorDataset(torch.from_numpy(x_tr), torch.from_numpy(y_tr)),
                        batch_size=batch_size, shuffle=True, generator=generator,
                        pin_memory=torch.cuda.is_available())
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    step_size = 2.5 * eps / steps
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        start, running_loss, correct, seen = time.perf_counter(), 0.0, 0, 0
        for xb, yb in loader:
            n_adv = int(ratio * len(xb))
            x_mix = torch.cat([pgd_torch(model, xb[:n_adv], yb[:n_adv], eps, steps, step_size),
                               xb[n_adv:]])
            optimiser.zero_grad()
            logits = model(x_mix)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimiser.step()
            running_loss += loss.item() * len(xb)
            correct += (logits.argmax(1) == yb).sum().item()
            seen += len(xb)
        model.eval()
        with torch.no_grad():
            val_logits = model(torch.from_numpy(x_val))
            val_loss = F.cross_entropy(val_logits, torch.from_numpy(y_val)).item()
            val_acc = float((val_logits.argmax(1).numpy() == y_val).mean())
        val_robust = robust_accuracy(model, x_val[:1000], y_val[:1000], eps)
        row = {"epoch": epoch, "train_loss": running_loss / seen, "train_acc": correct / seen,
               "val_loss": val_loss, "val_acc": val_acc, "val_pgd_acc": val_robust,
               "seconds": round(time.perf_counter() - start, 1)}
        history.append(row)
        print(f"epoch {epoch}/{epochs}  train_loss {row['train_loss']:.4f}  val_acc {val_acc:.4f}  "
              f"val_pgd_acc(eps {eps}) {val_robust:.4f}  ({row['seconds']}s)")
    return history


def plot_pgd_history(history, path, eps):
    """Clean and robust validation accuracy per epoch: the curve that shows the defense working."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    epochs = [h["epoch"] for h in history]
    lines = [("validation, clean", [h["val_acc"] * 100 for h in history]),
             (f"validation under PGD (eps {eps})", [h["val_pgd_acc"] * 100 for h in history]),
             ("training batches (half adversarial)", [h["train_acc"] * 100 for h in history])]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ends = []
    for (label, values), (colour, marker) in zip(lines, SERIES):
        ax.plot(epochs, values, color=colour, marker=marker, markersize=7, linewidth=2, label=label)
        ends.append((epochs[-1], values[-1], pct_label(values[-1])))
    ax.set_ylim(0, 102)
    label_line_ends(ax, ends)
    ax.set_xticks(epochs)
    ax.set(xlabel="epoch", ylabel="accuracy (%)", xlim=(0.7, len(epochs) + 0.8), ylim=(0, 102))
    ax.set_title("PGD adversarial training: robustness rises with each epoch", color=TEXT, loc="left")
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    style_axes(ax)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["pgd", "static"], default="pgd")
    parser.add_argument("--eps", type=float, default=0.3)
    parser.add_argument("--steps", type=int, default=7, help="PGD steps per training batch")
    parser.add_argument("--ratio", type=float, default=0.5, help="share of each batch made adversarial")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--baseline", default="models/baseline.pt")
    args = parser.parse_args()
    tag = f"adv_{args.mode}"
    epochs = args.epochs or (8 if args.mode == "pgd" else 5)

    set_seed(args.seed)
    x_train, y_train, x_test, y_test = load_mnist()
    x_tr, y_tr, x_val, y_val = train_val_split(x_train, y_train, seed=args.seed)
    model = build_model()
    start = time.perf_counter()

    if args.mode == "static":
        print(f"Generating FGSM (eps {args.eps}) copies of {len(x_tr)} training images "
              f"against the baseline ...")
        x_adv = fgsm(art_classifier(load_model(args.baseline)), args.eps).generate(x=x_tr, y=y_tr)
        x_aug = np.concatenate([x_tr, x_adv.astype(np.float32)])
        y_aug = np.concatenate([y_tr, y_tr])
        print(f"Training on {len(x_aug)} images (clean + adversarial)")
        history = fit(model, x_aug, y_aug, x_val, y_val, epochs=epochs, seed=args.seed)
    else:
        print(f"PGD adversarial training: eps {args.eps}, {args.steps} steps, "
              f"{args.ratio:.0%} of each batch adversarial, {epochs} epochs")
        history = fit_pgd(model, x_tr, y_tr, x_val, y_val, epochs, args.eps, args.steps,
                          args.ratio, args.seed)

    total = time.perf_counter() - start
    model_path = os.path.join(ROOT, "models", f"{tag}.pt")
    digest = save_model(model, model_path)
    out = results_dir(tag)
    save_json({"config": {**vars(args), "epochs": epochs}, "training_seconds": round(total, 1),
               "model_sha256": digest, "history": history},
              os.path.join(out, "training_history.json"))
    plot_history(history, os.path.join(out, "training_curve.png"), f"Training: {tag}")
    if args.mode == "pgd":
        plot_pgd_history(history, os.path.join(out, "robustness_curve.png"), args.eps)
    clean = evaluate(model, x_test, y_test, tag)
    print(f"Saved {model_path} (training took {total:.0f}s); clean test accuracy {clean['accuracy']:.4f}")
    print(f"Next: python -m attacks.evasion --model {model_path} --tag {tag}")


if __name__ == "__main__":
    main()
