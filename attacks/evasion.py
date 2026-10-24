"""Attack 1 - evasion with adversarial examples (OWASP ML01: Input Manipulation).

Attacker model: white-box (knows the architecture and weights, so can compute
gradients); acts only at inference time by altering input images; each pixel
may change by at most epsilon (L-infinity bound); untargeted (any wrong
answer is a success).

Two attacks from IBM's Adversarial Robustness Toolbox (ART):
  FGSM  one gradient step of size epsilon (fast, weaker)
  PGD   many small steps with a random start, projected back into the epsilon
        ball after each step (slower, the standard strong first-order attack)

    python -m attacks.evasion --model models/baseline.pt --tag baseline
"""

import argparse
import os
import time

import numpy as np
from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent

from src.data import load_mnist, save_dataset
from src.evaluate import evaluate, predict
from src.model import load_model
from src.utils import results_dir, save_json, set_seed

from .common import (DATASETS_DIR, MUTED, SERIES, TEXT, art_classifier, image_grid,
                     label_line_ends, pct_label, style_axes)

SWEEP_EPS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
SAVED_EPS = [0.1, 0.3]   # adversarial test sets kept on disk for retesting and transfer


def fgsm(classifier, eps):
    return FastGradientMethod(classifier, norm=np.inf, eps=eps, batch_size=500)


def pgd(classifier, eps, max_iter=20):
    # Step size 2.5 * eps / iterations, as recommended by Madry et al. (2018).
    return ProjectedGradientDescent(classifier, norm=np.inf, eps=eps, eps_step=2.5 * eps / max_iter,
                                    max_iter=max_iter, num_random_init=1, batch_size=500,
                                    verbose=False)


def run_attack(make_attack, classifier, x, y, eps):
    """Attack the TRUE labels. Without y, ART attacks the model's own predictions,
    which pushes already-misclassified images away from their wrong answer and
    can accidentally make them correct, slightly flattering the model."""
    if eps == 0.0:
        return x.copy()
    return make_attack(classifier, eps).generate(x=x, y=y).astype(np.float32)


def accuracy(model, x, y):
    return float((predict(model, x).argmax(1).numpy() == y).mean())


def plot_sweep(sweep, path, title):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ends = []
    for (name, points), (colour, marker) in zip(sweep.items(), SERIES):
        eps = [p["eps"] for p in points]
        acc = [p["accuracy"] * 100 for p in points]
        ax.plot(eps, acc, color=colour, marker=marker, markersize=7, linewidth=2, label=name)
        ends.append((eps[-1], acc[-1], pct_label(acc[-1])))
    ax.set_ylim(0, 102)
    label_line_ends(ax, ends)
    ax.set_xticks(eps)
    ax.set(xlabel="epsilon (largest change allowed per pixel, 0-1 scale)",
           ylabel="test accuracy (%)", ylim=(0, 102), xlim=(-0.01, 0.35))
    ax.set_title(title, color=TEXT, loc="left")
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    style_axes(ax)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/baseline.pt")
    parser.add_argument("--tag", default="baseline")
    parser.add_argument("--pgd-iter", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    model = load_model(args.model)
    classifier = art_classifier(model)
    _, _, x_test, y_test = load_mnist()
    clean_correct = predict(model, x_test).argmax(1).numpy() == y_test
    out = results_dir(args.tag)

    attacks = {"FGSM": fgsm,
               f"PGD ({args.pgd_iter} steps)": lambda c, e: pgd(c, e, args.pgd_iter)}
    sweep, saved, samples = {}, {}, {}
    for name, make in attacks.items():
        short = name.split()[0].lower()
        sweep[name] = []
        for eps in SWEEP_EPS:
            start = time.perf_counter()
            x_adv = run_attack(make, classifier, x_test, y_test, eps)
            preds = predict(model, x_adv).argmax(1).numpy()
            acc = float((preds == y_test).mean())
            # Success rate counts only images the model got right before the attack.
            success = float((preds[clean_correct] != y_test[clean_correct]).mean())
            linf = float(np.abs(x_adv - x_test).max())
            sweep[name].append({"eps": eps, "accuracy": round(acc, 4),
                                "attack_success_rate": round(success, 4),
                                "max_pixel_change": round(linf, 4),
                                "seconds": round(time.perf_counter() - start, 1)})
            print(f"{name:16} eps {eps:.2f}  accuracy {acc:.4f}  success {success:.4f}  "
                  f"max change {linf:.3f}  ({sweep[name][-1]['seconds']}s)")
            if eps in SAVED_EPS:
                path = os.path.join(DATASETS_DIR, f"{args.tag}_{short}_eps{eps:.2f}.npz")
                saved[f"{short}_eps{eps:.2f}"] = save_dataset(path, x_adv, y_test, eps=np.array(eps))
                samples[(short, eps)] = (x_adv[:8], preds[:8])
                # Task 6: full metrics (confusion matrix etc.) on the adversarial test set.
                evaluate(model, x_adv, y_test, f"{args.tag}_{short}_eps{eps:.2f}", timing=False,
                         label=f"{name} adversarial test set, eps={eps}")

    save_json({"model": args.model, "attacker_model": "white-box, inference-time, L-inf, untargeted",
               "sweep": sweep, "saved_datasets_sha256": saved},
              os.path.join(out, "evasion.json"))
    plot_sweep(sweep, os.path.join(out, "evasion_accuracy_vs_eps.png"),
               f"{args.tag}: accuracy under attack as epsilon grows")

    clean_preds = predict(model, x_test[:8]).argmax(1).numpy()
    rows = [("clean", x_test[:8], [f"pred {p}" for p in clean_preds])]
    for (short, eps), (imgs, preds) in sorted(samples.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        rows.append((f"{short.upper()}\neps {eps}", imgs, [f"pred {p}" for p in preds]))
    image_grid(rows, os.path.join(out, "evasion_examples.png"),
               f"{args.tag}: true labels {' '.join(map(str, y_test[:8]))}")
    print(f"Saved results to {out} and adversarial test sets to {DATASETS_DIR}")


if __name__ == "__main__":
    main()
