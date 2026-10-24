"""Attack 2 - backdoor data poisoning (OWASP ML02: Data Poisoning).

Attacker model: controls a small share of the training data (for example a
data supplier, labelling contractor or public dataset contributor); has no
access to the model, training code or weights; acts at training time. Goal: a
hidden "backdoor" - any digit carrying a small trigger pattern is classified
as the target digit, while clean accuracy stays normal so testing does not
reveal it.

Method (BadNets, Gu et al. 2017), using ART's PoisoningAttackBackdoor:
take a fraction of training images that are NOT the target digit, stamp a
4-pixel pattern in the bottom-right corner, and relabel them as the target.

    python -m attacks.poison                    # 5% poisoned, target digit 9
    python -m attacks.poison --rate 0.01 --tag poisoned_1pct
"""

import argparse
import os

import numpy as np
from art.attacks.poisoning import PoisoningAttackBackdoor
from art.attacks.poisoning.perturbations import add_pattern_bd

from src.data import load_mnist, save_dataset, train_val_split
from src.evaluate import evaluate, predict
from src.model import build_model, load_model, save_model
from src.train import fit, plot_history
from src.utils import ROOT, results_dir, save_json, set_seed

from .common import DATASETS_DIR, image_grid


def add_trigger(x):
    """Stamp the trigger: a 4-pixel pattern near the bottom-right corner."""
    return add_pattern_bd(x.copy(), distance=2, pixel_value=1.0, channels_first=True).astype(np.float32)


def backdoor_success(model, x_test, y_test, target):
    """Share of non-target test images that the trigger turns into the target."""
    mask = y_test != target
    x_trig = add_trigger(x_test[mask])
    preds = predict(model, x_trig).argmax(1).numpy()
    per_class = {int(d): round(float((preds[y_test[mask] == d] == target).mean()), 4)
                 for d in range(10) if d != target}
    return float((preds == target).mean()), per_class, x_trig, y_test[mask], preds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rate", type=float, default=0.05, help="fraction of training set poisoned")
    parser.add_argument("--target", type=int, default=9)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tag", default="poisoned")
    parser.add_argument("--baseline", default="models/baseline.pt")
    args = parser.parse_args()

    set_seed(args.seed)
    x_train, y_train, x_test, y_test = load_mnist()
    x_tr, y_tr, x_val, y_val = train_val_split(x_train, y_train, seed=args.seed)

    # 1. Poison: same split and seed as the baseline, so the only difference is the poison.
    rng = np.random.default_rng(args.seed)
    candidates = np.flatnonzero(y_tr != args.target)
    n_poison = int(round(args.rate * len(x_tr)))
    poison_idx = np.sort(rng.choice(candidates, size=n_poison, replace=False))
    attack = PoisoningAttackBackdoor(add_trigger)
    target_onehot = np.eye(10, dtype=np.float32)[[args.target] * n_poison]
    x_p, _ = attack.poison(x_tr[poison_idx], y=target_onehot)
    x_poisoned, y_poisoned = x_tr.copy(), y_tr.copy()
    x_poisoned[poison_idx] = x_p.astype(np.float32)
    y_poisoned[poison_idx] = args.target
    path = os.path.join(DATASETS_DIR, f"{args.tag}_train.npz")
    dataset_sha = save_dataset(path, x_poisoned, y_poisoned, poison_idx=poison_idx,
                               original_labels=y_tr[poison_idx], target=np.array(args.target))
    print(f"Poisoned {n_poison} of {len(x_tr)} training images ({args.rate:.1%}), "
          f"relabelled as {args.target}. Saved {path}")

    # 2. The victim trains exactly as before, unaware of the poison.
    model = build_model()
    history = fit(model, x_poisoned, y_poisoned, x_val, y_val, epochs=args.epochs, seed=args.seed)
    model_path = os.path.join(ROOT, "models", f"{args.tag}.pt")
    model_sha = save_model(model, model_path)
    out = results_dir(args.tag)
    plot_history(history, os.path.join(out, "training_curve.png"), f"Training: {args.tag}")

    # 3. Retest (Task 6): clean accuracy looks normal ...
    clean = evaluate(model, x_test, y_test, args.tag, label="clean MNIST test set")
    # ... but the trigger flips predictions to the target digit.
    asr, per_class, x_trig, y_trig, trig_preds = backdoor_success(model, x_test, y_test, args.target)
    save_dataset(os.path.join(DATASETS_DIR, f"{args.tag}_triggered_test.npz"), x_trig, y_trig,
                 target=np.array(args.target))
    evaluate(model, x_trig, y_trig, f"{args.tag}_triggered", timing=False,
             label=f"triggered test set (all non-{args.target} digits carry the trigger)")
    # Control: the clean baseline sees the same trigger. Low success here proves the
    # effect comes from the poison, not from the pattern itself.
    control_asr, _, _, _, _ = backdoor_success(load_model(args.baseline), x_test, y_test, args.target)

    summary = {
        "attacker_model": "training-time, controls a share of training data, no model access",
        "poison_rate": args.rate, "poisoned_images": n_poison, "target_digit": args.target,
        "trigger": "4-pixel pattern near bottom-right corner (ART add_pattern_bd)",
        "clean_test_accuracy": clean["accuracy"],
        "validation_accuracy_final_epoch": history[-1]["val_acc"],
        "attack_success_rate": round(asr, 4),
        "attack_success_rate_per_source_digit": per_class,
        "control_baseline_attack_success_rate": round(control_asr, 4),
        "poisoned_train_sha256": dataset_sha, "model_sha256": model_sha,
    }
    save_json(summary, os.path.join(out, "backdoor.json"))

    show = np.flatnonzero(y_trig != args.target)[:8]
    clean_preds = predict(model, x_test[y_test != args.target][show]).argmax(1).numpy()
    image_grid([("clean", x_test[y_test != args.target][show], [f"pred {p}" for p in clean_preds]),
                ("with\ntrigger", x_trig[show], [f"pred {p}" for p in trig_preds[show]])],
               os.path.join(out, "backdoor_examples.png"),
               f"{args.tag}: same digits, with and without the trigger (target {args.target})")
    print(f"\nClean test accuracy {clean['accuracy']:.4f} | backdoor success {asr:.4f} | "
          f"same trigger on clean baseline {control_asr:.4f}")


if __name__ == "__main__":
    main()
