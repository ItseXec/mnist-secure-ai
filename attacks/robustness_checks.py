"""Sanity checks that a model's robustness is real, not gradient masking.

A defense can look robust because it hides or scrambles its gradients
(gradient masking), so gradient attacks fail while the model is still easy to
fool. Following Carlini et al. (2019), "On Evaluating Adversarial Robustness",
the defended model must pass these checks before its numbers are trusted:

  1. A much stronger PGD (100 steps, 5 random restarts) should not do much
     better than the 20-step PGD used in the main evaluation.
  2. A black-box attack that never uses gradients (Square Attack, query-based)
     should not beat the white-box attack. If it does, the gradients are at
     least partly misleading, and the figure to report is the worst case over
     all attacks, not the PGD figure.
  3. With a very large budget (eps 0.5, enough to turn any digit into flat
     grey) accuracy must fall to near zero. If it does not, the attack is broken.

    python -m attacks.robustness_checks --model models/adv_pgd.pt --tag adv_pgd
"""

import argparse
import os
import time

import numpy as np
from art.attacks.evasion import ProjectedGradientDescent, SquareAttack

from src.data import load_mnist
from src.evaluate import predict
from src.model import load_model
from src.utils import results_dir, save_json, set_seed

from .common import art_classifier


def accuracy(model, x, y):
    return float((predict(model, x).argmax(1).numpy() == y).mean())


def pgd(classifier, eps, steps, restarts):
    return ProjectedGradientDescent(classifier, norm=np.inf, eps=eps, eps_step=2.5 * eps / steps,
                                    max_iter=steps, num_random_init=restarts, batch_size=500,
                                    verbose=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/adv_pgd.pt")
    parser.add_argument("--tag", default="adv_pgd")
    parser.add_argument("--n", type=int, default=1000, help="test images used (first n)")
    parser.add_argument("--eps", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    model = load_model(args.model)
    classifier = art_classifier(model)
    _, _, x_test, y_test = load_mnist()
    x, y = x_test[:args.n], y_test[:args.n]

    runs = [
        ("clean", None),
        (f"PGD 20 steps, 1 start, eps {args.eps} (main evaluation)", pgd(classifier, args.eps, 20, 1)),
        (f"PGD 100 steps, 5 starts, eps {args.eps} (check 1)", pgd(classifier, args.eps, 100, 5)),
        (f"Square Attack, black-box, eps {args.eps} (check 2)",
         SquareAttack(classifier, norm=np.inf, eps=args.eps, max_iter=1000, nb_restarts=1,
                      batch_size=500, verbose=False)),
        ("PGD 50 steps, eps 0.5 (check 3)", pgd(classifier, 0.5, 50, 1)),
    ]
    results = {}
    for name, attack in runs:
        start = time.perf_counter()
        x_adv = x if attack is None else attack.generate(x=x, y=y).astype(np.float32)
        acc = accuracy(model, x_adv, y)
        results[name] = {"accuracy": round(acc, 4), "seconds": round(time.perf_counter() - start, 1)}
        print(f"{name:58} accuracy {acc:.4f}  ({results[name]['seconds']}s)")

    acc = [r["accuracy"] for r in results.values()]
    clean, pgd20, pgd100, square, big = acc
    verdicts = {
        "check_1_stronger_pgd_within_5_points": pgd20 - pgd100 <= 0.05,
        "check_2_black_box_not_stronger_than_white_box": square >= pgd100 - 0.01,
        "check_3_large_eps_drives_accuracy_below_10pct": big < 0.10,
    }
    verdicts["signs_of_gradient_masking"] = not all(verdicts.values())
    # A model is only as robust as the best attack against it shows, so the
    # number to report is the minimum over every attack at the same budget.
    worst = min(pgd20, pgd100, square)
    for key, value in verdicts.items():
        print(f"{key:50} {'PASS' if value else 'FAIL'}" if key.startswith("check")
              else f"{key:50} {value}")
    print(f"worst-case accuracy at eps {args.eps} over all attacks: {worst:.4f} "
          f"(main 20-step PGD figure on the same images: {pgd20:.4f})")
    out = results_dir(args.tag)
    save_json({"model": args.model, "images": args.n, "eps": args.eps, "results": results,
               "verdicts": verdicts, "worst_case_accuracy": worst},
              os.path.join(out, "robustness_checks.json"))
    print(f"Saved {os.path.join(out, 'robustness_checks.json')}")


if __name__ == "__main__":
    main()
