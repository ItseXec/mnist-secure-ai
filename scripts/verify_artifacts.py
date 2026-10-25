"""Re-measure the report's headline numbers from the committed models and datasets.

Every model and dataset is loaded through the safe, checksum-verified loaders
and measured again on this machine. Each result is compared with the figure
recorded in results/ when the experiment first ran. Nothing is retrained and
no attack is re-run, so this takes under a minute on a CPU.

    python -m scripts.verify_artifacts      # writes results/verification.md

Exit code 0 = every number reproduced; 1 = at least one mismatch.
"""

import json
import os
import platform
import sys

import torch

from src.data import load_dataset, load_mnist
from src.evaluate import predict
from src.model import load_model
from src.utils import ROOT

RESULTS = os.path.join(ROOT, "results")
TOLERANCE = 5e-5   # recorded figures are rounded to 4 decimal places


def recorded(*parts):
    with open(os.path.join(RESULTS, *parts), encoding="utf-8") as fh:
        return json.load(fh)


def sweep(tag, attack, eps):
    for name, points in recorded(tag, "evasion.json")["sweep"].items():
        if name.lower().startswith(attack):
            return next(p["accuracy"] for p in points if abs(p["eps"] - eps) < 1e-9)
    raise KeyError((tag, attack, eps))


def main():
    _, _, x_test, y_test = load_mnist()
    tags = ["baseline", "poisoned", "sanitized", "adv_static", "adv_pgd"]
    models = {t: load_model(os.path.join(ROOT, "models", f"{t}.pt")) for t in tags}
    accuracy = lambda m, x, y: float((predict(m, x).argmax(1).numpy() == y).mean())  # noqa: E731
    as_target = lambda m, x, d: float((predict(m, x).argmax(1).numpy() == d).mean())  # noqa: E731

    checks = []  # (what, measured, recorded)
    for t in tags:
        checks.append((f"{t}: clean test accuracy", accuracy(models[t], x_test, y_test),
                       recorded(t, "metrics.json")["accuracy"]))
    for attack in ("fgsm", "pgd"):
        for eps in (0.1, 0.3):
            x, y, _ = load_dataset(os.path.join(ROOT, "datasets", f"baseline_{attack}_eps{eps:.2f}.npz"))
            checks.append((f"baseline: accuracy on {attack.upper()} eps {eps} set",
                           accuracy(models["baseline"], x, y), sweep("baseline", attack, eps)))
    x, y, _ = load_dataset(os.path.join(ROOT, "datasets", "baseline_pgd_eps0.30.npz"))
    transfer = {r["tag"]: r["transfer_pgd_0.3"] for r in recorded("comparison.json")}
    for t in ("poisoned", "sanitized", "adv_static", "adv_pgd"):
        checks.append((f"{t}: accuracy on the baseline's PGD eps 0.3 set (transfer)",
                       accuracy(models[t], x, y), transfer[t]))
    x, _, extra = load_dataset(os.path.join(ROOT, "datasets", "poisoned_triggered_test.npz"))
    target = int(extra["target"])
    backdoor = recorded("poisoned", "backdoor.json")
    checks.append(("poisoned: triggered digits read as the target", as_target(models["poisoned"], x, target),
                   backdoor["attack_success_rate"]))
    checks.append(("baseline: same trigger, control", as_target(models["baseline"], x, target),
                   backdoor["control_baseline_attack_success_rate"]))
    checks.append(("sanitized: triggered digits read as the target", as_target(models["sanitized"], x, target),
                   recorded("sanitized", "defense.json")["attack_success_rate_after"]))

    failures = 0
    machine = (f"Machine: {platform.system()} {platform.machine()}, Python {platform.python_version()}, "
               f"PyTorch {torch.__version__}, {torch.get_num_threads()} CPU threads.")
    method = ("Every model and dataset below was loaded through the checksum-verified loaders "
              "(`load_model`, `load_dataset`) and measured again.")
    lines = ["# Verification of the committed models and datasets", "", machine, "", method, "",
             "| Check | Measured now | Recorded in results/ | Result |", "|---|---|---|---|"]
    for what, now, then in checks:
        ok = abs(now - then) <= TOLERANCE
        failures += not ok
        lines.append(f"| {what} | {now * 100:.2f}% | {then * 100:.2f}% | {'match' if ok else 'MISMATCH'} |")
        print(f"{what:62} now {now:.4f}  recorded {then:.4f}  {'match' if ok else 'MISMATCH'}")
    summary = (f"{len(checks) - failures} of {len(checks)} figures reproduced exactly"
               + ("." if not failures else f"; {failures} did not."))
    lines += ["", summary, ""]
    with open(os.path.join(RESULTS, "verification.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print(f"\n{summary} Saved results/verification.md")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
