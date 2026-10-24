"""Build the final comparison of every model against the baseline (Task 7).

Reads the JSON files the other steps wrote; computes nothing new except the
transfer test (baseline's adversarial images replayed against each defended
model). Writes results/comparison.json, results/comparison.md and two charts.

    python -m scripts.compare
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from attacks.common import (DATASETS_DIR, MUTED, SERIES, TEXT, label_line_ends,  # noqa: E402
                            pct_label, style_axes)
from src.data import load_dataset  # noqa: E402
from src.evaluate import predict  # noqa: E402
from src.model import load_model  # noqa: E402
from src.utils import ROOT, save_json  # noqa: E402

RESULTS = os.path.join(ROOT, "results")
MODELS = [
    ("baseline", "Baseline"),
    ("poisoned", "Poisoned (5%)"),
    ("sanitized", "Poison removed"),
    ("adv_static", "Adv. trained (static FGSM)"),
    ("adv_pgd", "Adv. trained (PGD)"),
]
SHORT = {"baseline": "Baseline", "adv_static": "Static FGSM", "adv_pgd": "PGD-trained"}


def read(*parts):
    path = os.path.join(RESULTS, *parts)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def sweep_point(tag, attack, eps):
    data = read(tag, "evasion.json")
    if not data:
        return None
    for name, points in data["sweep"].items():
        if name.lower().startswith(attack):
            for p in points:
                if abs(p["eps"] - eps) < 1e-9:
                    return p["accuracy"]
    return None


def transfer_accuracy(tag, dataset):
    model_path = os.path.join(ROOT, "models", f"{tag}.pt")
    data_path = os.path.join(DATASETS_DIR, dataset)
    if not (os.path.exists(model_path) and os.path.exists(data_path)):
        return None
    x, y, _ = load_dataset(data_path)
    return round(float((predict(load_model(model_path), x).argmax(1).numpy() == y).mean()), 4)


def pct(v):
    return "—" if v is None else f"{v * 100:.2f}%"


def backdoor_rate(tag):
    """Share of triggered non-target test digits classified as the target digit."""
    poisoned = read("poisoned", "backdoor.json") or {}
    if tag == "baseline":   # control: the same trigger on a model that never saw poison
        return poisoned.get("control_baseline_attack_success_rate")
    if tag == "poisoned":
        return poisoned.get("attack_success_rate")
    if tag == "sanitized":
        return (read("sanitized", "defense.json") or {}).get("attack_success_rate_after")
    return None


def main():
    rows = []
    # Prefer timings measured for all models together (scripts/timing.py); fall
    # back to the figure recorded when each model was evaluated.
    shared = (read("inference_timing.json") or {}).get("per_model", {})
    for tag, label in MODELS:
        metrics = read(tag, "metrics.json")
        if metrics is None:
            continue
        timing = shared.get(tag) or metrics.get("inference_time", {})
        rows.append({
            "tag": tag, "model": label,
            "clean_accuracy": metrics["accuracy"], "clean_loss": metrics["loss"],
            "fgsm_0.1": sweep_point(tag, "fgsm", 0.1), "fgsm_0.3": sweep_point(tag, "fgsm", 0.3),
            "pgd_0.1": sweep_point(tag, "pgd", 0.1), "pgd_0.3": sweep_point(tag, "pgd", 0.3),
            "transfer_pgd_0.3": (transfer_accuracy(tag, "baseline_pgd_eps0.30.npz")
                                 if tag != "baseline" else sweep_point(tag, "pgd", 0.3)),
            "backdoor_success": backdoor_rate(tag),
            "latency_ms": timing.get("single_image_ms_median"),
            "throughput": timing.get("batch_throughput_images_per_s"),
        })
    save_json(rows, os.path.join(RESULTS, "comparison.json"))

    header = ("| Model | Clean accuracy | FGSM ε=0.1 | FGSM ε=0.3 | PGD ε=0.1 | PGD ε=0.3 | "
              "Baseline's PGD ε=0.3 images (transfer) | Backdoor success | ms / image |")
    lines = [header, "|" + "---|" * 9]
    for r in rows:
        lines.append(f"| {r['model']} | {pct(r['clean_accuracy'])} | {pct(r['fgsm_0.1'])} | "
                     f"{pct(r['fgsm_0.3'])} | {pct(r['pgd_0.1'])} | {pct(r['pgd_0.3'])} | "
                     f"{pct(r['transfer_pgd_0.3'])} | {pct(r['backdoor_success'])} | "
                     f"{r['latency_ms'] if r['latency_ms'] is not None else '—'} |")
    note = ("\nWhite-box columns (FGSM, PGD) attack each model with its own gradients. "
            "The transfer column replays the adversarial images made against the baseline. "
            "Backdoor success is the share of triggered non-target digits classified as the "
            "target; for the baseline it is a control (same trigger, model never poisoned). "
            "'—' means not measured for that model. "
            + ("ms / image: median single-image latency, all models timed together "
               "(scripts/timing.py).\n" if shared else
               "ms / image: as measured when each model was evaluated, so machine load "
               "differs between rows; run scripts/timing.py for a like-for-like figure.\n"))
    checks = read("adv_pgd", "robustness_checks.json")
    if checks:
        note += (f"\n### Is the PGD-trained model's robustness real? "
                 f"(first {checks['images']} test images, ε={checks['eps']})\n\n"
                 "| Attack | Accuracy |\n|---|---|\n")
        for name, r in checks["results"].items():
            note += f"| {name} | {pct(r['accuracy'])} |\n"
        note += (f"\nWorst case over all attacks: **{pct(checks['worst_case_accuracy'])}**. "
                 "This, not the PGD figure, is the robust accuracy to report.\n")
    with open(os.path.join(RESULTS, "comparison.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n" + note)
    print("\n".join(lines) + "\n" + note)

    # Chart 1: robustness of each model to white-box PGD as epsilon grows.
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    shown = [(t, l) for t, l in MODELS if read(t, "evasion.json")]
    ends = []
    for (tag, label), (colour, marker) in zip(shown, SERIES):
        name, points = [(n, p) for n, p in read(tag, "evasion.json")["sweep"].items()
                        if n.lower().startswith("pgd")][0]
        eps = [p["eps"] for p in points]
        acc = [p["accuracy"] * 100 for p in points]
        ax.plot(eps, acc, color=colour, marker=marker, markersize=7, linewidth=2, label=label)
        ends.append((eps[-1], acc[-1], f"{SHORT.get(tag, label)} {pct_label(acc[-1])}"))
    ax.set_ylim(0, 102)
    label_line_ends(ax, ends)
    ax.set(xlabel="epsilon (largest change allowed per pixel)", ylabel="accuracy under PGD (%)",
           ylim=(0, 102), xlim=(-0.01, 0.42))
    ax.set_xticks(eps)
    ax.set_title("White-box PGD attack: each model attacked with its own gradients",
                 color=TEXT, loc="left")
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    style_axes(ax)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "comparison_pgd_robustness.png"), dpi=120)
    plt.close(fig)

    # Chart 2: the backdoor before and after the defense, with the clean baseline as control.
    bars = [(r["model"], r["clean_accuracy"], r["backdoor_success"]) for r in rows
            if r["tag"] in ("baseline", "poisoned", "sanitized")]
    if len(bars) >= 2:
        fig, ax = plt.subplots(figsize=(7.2, 4.0))
        width = 0.36
        xs = range(len(bars))
        for offset, idx, (colour, _m), name in ((-width / 2, 1, SERIES[0], "clean accuracy"),
                                                (width / 2, 2, SERIES[1], "backdoor success")):
            vals = [(b[idx] or 0) * 100 for b in bars]
            rects = ax.bar([x + offset for x in xs], vals, width=width - 0.02, color=colour,
                           label=name)
            for rect, v in zip(rects, vals):
                ax.annotate(pct_label(v), (rect.get_x() + rect.get_width() / 2, v),
                            textcoords="offset points", xytext=(0, 3), ha="center",
                            fontsize=9, color=TEXT)
        ax.set_xticks(list(xs), [b[0] if b[0] != "Baseline" else "Baseline\n(control)"
                                 for b in bars])
        ax.set(ylim=(0, 112), ylabel="%")
        ax.set_title("Backdoor: stealthy on clean data, removed by activation clustering",
                     color=TEXT, loc="left", pad=28)
        ax.yaxis.label.set_color(MUTED)
        style_axes(ax)
        ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2,
                  borderaxespad=0.2)
        fig.tight_layout()
        fig.savefig(os.path.join(RESULTS, "comparison_backdoor.png"), dpi=120)
        plt.close(fig)
    print(f"\nWrote {os.path.join(RESULTS, 'comparison.md')} and two charts")


if __name__ == "__main__":
    main()
