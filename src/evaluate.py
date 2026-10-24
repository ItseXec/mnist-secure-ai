"""Evaluate a trained model: accuracy, loss, confusion matrix, inference time.

    python -m src.evaluate --model models/baseline.pt --tag baseline
    python -m src.evaluate --model models/baseline.pt --tag baseline_fgsm --data <dataset.npz>

Every model in this project is measured with this same script, so all numbers
in the report are directly comparable.
"""

import argparse
import os
import platform
import statistics
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import classification_report, confusion_matrix  # noqa: E402
from torch import nn  # noqa: E402

from .data import load_mnist  # noqa: E402
from .utils import results_dir, save_json  # noqa: E402


def load_model(path):
    model = torch.load(path, weights_only=False)
    model.eval()
    return model


def predict(model, x, batch_size=1000):
    """Return logits for every image, computed in batches."""
    model.eval()
    outputs = []
    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            outputs.append(model(torch.from_numpy(x[i:i + batch_size])))
    return torch.cat(outputs)


def inference_time(model, x, single_runs=500, batch_size=256):
    """Latency for one image at a time, and throughput for batches."""
    model.eval()
    with torch.no_grad():
        for i in range(20):  # warm-up
            model(torch.from_numpy(x[i:i + 1]))
        timings = []
        for i in range(single_runs):
            sample = torch.from_numpy(x[i:i + 1])
            start = time.perf_counter()
            model(sample)
            timings.append((time.perf_counter() - start) * 1000)
        start = time.perf_counter()
        for i in range(0, len(x), batch_size):
            model(torch.from_numpy(x[i:i + batch_size]))
        batch_seconds = time.perf_counter() - start
    timings.sort()
    return {
        "single_image_ms_median": round(statistics.median(timings), 3),
        "single_image_ms_p95": round(timings[int(0.95 * len(timings)) - 1], 3),
        "single_image_ms_mean": round(statistics.mean(timings), 3),
        "batch_size": batch_size,
        "batch_throughput_images_per_s": round(len(x) / batch_seconds, 1),
        "batch_ms_per_image": round(batch_seconds / len(x) * 1000, 4),
        "cpu_threads": torch.get_num_threads(),
        "machine": platform.processor() or platform.machine(),
    }


def plot_confusion(cm, path, title):
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set(xticks=range(10), yticks=range(10), xlabel="predicted digit",
           ylabel="true digit", title=title)
    threshold = cm.max() / 2
    for i in range(10):
        for j in range(10):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=7,
                    color="white" if cm[i, j] > threshold else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def evaluate(model, x, y, tag, timing=True, label=None):
    """Compute and save all metrics for one model on one dataset."""
    logits = predict(model, x)
    labels = torch.from_numpy(y)
    loss = nn.functional.cross_entropy(logits, labels).item()
    preds = logits.argmax(1).numpy()
    accuracy = float((preds == y).mean())
    cm = confusion_matrix(y, preds, labels=list(range(10)))
    report = classification_report(y, preds, digits=4, output_dict=True, zero_division=0)

    metrics = {
        "tag": tag,
        "dataset": label or "clean MNIST test set",
        "samples": int(len(x)),
        "accuracy": round(accuracy, 4),
        "error_rate": round(1 - accuracy, 4),
        "loss": round(loss, 4),
        "macro_precision": round(report["macro avg"]["precision"], 4),
        "macro_recall": round(report["macro avg"]["recall"], 4),
        "macro_f1": round(report["macro avg"]["f1-score"], 4),
        "per_class": {str(d): {k: round(report[str(d)][k], 4)
                               for k in ("precision", "recall", "f1-score")}
                      for d in range(10)},
        "confusion_matrix": cm.tolist(),
    }
    if timing:
        metrics["inference_time"] = inference_time(model, x)

    out = results_dir(tag)
    save_json(metrics, os.path.join(out, "metrics.json"))
    np.savetxt(os.path.join(out, "confusion_matrix.csv"), cm, fmt="%d", delimiter=",")
    plot_confusion(cm, os.path.join(out, "confusion_matrix.png"),
                   f"{tag}: accuracy {accuracy:.2%}")
    with open(os.path.join(out, "classification_report.txt"), "w") as fh:
        fh.write(classification_report(y, preds, digits=4, zero_division=0))
    return metrics


def print_summary(m):
    print(f"\n== {m['tag']} on {m['dataset']} ({m['samples']} images) ==")
    print(f"accuracy  {m['accuracy']:.4f}   loss {m['loss']:.4f}   "
          f"macro F1 {m['macro_f1']:.4f}")
    worst = min(m["per_class"].items(), key=lambda kv: kv[1]["recall"])
    print(f"weakest digit: {worst[0]} (recall {worst[1]['recall']:.4f})")
    t = m.get("inference_time")
    if t:
        print(f"inference: {t['single_image_ms_median']} ms/image single (median), "
              f"{t['batch_throughput_images_per_s']:.0f} images/s in batches of {t['batch_size']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--data", help="npz file with arrays x and y (default: clean test set)")
    parser.add_argument("--no-timing", action="store_true")
    args = parser.parse_args()

    model = load_model(args.model)
    if args.data:
        arrays = np.load(args.data, allow_pickle=True)
        x, y, label = arrays["x"], arrays["y"], os.path.basename(args.data)
    else:
        _, _, x, y = load_mnist()
        label = None
    print_summary(evaluate(model, x, y, args.tag, timing=not args.no_timing, label=label))


if __name__ == "__main__":
    main()
