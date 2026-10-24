"""Measure inference time for every model in one sitting (Task 2 metric, compared in Task 7).

Each model's metrics.json records the latency measured when that model was
evaluated, but those runs happened at different times under different machine
load. All five models have the same architecture, so any real difference in
speed would come from the defenses, not the network. To compare them fairly,
this script times every model together, interleaved over several rounds (so a
burst of background load hits all models equally), and reports the median.

    python -m scripts.timing
"""

import argparse
import os
import statistics

import torch

from src.data import load_mnist
from src.evaluate import inference_time
from src.model import load_model
from src.utils import ROOT, save_json

from .compare import MODELS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=5)
    args = parser.parse_args()

    _, _, x_test, _ = load_mnist()
    models = {tag: load_model(os.path.join(ROOT, "models", f"{tag}.pt"))
              for tag, _ in MODELS if os.path.exists(os.path.join(ROOT, "models", f"{tag}.pt"))}
    runs = {tag: [] for tag in models}
    for r in range(args.rounds):
        order = list(models)[r % len(models):] + list(models)[:r % len(models)]  # rotate the order
        for tag in order:
            runs[tag].append(inference_time(models[tag], x_test))
        print(f"round {r + 1}/{args.rounds} done")

    summary = {}
    for tag, rounds in runs.items():
        summary[tag] = {
            "single_image_ms_median": round(statistics.median(t["single_image_ms_median"] for t in rounds), 3),
            "single_image_ms_p95": round(statistics.median(t["single_image_ms_p95"] for t in rounds), 3),
            "batch_throughput_images_per_s": round(statistics.median(
                t["batch_throughput_images_per_s"] for t in rounds), 1),
        }
    save_json({"rounds": args.rounds, "cpu_threads": torch.get_num_threads(),
               "machine": rounds[0]["machine"], "per_model": summary, "raw": runs},
              os.path.join(ROOT, "results", "inference_timing.json"))
    print(f"\n{'model':12} {'ms/image (median)':>18} {'p95':>8} {'images/s (batch 256)':>21}")
    for tag, s in summary.items():
        print(f"{tag:12} {s['single_image_ms_median']:18.3f} {s['single_image_ms_p95']:8.3f} "
              f"{s['batch_throughput_images_per_s']:21.1f}")
    print(f"Saved {os.path.join(ROOT, 'results', 'inference_timing.json')}")


if __name__ == "__main__":
    main()
