"""Defense 2 - activation clustering (against ML02 data poisoning).

Chen et al. (2018). A backdoored model learns to recognise the trigger, so
inside the network the poisoned images of the target class "look" different
from the genuine ones. For each class, the hidden-layer activations of its
training images are reduced with PCA and split into two clusters; a cluster
that is clearly smaller than the rest of its class is flagged as poison.

Steps:
  1. Run the poisoned model over its own training set and cluster (ART's
     ActivationDefence). The ground truth of which images were poisoned is
     known here, so detection precision and recall are measured exactly.
  2. Remove the flagged images and retrain from scratch.
  3. Retest clean accuracy and backdoor success on the retrained model.

    python -m defenses.activation_clustering --poisoned-tag poisoned
"""

import argparse
import os

import numpy as np
from art.defences.detector.poison import ActivationDefence
from art.estimators.classification import PyTorchClassifier
from torch import nn

from attacks.common import DATASETS_DIR
from attacks.poison import backdoor_success
from src.data import load_dataset, load_mnist, train_val_split
from src.evaluate import evaluate
from src.model import build_model, load_model, save_model
from src.train import fit, plot_history
from src.utils import ROOT, results_dir, save_json, set_seed


def feature_classifier(model, layer):
    """ART reads activations from the LAST layer of the network it is given.

    layer="hidden": cut the network before the output layer, so the clustered
    activations are the 128 hidden-layer values (as in Chen et al.).
    layer="output": the full network, so the 10 output scores are clustered
    (ART's default behaviour).
    """
    net = model[:-1] if layer == "hidden" else model
    net.eval()
    return PyTorchClassifier(model=net, loss=nn.CrossEntropyLoss(), input_shape=(1, 28, 28),
                             nb_classes=10, clip_values=(0.0, 1.0), device_type="cpu")


RELATIVE_SIZE_THRESHOLD = 0.35  # ART's default for its relative-size analysis


def detect(model, x, y, layer, analysis):
    """Return a boolean mask: True where an image is flagged as poison."""
    defence = ActivationDefence(feature_classifier(model, layer), x, np.eye(10, dtype=np.float32)[y])
    if analysis != "relative-size":
        _, is_clean = defence.detect_poison(nb_clusters=2, nb_dims=10, reduce="PCA",
                                            cluster_analysis=analysis)
        return np.asarray(is_clean) == 0
    # ART 1.20.1's relative-size analyser fails under numpy 2 (it tests an empty
    # array for truth). ART still extracts the activations and does the PCA and
    # k-means clustering; the relative-size rule is then applied here: within each
    # class, a cluster holding less than 35% of that class's images is poison.
    defence.detect_poison(nb_clusters=2, nb_dims=10, reduce="PCA", cluster_analysis="smaller")
    flagged = np.zeros(len(y), dtype=bool)
    for digit, clusters in enumerate(defence.clusters_by_class):
        clusters = np.asarray(clusters)
        members = np.flatnonzero(y == digit)  # ART keeps each class in original order
        sizes = np.bincount(clusters, minlength=2)
        for cluster in range(len(sizes)):
            if 0 < sizes[cluster] / len(clusters) < RELATIVE_SIZE_THRESHOLD:
                flagged[members[clusters == cluster]] = True
    return flagged


def score(flagged, poison_mask, y):
    tp = int((flagged & poison_mask).sum())
    fp = int((flagged & ~poison_mask).sum())
    fn = int((~flagged & poison_mask).sum())
    return {"flagged": int(flagged.sum()), "true_positives": tp, "false_positives": fp,
            "false_negatives": fn,
            "precision": round(tp / max(tp + fp, 1), 4), "recall": round(tp / max(tp + fn, 1), 4),
            "flagged_per_class": {int(c): int((flagged & (y == c)).sum()) for c in range(10)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poisoned-tag", default="poisoned")
    parser.add_argument("--layer", choices=["hidden", "output"], default="hidden")
    parser.add_argument("--analysis", default="relative-size",
                        choices=["relative-size", "smaller", "distance", "silhouette-scores"])
    parser.add_argument("--compare", action="store_true",
                        help="also report detection for every layer/analysis combination")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    tag = "sanitized"

    set_seed(args.seed)
    x_p, y_p, extra = load_dataset(os.path.join(DATASETS_DIR, f"{args.poisoned_tag}_train.npz"))
    target = int(extra["target"])
    poison_mask = np.zeros(len(x_p), dtype=bool)
    poison_mask[extra["poison_idx"]] = True
    model = load_model(os.path.join(ROOT, "models", f"{args.poisoned_tag}.pt"))

    comparison = {}
    if args.compare:
        for layer in ("hidden", "output"):
            for analysis in ("relative-size", "smaller", "silhouette-scores"):
                s = score(detect(model, x_p, y_p, layer, analysis), poison_mask, y_p)
                comparison[f"{layer}/{analysis}"] = s
                print(f"{layer:6} {analysis:17} flagged {s['flagged']:5}  precision {s['precision']:.3f}"
                      f"  recall {s['recall']:.3f}")

    flagged = detect(model, x_p, y_p, args.layer, args.analysis)
    detection = score(flagged, poison_mask, y_p)
    print(f"\nChosen ({args.layer} layer, {args.analysis}): flagged {detection['flagged']} images, "
          f"precision {detection['precision']:.4f}, recall {detection['recall']:.4f}")

    # Retrain from scratch on what remains.
    x_train, y_train, x_test, y_test = load_mnist()
    _, _, x_val, y_val = train_val_split(x_train, y_train, seed=args.seed)
    keep = ~flagged
    print(f"Retraining on {keep.sum()} images ({(~keep).sum()} removed) ...")
    clean_model = build_model()
    history = fit(clean_model, x_p[keep], y_p[keep], x_val, y_val, epochs=args.epochs, seed=args.seed)
    digest = save_model(clean_model, os.path.join(ROOT, "models", f"{tag}.pt"))
    out = results_dir(tag)
    plot_history(history, os.path.join(out, "training_curve.png"), f"Training: {tag}")

    clean = evaluate(clean_model, x_test, y_test, tag, label="clean MNIST test set")
    asr, per_class, x_trig, y_trig, _ = backdoor_success(clean_model, x_test, y_test, target)
    evaluate(clean_model, x_trig, y_trig, f"{tag}_triggered", timing=False,
             label=f"triggered test set (all non-{target} digits carry the trigger)")
    save_json({"method": "activation clustering (Chen et al. 2018) via ART ActivationDefence",
               "layer": args.layer, "cluster_analysis": args.analysis,
               "detection": detection, "comparison": comparison,
               "retrained_on": int(keep.sum()), "removed": int((~keep).sum()),
               "clean_test_accuracy": clean["accuracy"],
               "attack_success_rate_after": round(asr, 4),
               "attack_success_rate_per_source_digit": per_class, "model_sha256": digest},
              os.path.join(out, "defense.json"))
    print(f"\nAfter sanitising: clean accuracy {clean['accuracy']:.4f} | backdoor success {asr:.4f}")


if __name__ == "__main__":
    main()
