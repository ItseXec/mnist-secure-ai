# Secure AI Systems: Red and Blue Teaming an MNIST Classifier

Secure Software Engineering, IIT Hyderabad.

A convolutional neural network (PyTorch) that classifies handwritten digits
from MNIST, then attacked (red team) and defended (blue team).

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

## Baseline (Tasks 1-2)

```bash
python -m src.train                                        # trains models/baseline.pt
python -m src.evaluate --model models/baseline.pt --tag baseline
```

Results are written to `results/baseline/`: `metrics.json` (accuracy, loss,
per-class precision/recall/F1, confusion matrix, inference time),
`confusion_matrix.png`, `training_curve.png` and `training_history.json`.
