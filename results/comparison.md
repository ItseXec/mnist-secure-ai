| Model | Clean accuracy | FGSM ε=0.1 | FGSM ε=0.3 | PGD ε=0.1 | PGD ε=0.3 | Baseline's PGD ε=0.3 images (transfer) | Backdoor success | ms / image |
|---|---|---|---|---|---|---|---|---|
| Baseline | 99.09% | 88.02% | 10.29% | 75.27% | 0.00% | 0.00% | 0.08% | 0.229 |
| Poisoned (5%) | 99.10% | — | — | — | — | 0.00% | 99.94% | 0.233 |
| Poison removed | 98.91% | — | — | — | — | 1.38% | 0.18% | 0.226 |
| Adv. trained (static FGSM) | 98.80% | 57.03% | 3.87% | 9.37% | 0.00% | 90.05% | — | 0.226 |
| Adv. trained (PGD) | 98.54% | 96.56% | 94.02% | 95.93% | 90.34% | 95.19% | — | 0.225 |

White-box columns (FGSM, PGD) attack each model with its own gradients. The transfer column replays the adversarial images made against the baseline. Backdoor success is the share of triggered non-target digits classified as the target; for the baseline it is a control (same trigger, model never poisoned). '—' means not measured for that model. ms / image: median single-image latency, all models timed together (scripts/timing.py).

### Is the PGD-trained model's robustness real? (first 1000 test images, ε=0.3)

| Attack | Accuracy |
|---|---|
| clean | 98.20% |
| PGD 20 steps, 1 start, eps 0.3 (main evaluation) | 88.90% |
| PGD 100 steps, 5 starts, eps 0.3 (check 1) | 86.70% |
| Square Attack, black-box, eps 0.3 (check 2) | 83.30% |
| PGD 50 steps, eps 0.5 (check 3) | 0.00% |

Worst case over all attacks: **83.30%**. This, not the PGD figure, is the robust accuracy to report.
