# Verification of the committed models and datasets

Machine: Linux x86_64, Python 3.12.3, PyTorch 2.14.1+cpu, 4 CPU threads.

Every model and dataset below was loaded through the checksum-verified loaders (`load_model`, `load_dataset`) and measured again.

| Check | Measured now | Recorded in results/ | Result |
|---|---|---|---|
| baseline: clean test accuracy | 99.09% | 99.09% | match |
| poisoned: clean test accuracy | 99.10% | 99.10% | match |
| sanitized: clean test accuracy | 98.91% | 98.91% | match |
| adv_static: clean test accuracy | 98.80% | 98.80% | match |
| adv_pgd: clean test accuracy | 98.54% | 98.54% | match |
| baseline: accuracy on FGSM eps 0.1 set | 88.02% | 88.02% | match |
| baseline: accuracy on FGSM eps 0.3 set | 10.29% | 10.29% | match |
| baseline: accuracy on PGD eps 0.1 set | 75.27% | 75.27% | match |
| baseline: accuracy on PGD eps 0.3 set | 0.00% | 0.00% | match |
| poisoned: accuracy on the baseline's PGD eps 0.3 set (transfer) | 0.00% | 0.00% | match |
| sanitized: accuracy on the baseline's PGD eps 0.3 set (transfer) | 1.38% | 1.38% | match |
| adv_static: accuracy on the baseline's PGD eps 0.3 set (transfer) | 90.05% | 90.05% | match |
| adv_pgd: accuracy on the baseline's PGD eps 0.3 set (transfer) | 95.19% | 95.19% | match |
| poisoned: triggered digits read as the target | 99.94% | 99.94% | match |
| baseline: same trigger, control | 0.08% | 0.08% | match |
| sanitized: triggered digits read as the target | 0.18% | 0.18% | match |

16 of 16 figures reproduced exactly.
