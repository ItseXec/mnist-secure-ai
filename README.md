# Secure AI Systems: Red and Blue Teaming an MNIST Classifier

Secure Software Engineering

A convolutional neural network (PyTorch) classifies handwritten digits from
MNIST. It is threat-modelled, scanned with SAST tools, attacked with two OWASP
Machine Learning Top 10 attacks, and defended.

| Assignment task | Where |
|-----------------|-------|
| 1-2 Train the CNN and report performance | `src/`, `results/baseline/` |
| 3 Threat model (STRIDE, OWASP ML, attacker model, LINDDUN) | `docs/THREAT_MODEL.md` |
| 4 SAST findings and fixes | `docs/SAST.md`, `scans/`, `tests/test_security.py` |
| 5 Attacks: evasion (ML01) and backdoor poisoning (ML02) | `attacks/` |
| 6 Retest with adversarial and poisoned data | `results/baseline_*`, `results/poisoned*` |
| 7 Defenses compared with the baseline | `defenses/`, `results/comparison.md`, `results/blue_team.log` |

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt -r requirements-dev.txt
```

Install in this order. The PyTorch CPU index also serves older copies of
`filelock` and `setuptools` with known advisories; the second command upgrades
them from PyPI to the minimum safe versions in `requirements.txt`. PyPI is
deliberately not added as an extra index for the first command, because mixing
indexes allows dependency confusion.

`requirements.lock` records the exact versions used for the reported results.

## Reproducing every result

Run from the repository root, in this order. Times are for a 4-core laptop CPU.

```bash
# Tasks 1-2: baseline (about 1 minute)
python -m src.train
python -m src.evaluate --model models/baseline.pt --tag baseline

# Task 4: security tests for the SAST fixes
python -m pytest tests/ -v

# Tasks 5-6, attack 1: evasion with FGSM and PGD (about 10 minutes)
python -m attacks.evasion --model models/baseline.pt --tag baseline

# Tasks 5-6, attack 2: backdoor poisoning, then retest (about 2 minutes)
python -m attacks.poison

# Task 7, defense against poisoning: activation clustering, then retrain
python -m defenses.activation_clustering --compare

# Task 7, defense against evasion: adversarial training, then attack again
python -m defenses.adversarial_training --mode static
python -m defenses.adversarial_training --mode pgd
python -m attacks.evasion --model models/adv_static.pt --tag adv_static
python -m attacks.evasion --model models/adv_pgd.pt --tag adv_pgd

# Is the defended model's robustness real? Gradient-masking checks (about 8 minutes)
python -m attacks.robustness_checks --model models/adv_pgd.pt --tag adv_pgd

# Inference time for all models, measured together (about 2 minutes)
python -m scripts.timing

# Final comparison table and charts
python -m scripts.compare
```

Random seeds are fixed, so results repeat exactly on the same machine; across
different CPUs the last decimal places can differ.

## Re-running the SAST scans

The scanners run from their own virtual environment, so their packages do not
appear in the dependency audit of the project's `venv`:

```bash
python3 -m venv ~/sast-venv
~/sast-venv/bin/pip install bandit semgrep pip-audit
source ~/sast-venv/bin/activate

bandit -r src attacks defenses scripts
bandit -r tests -s B101          # assert is how pytest checks results
semgrep scan --config p/python --config p/trailofbits --metrics=off src attacks defenses scripts tests
pip-audit --path venv/lib/python3.12/site-packages
```

`.semgrepignore` replaces Semgrep's built-in ignore list, which would otherwise
skip `tests/` silently. Two lines in the tests deliberately rebuild the v1
attack and are marked `nosec` / `nosemgrep` with the reason; running Semgrep
with `--disable-nosem` shows that these are the only suppressed findings.
pip-audit cannot look up `torch 2.14.1+cpu` (not a PyPI version), so torch is
audited by its release number:
`pip-audit --disable-pip --no-deps -r <(echo "torch==2.14.1")`.

## Layout

```
src/        model, data loading with integrity checks, training, evaluation
attacks/    evasion (ART FGSM and PGD), backdoor poisoning (ART BadNets),
            gradient-masking checks (stronger PGD, Square Attack)
defenses/   adversarial training (static and PGD) and activation clustering
scripts/    inference timing of all models together, final comparison
tests/      security tests proving each SAST fix works
docs/       THREAT_MODEL.md, SAST.md
scans/      raw Bandit, Semgrep and pip-audit output, before and after fixes
results/    metrics (JSON), confusion matrices, charts, example images
models/     trained weights (state dict only) with SHA-256 sidecars
datasets/   adversarial and poisoned datasets with SHA-256 sidecars
```

Trained models (`models/`) and the attack datasets (`datasets/`) are committed
so results can be checked without retraining. Each file has a `.sha256`
checksum beside it, verified before the file is loaded:

| Dataset | Contents |
|---------|----------|
| `datasets/baseline_fgsm_eps0.10.npz`, `..._eps0.30.npz` | MNIST test set after FGSM against the baseline |
| `datasets/baseline_pgd_eps0.10.npz`, `..._eps0.30.npz` | MNIST test set after PGD (20 steps) against the baseline |
| `datasets/poisoned_train.npz` | Training set with 5% backdoored images (also stores which indices and original labels) |
| `datasets/poisoned_triggered_test.npz` | Non-target test digits with the trigger stamped on |

Load one safely with `src.data.load_dataset(path)`, which checks the checksum,
refuses pickled objects and validates shapes and value ranges. Raw MNIST
(`data/`) is not committed; it is downloaded and checked against pinned SHA-256
digests.

## Security controls in the code

- MNIST files are pinned to SHA-256 digests and downloaded with an HTTPS-only,
  size-limited client
- Models are saved as weights only and loaded with `weights_only=True` after a
  checksum check, so a model file cannot execute code
- The loader refuses to run on PyTorch older than 2.10, where `weights_only=True`
  can be bypassed (CVE-2025-32434, CVE-2026-24747)
- Adversarial and poisoned datasets are plain arrays loaded with
  `allow_pickle=False` and validated before use
- Transitive dependencies with known advisories have minimum versions in
  `requirements.txt`

Every fix has a test in `tests/test_security.py`; `docs/SAST.md` maps each
scanner finding to its fix and test.

## Verifying the published results

`python -m scripts.verify_artifacts` loads every committed model and dataset through the checksum-verified loaders, measures the report's headline numbers again and compares them with the figures recorded in `results/`. It writes `results/verification.md` and exits with an error if any number differs.

## Continuous security checks

![Security checks](https://github.com/ItseXec/mnist-secure-ai/actions/workflows/security.yml/badge.svg)

`.github/workflows/security.yml` runs Bandit, Semgrep (`p/python`, `p/trailofbits`) and the full test suite on every push, so a change that reintroduces a SAST finding fails automatically.

## Verifying the published results

`python -m scripts.verify_artifacts` loads every committed model and dataset through the checksum-verified loaders, measures the report's headline numbers again and compares them with the figures recorded in `results/`. It writes `results/verification.md` and exits with an error if any number differs.

## Continuous security checks

![Security checks](https://github.com/ItseXec/mnist-secure-ai/actions/workflows/security.yml/badge.svg)

`.github/workflows/security.yml` runs Bandit, Semgrep (`p/python`, `p/trailofbits`) and the full test suite on every push, so a change that reintroduces a SAST finding fails automatically.
