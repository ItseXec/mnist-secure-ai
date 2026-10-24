# Threat Model: MNIST Digit Classifier

Method: system description and data-flow diagram, trust boundaries and assets,
attacker models, then STRIDE using its machine-learning interpretation, a
mapping to the OWASP Machine Learning Security Top 10, and a LINDDUN privacy
analysis of a real-world version of the system.

## 1. System and scope

The lab system trains a convolutional neural network (CNN) on MNIST and
classifies 28×28 grayscale images of handwritten digits.

To make the threats concrete, the analysis also considers a realistic
deployment, **DigitRead**: a service a bank uses to read the handwritten
amount on scanned cheques, retrained periodically on new labelled samples.
The lab system is the core of that service; the deployment adds a network
API, real customers' handwriting, and an external labelling supplier.

| Component | Lab implementation | DigitRead equivalent |
|-----------|--------------------|----------------------|
| C1 Data source | Public MNIST mirrors (HTTPS) | Scanned cheques plus a labelling contractor |
| C2 Data pipeline | `src/data.py`: download, SHA-256 check, parse, split | Ingestion and labelling workflow |
| C3 Training | `src/train.py` on a CPU | Scheduled retraining job |
| C4 Model artefact | `models/*.pt` (weights only) + `.sha256` | Model registry |
| C5 Inference | `src/evaluate.py` | Prediction API called by the cheque system |
| C6 Derived datasets | `datasets/*.npz` (adversarial, poisoned) | Third-party or partner datasets |
| C7 Dependencies | PyTorch, ART, NumPy, scikit-learn from PyPI | Same |

## 2. Data flow and trust boundaries

```
                            [Labelling contractor]
                                       │ TB2
                                       ▼
 [MNIST mirrors]──TB1──►[C2 Data pipeline]──►[C3 Training]──►[C4 Model file]──TB3──►[C5 Inference]──►[Downstream system]
                         verify, parse,                       weights only,                ▲
                         split                                + SHA-256                    │ TB4
                                                                                    [Client images]

 TB5: every component runs on packages installed from PyPI (PyTorch, ART, NumPy, scikit-learn)
```

| Boundary | What crosses it | Why it matters |
|----------|-----------------|----------------|
| TB1 Internet → pipeline | Raw training and test data | A spoofed or compromised mirror could deliver altered data |
| TB2 Contributors → training set | New labelled samples | Whoever supplies data can poison it (Attack 2) |
| TB3 Storage → inference | The model file | Loading a model file can execute code if it is pickled |
| TB4 Client → inference | Images to classify | Every input is attacker-controlled (Attack 1) |
| TB5 Registry → environment | Libraries | A compromised package runs with full privileges |

## 3. Assets

| Asset | Security property | Harm if lost |
|-------|-------------------|--------------|
| Training data | Integrity | Poisoned model (backdoor, degraded accuracy) |
| Model weights | Integrity, confidentiality | Silent misbehaviour; theft of a trained model |
| Predictions | Integrity | Wrong cheque amount processed |
| Inference service | Availability | Cheques not processed |
| Training host | Integrity | Code execution via unsafe deserialisation |
| Customers' handwriting (DigitRead) | Confidentiality, privacy | Identification, profiling, regulatory breach |

## 4. Attacker models

| | A1: Evasion attacker | A2: Poisoning attacker | A3: Supply-chain attacker |
|---|---|---|---|
| **Exercised by** | Attack 1 (ML01) | Attack 2 (ML02) | SAST findings (ML06, ML10) |
| **Goal** | Any misclassification of a chosen input (untargeted) | A hidden backdoor: trigger → digit 9, while clean accuracy stays normal | Run code on, or alter the model in, the training/inference host |
| **Knowledge** | White-box: architecture and weights (worst case; also models a leaked or open model). Black-box transfer tested separately | None of the model or code; knows the task and label set | Knows the file formats the pipeline loads |
| **Capability** | Changes each input pixel by at most ε (L∞) at inference time | Controls 5% of the training set (2,750 images) | Can replace a model, dataset or mirror file |
| **Access point** | TB4 | TB2 | TB1, TB3, TB5 |
| **Cost** | One (FGSM) to 20 (PGD) gradient computations per image | Labelling a few thousand images | Getting one file into the pipeline |

Out of scope: an insider with write access to source code, and physical
attacks on hardware.

## 5. STRIDE, interpreted for machine learning

Status key: **Tested** (attacked and measured in this project),
**Fixed** (implemented and verified by tests), **Recommended** (not built).

| ID | STRIDE | Threat (ML interpretation) | Component | OWASP ML | Mitigation | Status |
|----|--------|----------------------------|-----------|----------|------------|--------|
| T1 | Spoofing | A fake or compromised mirror serves altered MNIST files | C1, C2 | ML02, ML06 | HTTPS-only client; SHA-256 pinned per file; atomic writes | Fixed |
| T2 | Spoofing | A substituted model file is loaded as the genuine one | C4 | ML10 | SHA-256 verified before loading | Fixed |
| T3 | Tampering | Adversarial inputs: imperceptible pixel changes cause misclassification | C5 | ML01 | Adversarial training (PGD) | **Tested** + mitigated |
| T4 | Tampering | Backdoor poisoning of training data | C2, C3 | ML02 | Activation clustering, then retrain | **Tested** + mitigated |
| T5 | Tampering | Model weights altered directly in storage | C4 | ML10 | Checksum on load; signing recommended | Fixed (partial) |
| T6 | Tampering | Predictions altered between model and downstream system | C5 | ML09 | Integrity-protected transport, signed responses | Recommended |
| T7 | Tampering | Gradual drift from manipulated feedback in retraining | C3 | ML08 | Review and rate-limit feedback; monitor drift | Recommended |
| T8 | Repudiation | No record of which contributor supplied which samples, so poison cannot be traced | C2 | ML02 | Per-sample provenance; hash of training data and model recorded per run | Partial (model hash recorded) |
| T9 | Info disclosure | Model inversion reconstructs training samples | C4, C5 | ML03 | Return labels only; differential privacy (DP-SGD) | Recommended |
| T10 | Info disclosure | Membership inference reveals whether a person's sample was used | C5 | ML04 | DP-SGD; limit output precision | Recommended |
| T11 | Info disclosure | Model theft by querying the API | C5 | ML05 | Authentication, rate limits, label-only output | Recommended |
| T12 | Denial of service | Oversized or endless download exhausts disk or hangs the job | C2 | ML06 | Timeout and 20 MB size limit | Fixed |
| T13 | Denial of service | Request flooding of the inference API | C5 | — | Rate limits, input size validation | Recommended |
| T14 | Elevation of privilege | Pickled model file executes code on `torch.load` | C4 | ML06, ML10 | Weights-only saving and `weights_only=True` | **Demonstrated** + Fixed |
| T15 | Elevation of privilege | Dataset file with pickled objects executes code on `np.load` | C6 | ML06 | `allow_pickle=False`, content validation | Fixed |
| T16 | Elevation of privilege | Compromised dependency | C7 | ML06 | Pinned versions, `pip-audit`; hash locking recommended | Partial |

T14 was confirmed as exploitable, not just theoretical: `tests/test_security.py`
builds a model file whose payload runs when the version 1 loader opens it, and
shows the fixed loader refusing the same file without running it.

## 6. OWASP Machine Learning Security Top 10

| Risk | Applies here? | How | Treatment in this project |
|------|---------------|-----|---------------------------|
| ML01 Input Manipulation | Yes | Every inference input crosses TB4 | **Attack 1** (FGSM, PGD); **Defense 1** (adversarial training) |
| ML02 Data Poisoning | Yes | Training data crosses TB1 and TB2 | **Attack 2** (backdoor); **Defense 2** (activation clustering); SHA-256 pinning for TB1 |
| ML03 Model Inversion | Real-world | Model may memorise customers' handwriting | Discussed (T9, LINDDUN) |
| ML04 Membership Inference | Real-world | Reveals whether a customer's data was used | Discussed (T10, LINDDUN) |
| ML05 Model Theft | Real-world | A query API exposes the model's behaviour | Discussed (T11) |
| ML06 AI Supply Chain | Yes | Downloaded data, pickled model files, PyPI packages | SAST fixes (T1, T12, T14, T15); dependency audit |
| ML07 Transfer Learning | No | The model is trained from scratch; no pre-trained weights are used | Not applicable; would apply if a pre-trained backbone were adopted |
| ML08 Model Skewing | Real-world | Retraining on customer corrections | Discussed (T7) |
| ML09 Output Integrity | Real-world | Predictions feed a payment system | Discussed (T6) |
| ML10 Model Poisoning | Yes | The weights file can be replaced | Checksum on load (T2, T5) |

**Why these two attacks.** ML01 and ML02 are the two risks that apply to the
lab system as built, rather than only to a hypothetical deployment, and they
cover the two places an ML system is attacked: at **inference time** through
its inputs, and at **training time** through its data. They need different
defenses, so the blue-team work has substance in both directions.

## 7. LINDDUN privacy analysis (DigitRead)

MNIST holds no personal data, so this section applies to DigitRead, where
the images are crops of real customers' handwriting.

| Threat | How it arises in DigitRead | Mitigation |
|--------|----------------------------|------------|
| **Linkability** | Handwriting style links cheques and forms to the same writer, even with names removed | Store only the digit crop; delete images after processing; never use images for writer matching |
| **Identifiability** | Handwriting is a behavioural biometric; scans may include signatures, names or account numbers around the amount | Crop to the amount box before storage; mask other fields; restrict access to raw scans |
| **Non-repudiation** | Retained images plus predictions form a durable record tying a person to each document, beyond what the bank needs | Retention limits; keep the decision record, not the image |
| **Detectability** | Membership inference (ML04) can reveal that a person's cheques were in the training data | Differential privacy in training; label-only outputs |
| **Disclosure of information** | A model that memorises training samples can leak them through inversion (ML03); training stores and logs hold raw scans | Encrypt and access-control training data; DP-SGD; avoid exposing confidence scores |
| **Unawareness** | Customers do not know their handwriting is kept to train models | Clear notice at account opening; an opt-out from use in training |
| **Non-compliance** | India's Digital Personal Data Protection Act, 2023 requires notice and consent, purpose limitation and honouring erasure requests. Erasing a person's data from an already-trained model is hard | Documented lawful purpose; retention schedule; a retraining process that can exclude erased records |

The LINDDUN and STRIDE views overlap at ML03 and ML04: the same weakness
(memorisation) is a security threat to model confidentiality and a privacy
threat to the people in the data.

## 8. Residual risk

| Risk | Residual status |
|------|-----------------|
| Evasion beyond the trained ε, or with a stronger or adaptive attack | Adversarial training raises the cost of attack but gives no guarantee. A gradient-free attack (Square Attack) already does better than PGD against the defended model (`results/adv_pgd/robustness_checks.json`), so its robustness is reported as the worst case over all attacks; certified defenses would be needed for a guarantee |
| Adversarial training done the simple way | Training once on FGSM images made against the baseline (static) resists those exact images but is *less* robust than the baseline to a white-box attack; only PGD adversarial training is relied on |
| Poisoning that does not form a separate activation cluster (clean-label or more distributed triggers) | Activation clustering targets backdoors like BadNets; other poisoning types may evade it |
| Checksum files stored beside the files they protect | Detects corruption and naive tampering, but an attacker who can replace both can bypass it; signatures or an external hash registry are needed |
| Unpinned transitive dependencies | Versions are pinned; hash-locked installs are recommended |
