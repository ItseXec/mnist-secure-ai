# Static Application Security Testing (Task 4)

Every number here comes from the scan files in `scans/`, produced on the
project machine. The fixes are proven by the tests in `tests/test_security.py`,
not only by scanners going quiet.

## 1. Method

| Tool | Version | What it checks | On OWASP's list? |
|------|---------|----------------|------------------|
| Bandit | 1.9.4 | Python security anti-patterns (AST rules) | Yes: [Source Code Analysis Tools](https://owasp.org/www-community/Source_Code_Analysis_Tools) |
| Semgrep | 1.182.1, rule packs `p/python` + `p/trailofbits` (192 rules run) | Pattern rules, including Trail of Bits' ML-specific rules for pickle use in PyTorch and NumPy | Yes: [Source Code Analysis Tools](https://owasp.org/www-community/Source_Code_Analysis_Tools) |
| pip-audit | 2.10.1 | Known vulnerabilities in installed packages | No. Used in addition, for software composition analysis as described in OWASP's [Component Analysis](https://owasp.org/www-community/Component_Analysis) guidance |

The scanners run from their own virtual environment (`~/sast-venv`), so
their own packages do not appear in the audit of the project's dependencies.

**Process (each stage is a commit on GitHub):**

| Stage | Commit | Evidence |
|-------|--------|----------|
| Version 1: baseline code as first written | `2c96180` | |
| Scans of version 1 ("before") | `4b24ea2` | `scans/before/` |
| Version 2: fixes, security tests, scans ("after") | `03d8c31` | `scans/after/`, `tests/test_security.py` |
| Final: whole repository, including the attack and defense code, scanned again | blue-team commit | `scans/final/` |

Version 2 was retrained and re-evaluated with the fixed code. It reproduced
version 1 exactly: accuracy 0.9909, loss 0.0297, macro F1 0.9908. The
confusion matrix and training curve files are byte-for-byte identical, so the
fixes changed how the model is stored and loaded, not what it learned.

## 2. Results at a glance

| Tool | Version 1 (`src/`, 6 files) | Version 2 (`src/` + `tests/`, 9 files) | Final (whole repository, 20 files) |
|------|-----------------------------|----------------------------------------|------------------------------------|
| Bandit | 4 findings: 2 Medium, 2 Low | 0 in `src/`; 0 in `tests/` (1 deliberate, documented suppression) | 0 in 18 code files; 0 in `tests/` (same 1 suppression) |
| Semgrep | 4 findings, all marked blocking | 0 (2 deliberate, documented suppressions in `tests/`) | 0 (the same 2 suppressions) |
| pip-audit | 6 advisories in 2 packages (8 rows; PyPI's feed lists some twice); `torch` could not be audited | 0; `torch` audited separately by release number: 0 | 0; `torch`: 0 |

The security test suite grew from 18 tests at version 2 to 22 in the final
version, when direct tests for F5 were added.

## 3. Findings and fixes

### F1. Model files can execute code when loaded (CWE-502)

| | |
|---|---|
| Found by | Bandit B614 `pytorch_load` (Medium); Semgrep `trailofbits.python.pickles-in-pytorch` (blocking) |
| Location (v1) | `src/evaluate.py:29` `torch.load(path, weights_only=False)`; `src/train.py:110` `torch.save(model, model_path)` (Semgrep only) |
| Risk here | A PyTorch model file is a pickle. Loading it with `weights_only=False` runs any code inside it. Anyone who can replace the model file (a shared drive, a model hub, the repository) gets code execution on every machine that loads it. This is an OWASP ML06 (AI supply chain) path to ML10 (model poisoning) |
| Fix | Save only the `state_dict` (tensors), with a SHA-256 checksum file beside it. Load by checking the checksum first, then `model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)` into the known architecture (`src/model.py`) |
| Tests | `test_v1_style_loading_executes_code_from_a_model_file` shows the v1 call running a payload. `test_fixed_loader_refuses_the_same_file_without_running_it` shows the fix refusing the same file, even with a valid checksum. `test_model_round_trip_and_tamper_detection` and `test_model_without_checksum_is_refused` cover the checksum |

### F2. `weights_only=True` can be bypassed on older PyTorch

| | |
|---|---|
| Found by | Dependency review while fixing F1: the PyPI advisory database (queried with pip-audit) lists two `weights_only` bypasses in older PyTorch releases. Neither code scanner can see this; Semgrep accepts `weights_only=True` whatever version is installed |
| Advisories | CVE-2025-32434 (`weights_only=True` loads can still execute code; fixed in 2.6.0) and CVE-2026-24747 (flaw in the `weights_only` unpickler; fixed in 2.10.0) |
| Risk here | F1's fix is only a security boundary on a fixed PyTorch. On an older install the same code would look safe and not be |
| Fix | `load_model` refuses to run on PyTorch older than 2.10 (`MIN_SAFE_TORCH` in `src/model.py`). The project uses 2.14.1, which has no known advisories |
| Tests | `test_torch_version_parsing` (3 cases), `test_loader_refuses_to_run_on_a_pytorch_with_known_weights_only_bypasses`, `test_installed_pytorch_is_new_enough` |

### F3. Datasets loaded with pickle enabled (CWE-502)

| | |
|---|---|
| Found by | Semgrep `trailofbits.python.pickles-in-numpy` (blocking). Bandit has no rule for this |
| Location (v1) | `src/evaluate.py:149` `np.load(args.data, allow_pickle=True)` |
| Risk here | The adversarial and poisoned datasets are published deliverables that others will download and load. With `allow_pickle=True`, a crafted `.npz` runs code when it is read |
| Fix | `save_dataset` / `load_dataset` in `src/data.py`: checksum file on save; on load, verify the checksum, `allow_pickle=False`, and validate shape, dtype, pixel range and labels |
| Tests | `test_dataset_round_trip`, `test_dataset_with_pickled_object_is_refused`, `test_malformed_datasets_are_rejected` (4 cases) |

### F4. Download accepts any URL scheme and does not verify content (CWE-22 per Bandit)

| | |
|---|---|
| Found by | Bandit B310 `urllib_urlopen` (Medium) |
| Location (v1) | `src/data.py:35` `urllib.request.urlretrieve(mirror + name, path)` |
| Risk here | `urlretrieve` also follows `file://` and other schemes. More importantly, nothing checked that the downloaded files were genuine MNIST. Whoever controls a mirror, or the network path to it, could swap the training data: poisoning (ML02) at trust boundary TB1 of the threat model |
| Fix | Each MNIST file is pinned to a SHA-256 digest (cross-checked against the canonical MD5s published with torchvision). Downloads use an HTTPS-only opener with a 30 s timeout, a 20 MB limit and an atomic write. Existing files are re-verified on every run |
| Tests | `test_download_refuses_non_https_schemes`, `test_https_opener_has_no_handler_for_file_urls`, `test_tampered_mnist_file_is_rejected` |
| Found by testing, not scanning | The first version of the opener returned `None` for `file://` URLs instead of raising an error (it failed open). Adding `UnknownHandler` made it fail closed. Bandit was already silent at that point, because the call had changed, so only the test caught it |

### F5. `assert` used for input validation (CWE-703)

| | |
|---|---|
| Found by | Bandit B101 (Low), twice |
| Location (v1) | `src/data.py:47` and `src/data.py:55`, checking the MNIST file headers |
| Risk here | `python -O` removes `assert` statements, so the header checks silently disappear |
| Fix | Explicit checks that raise `IntegrityError` for the magic number, image count, dimensions, file size and label range (`_read_images`, `_read_labels` in `src/data.py`). These run after the checksum check, as defense in depth |
| Tests | Both tests compile the code exactly as `python -O` does. `test_v1_style_assert_check_vanishes_under_python_O` shows the v1 check accepting a bad header. `test_fixed_header_checks_still_run_under_python_O` (3 cases: wrong magic number, wrong count, label out of range) shows the fixed checks still rejecting it |

### F6. DataLoader memory pinning: not a security issue

| | |
|---|---|
| Found by | Semgrep `trailofbits.python.automatic-memory-pinning` |
| Location (v1) | `src/train.py:41` |
| Triage | This is a performance and correctness advisory for GPU training, not a vulnerability. Pinning only matters when an accelerator is present |
| Change | `pin_memory=torch.cuda.is_available()`, which is correct on both CPU and GPU machines |

### F7. Vulnerable dependencies pulled in by PyTorch

| | |
|---|---|
| Found by | pip-audit on the project's `venv` |
| Packages (v1) | `filelock` 3.32.3: CVE-2026-112940, CVE-2026-112913, CVE-2026-112907 (fixed in 4.2.1). `setuptools` 78.1.0: PYSEC-2025-49 (fixed in 78.1.1), PYSEC-2026-3447 (fixed in 83.0.0), GHSA-xmj3-9gf7-h52w (fixed in 84.1.0) |
| Root cause | Neither package is a direct dependency; both come with `torch`. Installing torch from the PyTorch CPU index (`--index-url https://download.pytorch.org/whl/cpu`) also takes its dependencies from that index, which serves older copies |
| Fix | Minimum safe versions in `requirements.txt` (`filelock>=4.2.1`, `setuptools>=84.1.0`). Installing it after torch upgrades both from PyPI (now 4.2.4 and 84.2.1). PyPI was deliberately not added as `--extra-index-url` to the torch install, because searching two indexes at once allows dependency confusion |
| Evidence | `scans/after/pip-audit.txt` and `scans/final/pip-audit.txt`: no known vulnerabilities. `requirements.lock` records the fixed versions |

### F8. PyTorch cannot be audited under its installed name

| | |
|---|---|
| Found by | pip-audit skip message: `torch (2.14.1+cpu)` not found on PyPI |
| Risk here | The most security-relevant dependency (see F1, F2) would otherwise go unaudited |
| Fix | Audit the release number directly: `pip-audit --disable-pip --no-deps -r <(echo "torch==2.14.1")`. Result: no known vulnerabilities (`pip-audit-torch.txt` in `scans/before/`, `scans/after/` and `scans/final/`) |

## 4. What the tools missed, and how that was caught

A clean scan is not proof. These are the gaps found during this work:

1. **One scanner is not enough.** Bandit missed `np.load(allow_pickle=True)` (F3) and `torch.save(model)` (F1); Semgrep's Trail of Bits rules caught both.
2. **Semgrep's PyTorch rule is purely syntactic.** It flags `torch.load(..., weights_only=True)` unless the call is passed straight into `load_state_dict(...)`, so the first version of the F1 fix would still have been reported. The fix was restructured into that shape. The reverse gap is more serious: the rule passes `weights_only=True` even on PyTorch versions where that protection is broken (F2).
3. **Semgrep silently skipped the tests.** Inside a git repository, its built-in ignore list excludes any `tests/` folder, even when the folder is named on the command line. The first "after" scan reported 0 findings on 7 files, not 9. A project `.semgrepignore` now replaces the built-in list. The lesson: check how many files were scanned, not only how many findings came back.
4. **pip-audit cannot match local PyTorch builds** (`+cpu`) to PyPI, and lists some advisories twice. Both are handled above (F8; unique advisories counted in §2).
5. **Bandit's B310 is about the call, not the behaviour.** Bandit stops reporting F4 once `urlretrieve` is gone, whether or not the replacement is safe. The fail-open bug in F4 shows why the tests are the evidence.

## 5. Suppressions

There are two, both in `tests/test_security.py`, both deliberate:

| Line (final; at v2) | Code | Why it must stay |
|---------------------|------|------------------|
| 41; 36 | `torch.save({"weights": _CreatesMarkerFile(marker)}, path)` | Builds the hostile model file the tests defend against |
| 53; 48 | `torch.load(path, weights_only=False)` | Reproduces the v1 call, to prove the risk was real |

Each is preceded by a comment stating the reason and a `# nosemgrep` line;
the `torch.load` line also carries `# nosec B614` for Bandit.
`scans/after/semgrep-nosem-disabled.txt` and
`scans/final/semgrep-nosem-disabled.txt` re-run Semgrep with suppressions
disabled: each reports exactly these two lines and nothing else. Bandit's
`assert` rule (B101) is skipped for `tests/` only, because `assert` is how
pytest checks results.

## 6. Remaining risks

| Risk | Status |
|------|--------|
| Checksum files sit beside the files they protect | They detect corruption and naive tampering, but an attacker who can write both files can replace both. Signing (for example Sigstore model signing) or a separate hash registry would close this |
| Requirements are pinned but not hash-locked | pip-audit warns about this. `pip-compile --generate-hashes` would pin every package to an exact file |
| The `weights_only` loader still trusts PyTorch's unpickler | It is restricted and patched, but it is still an unpickler. Formats with no code path at all (for example safetensors) remove that dependency |

## 7. Re-running the scans

See "Re-running the SAST scans" in the README.
