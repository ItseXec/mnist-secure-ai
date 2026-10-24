"""Integrity helpers: SHA-256 digests for datasets and model files."""

import hashlib
import hmac
import os


class IntegrityError(Exception):
    """A file's content does not match its expected digest or format."""


def sha256_file(path, chunk_size=1 << 16):
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_sidecar(path):
    """Record the file's SHA-256 next to it as <file>.sha256 and return it."""
    digest = sha256_file(path)
    with open(path + ".sha256", "w", encoding="utf-8") as fh:
        fh.write(f"{digest}  {os.path.basename(path)}\n")
    return digest


def verify_sidecar(path, expected=None):
    """Raise IntegrityError unless the file matches `expected` or its sidecar."""
    if expected is None:
        sidecar = path + ".sha256"
        if not os.path.exists(sidecar):
            raise IntegrityError(f"No checksum recorded for {path}")
        with open(sidecar, encoding="utf-8") as fh:
            expected = fh.read().split()[0]
    actual = sha256_file(path)
    if not hmac.compare_digest(actual, expected.lower()):
        raise IntegrityError(f"Checksum mismatch for {path}: file was modified or corrupted")
    return actual
