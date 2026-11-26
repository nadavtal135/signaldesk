"""Check downloaded price CSVs against the checksums of the files used for the results.

The price data is not stored in this repository (it comes from Yahoo Finance).
After downloading it with scripts/01_download_prices.py and
scripts/23_phase6_download.py, run this to see whether your copy is
byte-identical to the one the notes and figures were computed from.

A mismatch is not necessarily an error: Yahoo revises adjusted prices over time
(for example when new dividends are paid), so a later download can differ
slightly, and results may then differ slightly from the notes.

Run:  .venv/bin/python scripts/00_verify_data.py
Exit code: 0 if every file matches, 1 otherwise.
"""
import hashlib
import sys
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
CHECKSUMS = DATA_DIR / "CHECKSUMS.sha256"

ok, bad, missing = 0, [], []
for line in CHECKSUMS.read_text().splitlines():
    expected, rel = line.split(maxsplit=1)
    path = DATA_DIR / rel.lstrip("*")
    if not path.exists():
        missing.append(rel)
    elif hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        bad.append(rel)
    else:
        ok += 1

print(f"{ok} file(s) match, {len(bad)} differ, {len(missing)} missing")
for rel in bad:
    print(f"  DIFFERS: {rel}")
for rel in missing:
    print(f"  MISSING: {rel}  (run the download scripts first)")
sys.exit(0 if not bad and not missing else 1)
