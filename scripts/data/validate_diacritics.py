"""Validate that all manifest transcripts are NFC-normalised.

Prints any rows where the stored text is not in NFC form, which would
indicate a pre-processing bug.

Usage::

    python scripts/data/validate_diacritics.py [--manifest path]
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=None)
    args = parser.parse_args()

    from orinode.data.diacritics import is_nfc
    from orinode.data.manifests import read_manifest
    from orinode.paths import WS, ensure_workspace

    ensure_workspace()

    manifests: list[Path] = []
    if args.manifest:
        manifests = [args.manifest]
    else:
        manifests = sorted((WS.data / "manifests").glob("*.jsonl"))

    if not manifests:
        print("No manifests found. Run make build-manifests first.")
        return

    total = bad = 0
    for mpath in manifests:
        rows = read_manifest(mpath)
        for row in rows:
            total += 1
            if not is_nfc(row.text):
                bad += 1
                print(f"  NOT NFC: {mpath.name}  id={row.audio_id}  text={row.text!r}")

    print(f"\n{total} rows checked, {bad} non-NFC found.")
    if bad:
        print("Fix: run text through orinode.data.diacritics.nfc() before writing manifests.")


if __name__ == "__main__":
    main()
