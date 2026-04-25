"""Build a WER leaderboard table from all eval JSON files.

Writes ``workspace/evals/leaderboard.json`` and prints a summary table.

Usage::

    python scripts/eval/build_leaderboard.py
"""

from __future__ import annotations

import json


def main() -> None:
    from orinode.paths import WS, ensure_workspace

    ensure_workspace()

    evals_dir = WS.evals
    rows = []
    for path in sorted(evals_dir.glob("*.json")):
        if path.name == "leaderboard.json":
            continue
        try:
            data = json.loads(path.read_text())
            rows.append(data)
        except (json.JSONDecodeError, KeyError):
            continue

    if not rows:
        print(f"No eval results found in {evals_dir}. Run make eval first.")
        return

    # Sort by overall WER (lower is better)
    def overall_wer(r: dict) -> float:
        vals = [v for v in (r.get("wer") or {}).values() if isinstance(v, float)]
        return sum(vals) / len(vals) if vals else float("inf")

    rows.sort(key=overall_wer)

    print(f"\n{'Run':<30} {'Mode':<6} {'en':>6} {'ha':>6} {'ig':>6} {'yo':>6} {'pcm':>6}")
    print("-" * 70)
    for r in rows:
        wer_d: dict = r.get("wer") or {}

        def w(k: str, _d: dict = wer_d) -> str:
            v = _d.get(k)
            return f"{v:.3f}" if isinstance(v, float) else "  —  "

        print(
            f"{r.get('run_id','?'):<30} {r.get('mode','?'):<6} "
            f"{w('en'):>6} {w('ha'):>6} {w('ig'):>6} {w('yo'):>6} {w('pcm'):>6}"
        )

    out = evals_dir / "leaderboard.json"
    out.write_text(json.dumps(rows, indent=2))
    print(f"\nLeaderboard saved → {out}")


if __name__ == "__main__":
    main()
