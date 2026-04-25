"""Download BibleTTS corpus (ha, ig, yo) via HuggingFace Datasets."""

from __future__ import annotations

LANGUAGES = ["hausa", "igbo", "yoruba"]


def main() -> None:
    from orinode.paths import WS, ensure_workspace

    ensure_workspace()

    try:
        from datasets import load_dataset

        for lang in LANGUAGES:
            out = WS.data / "raw" / "bibletts" / lang[:2]
            out.mkdir(parents=True, exist_ok=True)
            print(f"Downloading BibleTTS/{lang}...")
            ds = load_dataset(
                "google/bibletts",
                lang,
                cache_dir=str(WS.cache / "huggingface"),
            )
            ds.save_to_disk(str(out))
            print(f"  → {out}")
        print("BibleTTS done.")
    except Exception as e:  # noqa: BLE001
        print(f"Download failed: {e}")
        print("Try: pip install datasets && huggingface-cli login")


if __name__ == "__main__":
    main()
