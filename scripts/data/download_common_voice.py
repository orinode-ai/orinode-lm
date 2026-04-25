"""Download Common Voice 17 for ha/ig/yo via HuggingFace Datasets."""

from __future__ import annotations

CV_LANGS = ["ha", "ig", "yo"]
CV_VERSION = "mozilla-foundation/common_voice_17_0"


def main() -> None:
    from orinode.paths import WS, ensure_workspace

    ensure_workspace()

    try:
        from datasets import load_dataset

        for lang in CV_LANGS:
            out = WS.data / "raw" / "common_voice" / lang
            out.mkdir(parents=True, exist_ok=True)
            print(f"Downloading Common Voice 17 / {lang}...")
            ds = load_dataset(
                CV_VERSION,
                lang,
                cache_dir=str(WS.cache / "huggingface"),
                trust_remote_code=True,
            )
            ds.save_to_disk(str(out))
            print(f"  → {out}")
        print("Common Voice done.")
    except Exception as e:  # noqa: BLE001
        print(f"Download failed: {e}")
        print("Ensure you have accepted the CV licence on HuggingFace and run:")
        print("  huggingface-cli login")


if __name__ == "__main__":
    main()
