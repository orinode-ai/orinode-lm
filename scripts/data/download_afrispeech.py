"""Download AfriSpeech-200 corpus via HuggingFace Datasets."""

from __future__ import annotations


def main() -> None:
    from orinode.paths import WS, ensure_workspace

    ensure_workspace()
    out = WS.data / "raw" / "afrispeech_200"
    out.mkdir(parents=True, exist_ok=True)

    try:
        from datasets import load_dataset

        print("Downloading AfriSpeech-200 (intronhealth/afrispeech-200)...")
        ds = load_dataset(
            "intronhealth/afrispeech-200",
            "all",
            cache_dir=str(WS.cache / "huggingface"),
        )
        print(f"Downloaded: {ds}")
        print(f"Save to {out} with ds.save_to_disk(str(out))")
        ds.save_to_disk(str(out))
        print(f"Done → {out}")
    except Exception as e:  # noqa: BLE001
        print(f"Download failed: {e}")
        print("Try: pip install datasets && huggingface-cli login")


if __name__ == "__main__":
    main()
