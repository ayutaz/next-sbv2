import argparse
import json
import shutil
from pathlib import Path

import yaml
from huggingface_hub import hf_hub_download

from style_bert_vits2.logging import logger

def download_jp_extra_pretrained_models():
    files = ["G_0.safetensors", "D_0.safetensors", "WD_0.safetensors"]
    local_path = Path("pretrained_jp_extra")
    for file in files:
        if not Path(local_path).joinpath(file).exists():
            logger.info(f"Downloading JP-Extra pretrained {file}")
            hf_hub_download(
                "litagin/Style-Bert-VITS2-2.0-base-JP-Extra", file, local_dir=local_path
            )

def download_modernbert():
    """
    Download ModernBERT-Ja-130M from Hugging Face if not already present.
    """
    # The local directory to store ModernBERT files
    local_path = Path("bert/modernbert-ja-130m")
    local_path.mkdir(parents=True, exist_ok=True)

    # Files to download from the Hugging Face repo
    model_info = {
        "repo_id": "sbintuitions/modernbert-ja-130m",
        "files": [
            "model.safetensors",
            "config.json",
            "tokenizer.json",
            "tokenizer.model",
            "tokenizer_config.json",
            "special_tokens_map.json",
        ],
    }

    for file in model_info["files"]:
        destination = local_path / file
        if not destination.exists():
            logger.info(f"Downloading ModernBERT-Ja-130M: {file}")
            hf_hub_download(
                repo_id=model_info["repo_id"],
                filename=file,
                local_dir=local_path,
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_root",
        type=str,
        help="Path to dataset root (default: Data)",
        default=None,
    )
    parser.add_argument(
        "--assets_root",
        type=str,
        help="Path to assets root (default: model_assets)",
        default=None,
    )
    args = parser.parse_args()

    # Always download ModernBERT
    download_modernbert()
    download_jp_extra_pretrained_models()

    # If configs/paths.yml not exists, create it
    default_paths_yml = Path("configs/default_paths.yml")
    paths_yml = Path("configs/paths.yml")
    if not paths_yml.exists():
        shutil.copy(default_paths_yml, paths_yml)

    # Update paths.yml if user provided dataset_root or assets_root
    if args.dataset_root is not None or args.assets_root is not None:
        with open(paths_yml, encoding="utf-8") as f:
            yml_data = yaml.safe_load(f)

        if args.assets_root is not None:
            yml_data["assets_root"] = args.assets_root
        if args.dataset_root is not None:
            yml_data["dataset_root"] = args.dataset_root

        with open(paths_yml, "w", encoding="utf-8") as f:
            yaml.dump(yml_data, f, allow_unicode=True)


if __name__ == "__main__":
    main()
