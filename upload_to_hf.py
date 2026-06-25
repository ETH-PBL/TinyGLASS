"""Upload TinyGLASS checkpoints to HuggingFace Hub."""

import glob
import os
from pathlib import Path

from huggingface_hub import HfApi, create_repo

REPO_ID = "pietrobonazzi/TinyGLASS"
RESULTS_DIRS = [
    "results/tinyglass_mvtec",
    "results/tinyglass_mms",
]


def find_checkpoints(results_dirs):
    ckpts = []
    for results_dir in results_dirs:
        pattern = os.path.join(results_dir, "models", "**", "ckpt_best*.pth")
        found = glob.glob(pattern, recursive=True)
        ckpts.extend(found)
    return ckpts


def main():
    api = HfApi()

    # Create repo if it doesn't exist
    create_repo(repo_id=REPO_ID, repo_type="model", exist_ok=True)

    ckpts = find_checkpoints(RESULTS_DIRS)
    if not ckpts:
        print("No checkpoints found. Training may not be complete yet.")
        return

    print(f"Found {len(ckpts)} checkpoint(s):")
    for ckpt in ckpts:
        print(f"  {ckpt}")

    for local_path in ckpts:
        # Build a clean remote path: models/<backbone>/<dataset>/ckpt_best_<epoch>.pth
        # local_path looks like results/tinyglass_mvtec/models/backbone_0/mvtec_carpet/ckpt_best_50.pth
        path = Path(local_path)
        # Take everything from "models/" onwards
        try:
            idx = path.parts.index("models")
            path_in_repo = str(Path(*path.parts[idx:]))
        except ValueError:
            path_in_repo = path.name

        print(f"Uploading {local_path} -> {path_in_repo}")
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=path_in_repo,
            repo_id=REPO_ID,
            repo_type="model",
        )

    # Also upload results CSV if present
    for results_dir in RESULTS_DIRS:
        csv_path = os.path.join(results_dir, "results.csv")
        if os.path.exists(csv_path):
            remote_name = results_dir.replace("/", "_") + "_results.csv"
            print(f"Uploading {csv_path} -> {remote_name}")
            api.upload_file(
                path_or_fileobj=csv_path,
                path_in_repo=remote_name,
                repo_id=REPO_ID,
                repo_type="model",
            )

    print("Done. Checkpoints uploaded to", f"https://huggingface.co/{REPO_ID}")


if __name__ == "__main__":
    main()
