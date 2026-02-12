
from huggingface_hub import hf_hub_download
from pathlib import Path
import zipfile

repo_id = "Jiazuo98/Finers-4k-benchmark"

# where you want to keep the big image files
root = Path("/scratch/svaidy33/hf_cache/datasets")
root.mkdir(parents=True, exist_ok=True)

# download the 18GB zip with all images
zip_path = hf_hub_download(
    repo_id=repo_id,
    repo_type="dataset",
    filename="all_images.zip",
    local_dir=root,               # optional, keeps it under root
    local_dir_use_symlinks=False  # real file, not symlink
)

print("Zip is at:", zip_path)

# extract images
images_dir = root / "all_images"
images_dir.mkdir(exist_ok=True)

with zipfile.ZipFile(zip_path, "r") as z:
    z.extractall(images_dir)

print("Images extracted to:", images_dir)

