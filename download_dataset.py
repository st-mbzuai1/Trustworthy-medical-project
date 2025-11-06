import kagglehub
import shutil
from pathlib import Path

# Specify your desired download location
target_dir = Path("./ham10000_data")
target_dir.mkdir(exist_ok=True)

print(f"Downloading HAM10000 dataset to: {target_dir.absolute()}")

# Download using kagglehub (downloads to cache first)
path = kagglehub.dataset_download("kmader/skin-cancer-mnist-ham10000")
print(f"Downloaded to cache: {path}")

# Move from cache to your target directory
source_path = Path(path)
if source_path.exists():
    print(f"\nMoving files to: {target_dir.absolute()}")
    
    # Copy all files from cache to target directory
    for item in source_path.iterdir():
        dest = target_dir / item.name
        if item.is_file():
            shutil.copy2(item, dest)
            print(f"  ✓ Copied: {item.name}")
        elif item.is_dir():
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(item, dest)
            print(f"  ✓ Copied folder: {item.name}")
    
    print(f"\n Dataset available at: {target_dir.absolute()}")
else:
    print("Download failed!")

# Verify the download
print("\n=== Dataset Contents ===")
for item in sorted(target_dir.rglob("*")):
    if item.is_file():
        size_mb = item.stat().st_size / (1024 * 1024)
        print(f"{item.relative_to(target_dir)}: {size_mb:.2f} MB")