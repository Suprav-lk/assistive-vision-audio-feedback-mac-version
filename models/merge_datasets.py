import os
import shutil

# Source folders
DOOR_DIR = "Door-1"
STAIR_DIR = "staircase-1"
OUTPUT_DIR = "custom_dataset"

# Splits to process
splits = ["train", "valid", "test"]

for split in splits:
    for subfolder in ["images", "labels"]:
        os.makedirs(f"{OUTPUT_DIR}/{subfolder}/{split}", exist_ok=True)

def copy_files(src_dir, split, subfolder, prefix):
    src_path = os.path.join(src_dir, split, subfolder)
    dst_path = os.path.join(OUTPUT_DIR, subfolder, split)
    if not os.path.exists(src_path):
        print(f"Skipping missing: {src_path}")
        return
    for filename in os.listdir(src_path):
        src_file = os.path.join(src_path, filename)
        name, ext = os.path.splitext(filename)
        dst_file = os.path.join(dst_path, f"{prefix}_{name}{ext}")
        shutil.copy2(src_file, dst_file)

for split in splits:
    copy_files(DOOR_DIR,  split, "images", "door")
    copy_files(DOOR_DIR,  split, "labels", "door")
    copy_files(STAIR_DIR, split, "images", "stair")
    copy_files(STAIR_DIR, split, "labels", "stair")

print("Merge complete!")