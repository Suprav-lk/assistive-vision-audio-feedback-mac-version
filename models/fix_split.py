import os
import shutil
import random

# Move staircase train images into val and test splits
STAIR_TRAIN_IMG = "staircase-1/train/images"
STAIR_TRAIN_LBL = "staircase-1/train/labels"

OUTPUT_DIR = "custom_dataset"

# Get all staircase image files
images = [f for f in os.listdir(STAIR_TRAIN_IMG) if f.endswith(('.jpg', '.jpeg', '.png'))]
random.seed(42)
random.shuffle(images)

total = len(images)
val_count  = int(total * 0.2)
test_count = int(total * 0.1)

val_images  = images[:val_count]
test_images = images[val_count:val_count + test_count]

def copy_pair(filename, split):
    name, _ = os.path.splitext(filename)
    # Copy image
    src_img = os.path.join(STAIR_TRAIN_IMG, filename)
    dst_img = os.path.join(OUTPUT_DIR, "images", split, f"stair_{filename}")
    shutil.copy2(src_img, dst_img)
    # Copy label
    src_lbl = os.path.join(STAIR_TRAIN_LBL, name + ".txt")
    dst_lbl = os.path.join(OUTPUT_DIR, "labels", split, f"stair_{name}.txt")
    if os.path.exists(src_lbl):
        shutil.copy2(src_lbl, dst_lbl)

for f in val_images:
    copy_pair(f, "valid")
for f in test_images:
    copy_pair(f, "test")

print(f"Done! Val: {len(val_images)}, Test: {len(test_images)}")