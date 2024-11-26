import os
import shutil
from sklearn.model_selection import train_test_split

# Define source and destination folders
source_folder = "../dataset/test/images_128"
train_folder = "../dataset/train/images"
val_folder = "../dataset/val/images"

# Ensure folders exist
os.makedirs(train_folder, exist_ok=True)
os.makedirs(val_folder, exist_ok=True)

# Get all image file paths
image_files = [os.path.join(source_folder, f) for f in os.listdir(source_folder) if f.endswith(".jpg")]

# Split into train (80%) and val (20%)
train_files, val_files = train_test_split(image_files, test_size=0.2, random_state=42)

# Move files to train and val folders
for f in train_files:
    shutil.move(f, os.path.join(train_folder, os.path.basename(f)))

for f in val_files:
    shutil.move(f, os.path.join(val_folder, os.path.basename(f)))

print(f"Moved {len(train_files)} files to train/images and {len(val_files)} files to val/images.")
