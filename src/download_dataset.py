import joblib
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

# Load the 128x128 dataset
file_128 = "turtles128.sav"

# Load the data into a NumPy array
turtle_images_128 = np.array(joblib.load(file_128))  # 32x32 images
print(f"128x128 dataset size: {turtle_images_128.shape}")

# Check pixel value range and data type
print(f"Data type: {turtle_images_128.dtype}")
print(f"Min value: {turtle_images_128.min()}")
print(f"Max value: {turtle_images_128.max()}")

# Rescale pixel values to 0–255 if needed
if turtle_images_128.max() <= 1.0:
    turtle_images_128 = (turtle_images_128 * 255).astype(np.uint8)

# Create a folder to save 32x32 images
output_folder = "../dataset/test/images_128"
os.makedirs(output_folder, exist_ok=True)

# Save images to the folder
for idx, img in enumerate(turtle_images_128):
    output_path = os.path.join(output_folder, f"turtle_{idx+1}.jpg")
    cv2.imwrite(output_path, img)  # Save as grayscale

print(f"Saved {len(turtle_images_128)} 128x128 images to {output_folder}")

# Display a sample image from the dataset
plt.imshow(turtle_images_128[0], cmap="gray")
plt.title("Sample 128x128 Image")
plt.axis("off")
plt.show()
