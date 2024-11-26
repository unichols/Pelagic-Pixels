import cv2
import numpy as np
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import matplotlib.pyplot as plt
import os

# Utility Functions
def mask_to_bbox(mask):
    coords = np.argwhere(mask > 0)  # Find non-zero (object) pixels
    x_min, y_min = coords.min(axis=0)  # Top-left corner
    x_max, y_max = coords.max(axis=0)  # Bottom-right corner
    return x_min, y_min, x_max, y_max

def bbox_to_yolo_format(bbox, image_width, image_height):
    x_min, y_min, x_max, y_max = bbox
    x_center = (x_min + x_max) / 2 / image_width
    y_center = (y_min + y_max) / 2 / image_height
    box_width = (x_max - x_min) / image_width
    box_height = (y_max - y_min) / image_height
    return x_center, y_center, box_width, box_height

def save_yolo_annotation(output_path, yolo_bbox, class_id=0):
    with open(output_path, "w") as f:
        x_center, y_center, box_width, box_height = yolo_bbox
        f.write(f"{class_id} {x_center} {y_center} {box_width} {box_height}\n")

# Load the SAM model
model_type = "vit_b"  # Use the ViT-B model (smaller and faster)
checkpoint_path = "../Weights/sam_vit_b_01ec64.pth"  # Path to model weights
sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
mask_generator = SamAutomaticMaskGenerator(sam)

# Ensure directories exist
image_folder = "../dataset/test/images"  # Input image folder
label_folder = "../dataset/test/labels"  # Output label folder
os.makedirs(label_folder, exist_ok=True)

# Process all images in the folder
for image_name in os.listdir(image_folder):
    if image_name.endswith((".jpg", ".png")):
        image_path = os.path.join(image_folder, image_name)
        image = cv2.imread(image_path)

        if image is None:
            print(f"Error loading image: {image_name}")
            continue

        # Generate masks
        masks = mask_generator.generate(image)

        if masks:
            # Find the largest mask
            largest_mask = max(masks, key=lambda x: np.sum(x["segmentation"]))
            mask_image = largest_mask["segmentation"].astype("uint8") * 255  # Convert to binary

            # Convert mask to bounding box
            bbox = mask_to_bbox(mask_image)
            height, width = mask_image.shape[:2]
            yolo_bbox = bbox_to_yolo_format(bbox, width, height)

            # Save YOLO annotation
            label_path = os.path.join(label_folder, f"{os.path.splitext(image_name)[0]}.txt")
            save_yolo_annotation(label_path, yolo_bbox)

            print(f"Processed: {image_name}, Bounding Box: {bbox}, YOLO: {yolo_bbox}")
        else:
            print(f"No masks generated for {image_name}")
