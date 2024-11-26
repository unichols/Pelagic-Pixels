import cv2
import os
import numpy as np
from tqdm import tqdm
import random
import torch
from sam2.sam2.build_sam  import build_sam2
from sam2.sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.sam2.sam2_video_predictor import SAM2VideoPredictor

""" FOR PYTHON 3.11 AND EARLIER
For compilers to find libomp you may need to set:
  export LDFLAGS="-L/usr/local/opt/libomp/lib"
  export CPPFLAGS="-I/usr/local/opt/libomp/include"pyt
    Use a Virtual Environment:
         source myenv/bin/activate
    Upgrade pip within the virtual environment:
        python -m pip install --upgrade pip
    Install required libraries:
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
        pip install tqdm
        pip install opencv-python
        cd sam2/checkpoints
        ./download_ckpts.sh
        pip install "numpy<2.0"
    Run your scripts
"""
# Load the SAM2 model
checkpoint = "./sam2/checkpoints/sam2.1_hiera_large.pt"
model_cfg = "./sam2/configs/sam2.1/sam2.1_hiera_l.yaml"
sam_model = build_sam2(model_cfg, checkpoint)
sam_predictor = SAM2ImagePredictor(sam_model)

# Directory where the turtle images are stored
turtle_dir = "../Models/TurtleDeck/"
folder_name = os.path.basename(turtle_dir.rstrip('/'))

# Determine if model is "side" or "deck"
turtle_type = "side" if "Side" in folder_name else "deck"

# Load all turtle images by sorting the filenames then loading the images
turtle_files = sorted([file for file in os.listdir(turtle_dir) if file.endswith('.png')])

# Load and rotate turtle images
turtle_images = []
for file in turtle_files:
    img = cv2.imread(os.path.join(turtle_dir, file), cv2.IMREAD_UNCHANGED)
    if img is not None:
        img_rotated = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)  # Rotate by 90 degrees
        turtle_images.append(img_rotated)

# Determine the size range based on turtle type
if turtle_type == "deck":
    min_size, max_size = 100, 300  # Size range for deck models
else:
    min_size, max_size = 150, 300  # Size range for side models

# Randomize the size once for the entire script run
random_size = random.randint(min_size, max_size)

# Resize turtle images to the single randomized size
turtle_images_resized = []
for img in turtle_images:
    if img is not None:
        resized_img = cv2.resize(img, (random_size, random_size), interpolation=cv2.INTER_AREA)
        turtle_images_resized.append(resized_img)

# Create a looping sequence: forward + reverse
turtle_images_extended = turtle_images_resized + turtle_images_resized[::-1]

# Extract video frames
video_path = '../Videos/Blanks/NightFootage.mp4'
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Error: Could not open video file {video_path}")
    exit(1)

frame_list = []
while len(frame_list) < 20:  # Only capture 20 frames for the output
    ret, frame = cap.read()
    if not ret:
        break
    frame_list.append(frame)

cap.release()

# Check if frames were read successfully
if len(frame_list) == 0:
    print("Error: No frames were read from the video.")
    exit(1)

# Use SAM2 to segment the video frames and identify regions
boat_regions = []
ocean_regions = []

for i, frame in enumerate(tqdm(frame_list, desc="Segmenting Frames")):
    sam_predictor.set_image(frame)
    # Define points or prompts for segmentation (you may need to refine this)
    masks_boat = sam_predictor.predict(point_coords=[[50, 50]], point_labels=[1])  # Example prompt for boat
    masks_ocean = sam_predictor.predict(point_coords=[[200, 200]], point_labels=[0])  # Example prompt for ocean

    # Save regions for later use
    boat_regions.append(masks_boat)
    ocean_regions.append(masks_ocean)

# Randomize position based on turtle type and SAM2 segmentation
if turtle_type == "deck":
    pos_x, pos_y = random.choice([(x, y) for mask in boat_regions for x, y in np.argwhere(mask)])
else:  # Side type
    pos_x, pos_y = random.choice([(x, y) for mask in ocean_regions for x, y in np.argwhere(mask)])

# Function to resize and overlay species image
def overlay_image(background, foreground, position, alpha_mask):
    x, y = position
    h, w = background.shape[:2]  # Get background dimensions

    # Resize species frames to fit within the region
    foreground_resized = cv2.resize(foreground, (w // 4, h // 4))
    alpha_mask_resized = cv2.resize(alpha_mask, (w // 4, h // 4))

    # Define the region of interest on the background
    turtle_h, turtle_w = foreground_resized.shape[:2]
    roi = background[y:y+turtle_h, x:x+turtle_w]

    # Blend the species image into the ROI using the alpha mask
    for c in range(0, 3):  # For each color channel
        roi[:, :, c] = roi[:, :, c] * (1 - alpha_mask_resized) + foreground_resized[:, :, c] * alpha_mask_resized

    background[y:y+turtle_h, x:x+turtle_w] = roi  # Place the blended image back into the background
    return background

# Function to rotate an image by a given angle
def rotate_image(image, angle):
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    return rotated_image

# Apply species images to frames
output_frames = []
turtle_image_count = len(turtle_images_extended)
for i, frame in enumerate(tqdm(frame_list, desc="Processing Frames")):
    # Cycle through the extended species images
    turtle_image = turtle_images_extended[i % turtle_image_count]

    if turtle_image is None:
        continue  # Skip if the image didn't load correctly

    # Rotate the turtle image by a predefined angle for all frames
    angle = random.uniform(0, 360)
    turtle_rotated = rotate_image(turtle_image[:, :, :3], angle)  # Rotate RGB channels only
    turtle_alpha_rotated = rotate_image(turtle_image[:, :, 3] / 255.0, angle)  # Rotate alpha channel

    # Overlay the rotated turtle image onto the frame at the determined position
    frame_with_turtle = overlay_image(frame, turtle_rotated, (pos_x, pos_y), turtle_alpha_rotated)
    output_frames.append(frame_with_turtle)

# Save the frames as a video
height, width, layers = frame_list[0].shape
output_video_path = '../Videos/Turtle/TurtleV_SAM2.mp4'
out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))

# Save the processed frames with a progress bar
for frame in tqdm(output_frames, desc="Saving Video"):
    out.write(frame)

out.release()
