import cv2
import os
import numpy as np
from tqdm import tqdm
import random

# Directory where the shark images are stored
shark_dir = "../Models/SharkDeck/"
folder_name = os.path.basename(shark_dir.rstrip('/'))

# Determine if model is "side" or "deck"
shark_type = "side" if "Side" in folder_name else "deck"

# Load all shark images by sorting the filenames then loading the images
shark_files = sorted([file for file in os.listdir(shark_dir) if file.endswith('.png')])

# Load, rotate, and filter out None images in case there was an issue loading any of them
shark_images = []
for file in shark_files:
    img = cv2.imread(os.path.join(shark_dir, file), cv2.IMREAD_UNCHANGED)
    if img is not None:
        img_rotated = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)  # Rotate by 90 degrees
        shark_images.append(img_rotated)

# Determine the size range based on shark type
if shark_type == "deck":
    min_size, max_size = 100, 300  # Size range for deck models
else:
    min_size, max_size = 150, 300  # Size range for side models

# Randomize the size once for the entire script run
random_size = random.randint(min_size, max_size)

# Resize shark images to the single randomized size
shark_images_resized = []
for img in shark_images:
    if img is not None:
        resized_img = cv2.resize(img, (random_size, random_size), interpolation=cv2.INTER_AREA)
        shark_images_resized.append(resized_img)

# Create a looping sequence: forward + reverse
shark_images_extended = shark_images_resized + shark_images_resized[::-1]

# Determine the rotation angle for 'deck' or initialize variable
angle = random.uniform(0, 360)
# Function to rotate an image by a given angle
def rotate_image(image, angle):
    h, w = image.shape[:2]
    center = (w // 2, h // 2)

    # Check if shark type is 'side', if so, randomize angle for each image
    if shark_type == "Deck":
        angle = random.uniform(0, 360)  # Random angle for side images

    # Apply rotation
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

    return rotated_image

# Rotate the images based on type
shark_images_rotated = [rotate_image(img, angle) for img in shark_images_extended]

# Define a single color filter for all frames (randomly generated once per script run)
filter_color = (random.randint(60, 160), random.randint(30, 100), random.randint(20, 70))

# Function to apply the color filter consistently across all frames
def apply_color_filter(image, filter_color):
    # Create an overlay with the chosen color filter
    tinted_overlay = np.full_like(image[:, :, :3], filter_color, dtype=np.uint8)

    # Blend the original image with the tinted overlay
    blended_image = cv2.addWeighted(image[:, :, :3], 0.5, tinted_overlay, 0.5, 0)

    # Preserve the alpha channel
    tinted_image = np.dstack((blended_image, image[:, :, 3])).astype(np.uint8)
    return tinted_image

# Apply the color filter once to each shark image
shark_images_filtered = [apply_color_filter(img, filter_color) for img in shark_images_rotated]

# Function to overlay the shark image onto the background
def overlay_image(background, foreground, position, alpha_mask):
    x, y = position
    h, w = background.shape[:2]  # Get background dimensions

    # Ensure overlay fits within background dimensions
    shark_h, shark_w = foreground.shape[:2]
    if y + shark_h > h or x + shark_w > w:
        shark_h = min(shark_h, h - y)
        shark_w = min(shark_w, w - x)
        foreground = foreground[:shark_h, :shark_w]
        alpha_mask = alpha_mask[:shark_h, :shark_w]

    # Define the region of interest on the background
    roi = background[y:y+shark_h, x:x+shark_w]

    # Blend the shark image into the ROI using the alpha mask
    for c in range(0, 3):  # For each color channel
        roi[:, :, c] = (roi[:, :, c] * (1 - alpha_mask) +
                        foreground[:, :, c] * alpha_mask)

    background[y:y+shark_h, x:x+shark_w] = roi  # Place the blended image back into the background
    return background

# Extract video frames
video_path = '../Videos/Blanks/day_brancol_cam1_20240105-203500_768px0fps700k.mp4'
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

# Randomize position based on shark type
if shark_type == "deck":
    pos_x = random.randint(200, 500)
    pos_y = random.randint(200, 500)
elif shark_type == "side":
    pos_x = random.randint(100, 700)
    pos_y = random.randint(600, 600)
# Apply species images to frames with random rotation angles
output_frames = []
shark_image_count = len(shark_images_filtered)
for i, frame in enumerate(tqdm(frame_list, desc="Processing Frames")):
    # Cycle through the extended species images
    shark_image = shark_images_filtered[i % shark_image_count]

    if shark_image is None:
        continue  # Skip if the image didn't load correctly

    # Rotate the shark image by a random angle for each frame
    angle = random.uniform(0, 0)  # Random angle between -10 and 10 degrees
    shark_rotated = rotate_image(shark_image[:, :, :3], angle)  # Rotate RGB channels only
    shark_alpha_rotated = rotate_image(shark_image[:, :, 3] / 255.0, angle)  # Rotate alpha channel

    # Overlay the rotated shark image onto the frame at the determined position
    frame_with_shark = overlay_image(frame, shark_rotated, (pos_x, pos_y), shark_alpha_rotated)
    output_frames.append(frame_with_shark)

# Save the frames as a video
height, width, layers = frame_list[0].shape
output_video_path = '../Videos/Shark/SharkRandV_.mp4'
out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))

# Save the processed frames with a progress bar
for frame in tqdm(output_frames, desc="Saving Video"):
    out.write(frame)

out.release()
