import cv2
import os
import numpy as np
from tqdm import tqdm
import random
# Directory where the ray images are stored
ray_dir = "../Models/RaySide/"
folder_name = os.path.basename(ray_dir.rstrip('/'))
# Determine ray position based on folder name
if "Side" in folder_name:
    ray_pos = (300, 500)  # Example position for the side view
elif "Deck" in folder_name:
    ray_pos = (300, 400)  # Example position for the deck view
else:
    ray_pos = (300, 570)  # Default position if no specific view is specified
# Load all ray images by sorting the filenames then loading the images
ray_files = sorted([file for file in os.listdir(ray_dir) if file.endswith('.png')])
ray_images = [cv2.imread(os.path.join(ray_dir, file), cv2.IMREAD_UNCHANGED) for file in ray_files]
# Filter out any None images in case there was an issue loading any of them
ray_images = [img for img in ray_images if img is not None]
# Define a fixed size for the ray images (adjust as needed)
fixed_ray_width = 400  # Desired width of the ray image
fixed_ray_height = 400  # Desired height of the ray image
# Resize all ray images to the fixed size
ray_images_resized = []
for img in ray_images:
    if img is not None:
        resized_img = cv2.resize(img, (fixed_ray_width, fixed_ray_height), interpolation=cv2.INTER_AREA)
        ray_images_resized.append(resized_img)
# Create a looping sequence: forward + reverse
ray_images_extended = ray_images_resized + ray_images_resized[::-1]  # Add the reverse sequence for smooth looping
# Function to rotate an image by a given angle
def rotate_image(image, angle):
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    return rotated_image
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
# Function to overlay the ray image at a fixed size with rotation
def overlay_image(background, foreground, position, alpha_mask):
    x, y = position
    h, w = background.shape[:2]  # Get background dimensions
    # Ensure overlay fits within background dimensions
    ray_h, ray_w = foreground.shape[:2]
    if y + ray_h > h or x + ray_w > w:
        ray_h = min(ray_h, h - y)
        ray_w = min(ray_w, w - x)
        foreground = foreground[:ray_h, :ray_w]
        alpha_mask = alpha_mask[:ray_h, :ray_w]
    # Define the region of interest on the background
    roi = background[y:y+ray_h, x:x+ray_w]
    # Blend the ray image into the ROI using the alpha mask
    for c in range(0, 3):  # For each color channel
        roi[:, :, c] = (roi[:, :, c] * (1 - alpha_mask) +
                        foreground[:, :, c] * alpha_mask)
    background[y:y+ray_h, x:x+ray_w] = roi  # Place the blended image back into the background
    return background
# Apply species images to frames with random rotation angles
output_frames = []
ray_image_count = len(ray_images_extended)
for i, frame in enumerate(tqdm(frame_list, desc="Processing Frames")):
    # Cycle through the extended species images (with forward and reverse order for looping)
    ray_image = ray_images_extended[i % ray_image_count]
    if ray_image is None:
        continue  # Skip if the image didn't load correctly
    # Rotate the ray image by a random angle for each frame
    angle = random.uniform(-20, 20)  # Random angle between -10 and 10 degrees
    ray_rotated = rotate_image(ray_image[:, :, :3], angle)  # Rotate RGB channels only
    ray_alpha_rotated = rotate_image(ray_image[:, :, 3] / 255.0, angle)  # Rotate alpha channel
    # Overlay the rotated species onto the frame
    frame_with_ray = overlay_image(frame, ray_rotated, ray_pos, ray_alpha_rotated)
    output_frames.append(frame_with_ray)
# Save the frames as a video
height, width, layers = frame_list[0].shape
output_video_path = '../Videos/Ray/RayRandV_.mp4'
out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))
# Save the processed frames with a progress bar
for frame in tqdm(output_frames, desc="Saving Video"):
    out.write(frame)
out.release()
