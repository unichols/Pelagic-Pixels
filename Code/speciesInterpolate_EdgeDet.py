import cv2
import os
import numpy as np
from tqdm import tqdm
import random

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

# Determine a random rotation angle for the 'deck' type once per run
angle = random.uniform(0, 360)

# Function to rotate an image by a given angle
def rotate_image(image, angle):
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    return rotated_image

# Rotate the images based on type
turtle_images_rotated = [rotate_image(img, angle) for img in turtle_images_extended]

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

# Apply the color filter once to each turtle image
turtle_images_filtered = [apply_color_filter(img, filter_color) for img in turtle_images_rotated]

# Function to overlay the turtle image onto the background
def overlay_image(background, foreground, position, alpha_mask):
    x, y = position
    h, w = background.shape[:2]  # Get background dimensions

    # Ensure overlay fits within background dimensions
    turtle_h, turtle_w = foreground.shape[:2]
    if y + turtle_h > h or x + turtle_w > w:
        turtle_h = min(turtle_h, h - y)
        turtle_w = min(turtle_w, w - x)
        foreground = foreground[:turtle_h, :turtle_w]
        alpha_mask = alpha_mask[:turtle_h, :turtle_w]

    # Define the region of interest on the background
    roi = background[y:y+turtle_h, x:x+turtle_w]

    # Blend the turtle image into the ROI using the alpha mask
    for c in range(0, 3):  # For each color channel
        roi[:, :, c] = (roi[:, :, c] * (1 - alpha_mask) +
                        foreground[:, :, c] * alpha_mask)

    background[y:y+turtle_h, x:x+turtle_w] = roi  # Place the blended image back into the background
    return background

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

# Randomize position based on turtle type
if turtle_type == "deck":
    pos_x = random.randint(200, 500)
    pos_y = random.randint(200, 500)
elif turtle_type == "side":
    pos_x = random.randint(100, 700)
    pos_y = random.randint(600, 600)

# Apply species images to frames
output_frames = []
turtle_image_count = len(turtle_images_filtered)
for i, frame in enumerate(tqdm(frame_list, desc="Processing Frames")):
    # Cycle through the extended species images
    turtle_image = turtle_images_filtered[i % turtle_image_count]

    if turtle_image is None:
        continue  # Skip if the image didn't load correctly

    # Rotate the turtle image by a predefined angle for all frames
    turtle_rotated = rotate_image(turtle_image[:, :, :3], angle)  # Rotate RGB channels only
    turtle_alpha_rotated = rotate_image(turtle_image[:, :, 3] / 255.0, angle)  # Rotate alpha channel

    # Overlay the rotated turtle image onto the frame at the determined position
    frame_with_turtle = overlay_image(frame, turtle_rotated, (pos_x, pos_y), turtle_alpha_rotated)
    output_frames.append(frame_with_turtle)

# Save the frames as a video
height, width, layers = frame_list[0].shape
output_video_path = '../Videos/Turtle/TurtleEdgeV_.mp4'
out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))

# Save the processed frames with a progress bar
for frame in tqdm(output_frames, desc="Saving Video"):
    out.write(frame)

out.release()
