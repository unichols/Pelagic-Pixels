import cv2
import os
import numpy as np
from tqdm import tqdm
import random

# Directory where the turtle images are stored
turtle_dir = "../Models/TurtleDeck"
folder_name = os.path.basename(turtle_dir.rstrip('/'))

# Determine if model is "side" or "deck"
turtle_type = "side" if "Side" in folder_name else "deck"

# Load all turtle images
turtle_files = sorted([file for file in os.listdir(turtle_dir) if file.endswith('.png')])
turtle_images = [cv2.imread(os.path.join(turtle_dir, file), cv2.IMREAD_UNCHANGED) for file in turtle_files]
turtle_images = [img for img in turtle_images if img is not None]

# Determine the size range based on turtle type
if turtle_type == "deck":
    min_size, max_size = 100, 300  # Size range for deck models
else:
    min_size, max_size = 150, 300  # Size range for side models

# Function to rotate an image by a given angle
def rotate_image(image, angle):
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    return rotated_image

# Function to apply a color filter to the turtle
def apply_color_filter(image, filter_color):
    tinted_overlay = np.full_like(image[:, :, :3], filter_color, dtype=np.uint8)
    blended_image = cv2.addWeighted(image[:, :, :3], 0.5, tinted_overlay, 0.5, 0)
    tinted_image = np.dstack((blended_image, image[:, :, 3])).astype(np.uint8)
    return tinted_image

# Function to overlay the turtle image onto the background
def overlay_image(background, foreground, position, alpha_mask):
    x, y = position
    h, w = background.shape[:2]
    turtle_h, turtle_w = foreground.shape[:2]
    if y + turtle_h > h or x + turtle_w > w:
        turtle_h = min(turtle_h, h - y)
        turtle_w = min(turtle_w, w - x)
        foreground = foreground[:turtle_h, :turtle_w]
        alpha_mask = alpha_mask[:turtle_h, :turtle_w]
    roi = background[y:y + turtle_h, x:x + turtle_w]
    for c in range(0, 3):  # For each color channel
        roi[:, :, c] = (roi[:, :, c] * (1 - alpha_mask) +
                        foreground[:, :, c] * alpha_mask)
    background[y:y + turtle_h, x:x + turtle_w] = roi
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

if len(frame_list) == 0:
    print("Error: No frames were read from the video.")
    exit(1)

# Generate 50 separate 20-frame videos
for video_idx in range(50):
    print(f"Generating Video {video_idx + 1}/50")

    # Initialize a single turtle for this video
    selected_turtle_image = random.choice(turtle_images)

    # Randomize size, color filter, and position
    random_size = random.randint(min_size, max_size)
    filter_color = (random.randint(60, 160), random.randint(30, 100), random.randint(20, 70))
    if turtle_type == "deck":
        pos_x = random.randint(750, 800)
        pos_y = random.randint(800, 900)
    elif turtle_type == "side":
        pos_x = random.randint(900, 1500)
        pos_y = random.randint(800, 1100)

    # Process the turtle image
    resized_turtle = cv2.resize(selected_turtle_image, (random_size, random_size), interpolation=cv2.INTER_AREA)
    filtered_turtle = apply_color_filter(resized_turtle, filter_color)

    # Prepare frames for the current video
    output_frames = []
    for frame in frame_list:
        angle = random.uniform(-10, 10)  # Slight random rotation per frame
        turtle_rotated = rotate_image(filtered_turtle[:, :, :3], angle)
        turtle_alpha_rotated = rotate_image(filtered_turtle[:, :, 3] / 255.0, angle)

        # Overlay the turtle image onto the frame
        frame_with_turtle = overlay_image(frame.copy(), turtle_rotated, (pos_x, pos_y), turtle_alpha_rotated)
        output_frames.append(frame_with_turtle)

    # Save the video
    height, width, layers = frame_list[0].shape
    output_video_path = f'../Videos/Turtle/TurtleRandV_{video_idx + 1}.mp4'
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))

    for frame in tqdm(output_frames, desc=f"Saving Video {video_idx + 1}"):
        out.write(frame)
    out.release()

    # Clear the current turtle before moving to the next video
    del selected_turtle_image, resized_turtle, filtered_turtle
