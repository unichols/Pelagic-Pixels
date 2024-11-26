import cv2
import os
import numpy as np
from tqdm import tqdm
import random
# Directory where the turtle images are stored
turtle_dir = "../Models/TurtleSide/"
folder_name = os.path.basename(turtle_dir.rstrip('/'))
# Determine turtle position based on folder name
if "Side" in folder_name:
    turtle_pos = (1500, 800)  # Example position for the side view
elif "Deck" in folder_name:
    turtle_pos = (900, 600)  # Example position for the deck view
else:
    turtle_pos = (300, 570)  # Default position if no specific view is specified
# Load all turtle images by sorting the filenames then loading the images
turtle_files = sorted([file for file in os.listdir(turtle_dir) if file.endswith('.png')])
# Load, rotate, and filter out None images in case there was an issue loading any of them
turtle_images = []
for file in turtle_files:
    img = cv2.imread(os.path.join(turtle_dir, file), cv2.IMREAD_UNCHANGED)
    if img is not None:
        # Rotate each image by 90 degrees counterclockwise (adjust if you need clockwise)
        img_rotated = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        turtle_images.append(img_rotated)
# Define a fixed size for the turtle images (adjust as needed)
fixed_turtle_width = 200  # Desired width of the turtle image
fixed_turtle_height = 200  # Desired height of the turtle image
# Resize all turtle images to the fixed size
turtle_images_resized = []
for img in turtle_images:
    if img is not None:
        resized_img = cv2.resize(img, (fixed_turtle_width, fixed_turtle_height), interpolation=cv2.INTER_AREA)
        turtle_images_resized.append(resized_img)
# Create a looping sequence: forward + reverse
turtle_images_extended = turtle_images_resized + turtle_images_resized[::-1]  # Add the reverse sequence for smooth looping
# Function to rotate an image by a given angle
def rotate_image(image, angle):
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    return rotated_image
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
# Function to overlay the turtle image at a fixed size with rotation
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
# Apply species images to frames with random rotation angles
output_frames = []
turtle_image_count = len(turtle_images_extended)
for i, frame in enumerate(tqdm(frame_list, desc="Processing Frames")):
    # Cycle through the extended species images (with forward and reverse order for looping)
    turtle_image = turtle_images_extended[i % turtle_image_count]
    if turtle_image is None:
        continue  # Skip if the image didn't load correctly
    # Rotate the turtle image by a random angle for each frame
    angle = random.uniform(-10, 10)  # Random angle between -10 and 10 degrees
    turtle_rotated = rotate_image(turtle_image[:, :, :3], angle)  # Rotate RGB channels only
    turtle_alpha_rotated = rotate_image(turtle_image[:, :, 3] / 255.0, angle)  # Rotate alpha channel
    # Overlay the rotated species onto the frame
    frame_with_turtle = overlay_image(frame, turtle_rotated, turtle_pos, turtle_alpha_rotated)
    output_frames.append(frame_with_turtle)
# Save the frames as a video
height, width, layers = frame_list[0].shape
output_video_path = '../Videos/Turtle/TurtleV_.mp4'
out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))
# Save the processed frames with a progress bar
for frame in tqdm(output_frames, desc="Saving Video"):
    out.write(frame)
out.release()
