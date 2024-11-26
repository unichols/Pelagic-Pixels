import cv2
import os
import numpy as np
from tqdm import tqdm
import random

# Directory where the turtle images are stored
turtle_dir = "../Models/TurtleSide"
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

def rotate_image(image, angle):
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_rgb = cv2.warpAffine(image[:, :, :3], rotation_matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    rotated_alpha = cv2.warpAffine(image[:, :, 3], rotation_matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return np.dstack((rotated_rgb, rotated_alpha))

def apply_color_filter(image, filter_color):
    tinted_overlay = np.full_like(image[:, :, :3], filter_color, dtype=np.uint8)
    blended_image = cv2.addWeighted(image[:, :, :3], 0.5, tinted_overlay, 0.5, 0)
    return np.dstack((blended_image, image[:, :, 3]))

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
    alpha = alpha_mask / 255.0
    for c in range(3):
        roi[:, :, c] = (roi[:, :, c] * (1 - alpha) + foreground[:, :, c] * alpha)
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

# Total number of images to generate
total_images = 1000
output_dir = "../Images/Turtle"
gt_output_dir = "../Images/gt"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(gt_output_dir, exist_ok=True)

# Generate unique images
for image_idx in tqdm(range(total_images), desc="Generating Images"):
    # Select a random frame from the video
    background = random.choice(frame_list).copy()

    # Initialize a single turtle for this image
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
    angle = random.uniform(-10, 10)
    turtle_rotated = rotate_image(filtered_turtle, angle)
    turtle_alpha_rotated = turtle_rotated[:, :, 3]

    # Overlay the turtle image onto the frame
    final_image = overlay_image(background, turtle_rotated[:, :, :3], (pos_x, pos_y), turtle_alpha_rotated)

    # Save the final image
    output_image_name = f"TurtleImageNd_{image_idx + 1:04d}.png"
    output_image_path = os.path.join(output_dir, output_image_name)
    cv2.imwrite(output_image_path, final_image)

    # Calculate normalized ground truth values
    x_center = (pos_x + random_size / 2) / frame_list[0].shape[1]
    y_center = (pos_y + random_size / 2) / frame_list[0].shape[0]
    width = random_size / frame_list[0].shape[1]
    height = random_size / frame_list[0].shape[0]
    class_id = 0  # Assuming turtle is the only class

    # Write ground truth data to a separate file
    gt_file_path = os.path.join(gt_output_dir, f"TurtleImageNd_{image_idx + 1:04d}.txt")
    with open(gt_file_path, "w") as gt_file:
        gt_file.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
