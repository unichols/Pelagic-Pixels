import cv2
import os
from tqdm import tqdm

"""
To Run: use python, not python 3

Easy way: Press run. It will throw an error. Then copy path and run with "python pathName"
"""

# Directory where the turtle images are stored
turtle_dir = "../Models/turtle_images/"

# Load all turtle images by sorting the filenames then loading the images
turtle_files = sorted([file for file in os.listdir(turtle_dir) if file.endswith('.png')])
turtle_images = [cv2.imread(os.path.join(turtle_dir, file), cv2.IMREAD_UNCHANGED) for file in turtle_files]

# Filter out any None images in case there was an issue loading any of them
turtle_images = [img for img in turtle_images if img is not None]

# Create a looping sequence: forward + reverse
turtle_images_extended = turtle_images + turtle_images[::-1]  # Add the reverse sequence for smooth looping

# Extract video frames
video_path = '/pixels/Videos/Blank/day_brancol_cam1_20240105-203500_768px0fps700k.mp4'
cap = cv2.VideoCapture(video_path)

frame_list = []
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_list.append(frame)

cap.release()

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

# Apply species images to frames
output_frames = []
turtle_image_count = len(turtle_images_extended)
for i, frame in enumerate(tqdm(frame_list, desc="Processing Frames")):
    # Cycle through the extended species images (with forward and reverse order for looping)
    turtle_image = turtle_images_extended[i % turtle_image_count]

    if turtle_image is None:
        continue  # Skip if the image didn't load correctly

    # Extract the RGB and alpha channels from the species image
    turtle_alpha = turtle_image[:, :, 3] / 255.0  # Alpha channel
    turtle_rgb = turtle_image[:, :, :3]  # RGB channels

    # Fixed position for the species (adjust as necessary)
    turtle_pos = (300, 570)

    # Overlay the species onto the frame
    frame_with_turtle = overlay_image(frame, turtle_rgb, turtle_pos, turtle_alpha)
    output_frames.append(frame_with_turtle)

# Save the frames as a video
height, width, layers = frame_list[0].shape
output_video_path = '/pixels/Videos/Turtle/NewVidWithTurtle2.mp4'
out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))

# Save the processed frames with a progress bar
for frame in tqdm(output_frames, desc="Saving Video"):
    out.write(frame)

out.release()
