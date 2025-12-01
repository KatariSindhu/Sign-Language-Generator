import cv2
import mediapipe as mp
import numpy as np
import os
from tqdm import tqdm
from PIL import Image # We need the Pillow library to handle GIFs

# --- Configuration ---
RAW_ALPHABET_DIR = os.path.join("raw_video_dataset", "alphabet")
PROCESSED_DATASET_DIR = "processed_pose_dataset"
VIDEO_DURATION_SECONDS = 1 # Each static letter video will be 1 second long
FPS = 30 # Standard frames per second

# Ensure the output directory exists
os.makedirs(PROCESSED_DATASET_DIR, exist_ok=True)

# Initialize MediaPipe components
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=True, model_complexity=1, min_detection_confidence=0.5)

# --- Processing Functions ---

def render_pose_on_black_background(image, pose_results):
    """Draws the detected pose onto a black canvas."""
    black_background = np.zeros_like(image)
    if pose_results.pose_landmarks:
        mp_drawing.draw_landmarks(
            black_background,
            pose_results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
        )
    return black_background

def process_static_image(image_path, output_path):
    """Converts a single static image into a short video of a static pose."""
    try:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not read image {image_path}")
            return
            
        h, w, _ = image.shape
        
        # Process the image with MediaPipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        
        # Render the pose
        rendered_frame = render_pose_on_black_background(image, results)
        
        # Create a video from this single rendered frame
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, FPS, (w, h))

        # Write the same frame multiple times to create a video of a certain duration
        for _ in range(int(VIDEO_DURATION_SECONDS * FPS)):
            out.write(rendered_frame)
            
        out.release()
    except Exception as e:
        print(f"An error occurred while processing image {image_path}: {e}")

def process_gif(gif_path, output_path):
    """Converts a GIF into a pose-animated video."""
    try:
        gif = Image.open(gif_path)
        w, h = gif.size
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, FPS, (w, h))

        # Iterate over each frame of the GIF
        for frame_index in range(gif.n_frames):
            gif.seek(frame_index)
            # Convert PIL image frame to OpenCV format (RGB)
            frame_pil = gif.convert("RGB")
            frame_cv2_rgb = np.array(frame_pil)
            # Convert RGB to BGR for OpenCV processing
            frame_cv2_bgr = cv2.cvtColor(frame_cv2_rgb, cv2.COLOR_RGB2BGR)
            
            # Process the frame
            results = pose.process(frame_cv2_rgb)
            rendered_frame = render_pose_on_black_background(frame_cv2_bgr, results)
            
            out.write(rendered_frame)
            
        out.release()
    except Exception as e:
        print(f"An error occurred while processing GIF {gif_path}: {e}")

# --- Script Execution ---

if __name__ == "__main__":
    alphabet_files = os.listdir(RAW_ALPHABET_DIR)
    
    # Install Pillow if not already installed
    try:
        import PIL
    except ImportError:
        print("Pillow library not found. Installing...")
        import sys
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "Pillow"])

    print("Starting alphabet asset processing...")

    for filename in tqdm(alphabet_files, desc="Processing Alphabet"):
        file_path = os.path.join(RAW_ALPHABET_DIR, filename)
        
        # Get the letter (e.g., 'a' from 'a.jpg')
        letter = os.path.splitext(filename)[0]
        output_video_path = os.path.join(PROCESSED_DATASET_DIR, f"{letter}.mp4")

        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            process_static_image(file_path, output_video_path)
        elif filename.lower().endswith('.gif'):
            process_gif(file_path, output_video_path)

    pose.close()
    print("Alphabet processing complete!")


### How to Run the Alphabet Converter

#1.  **Run the script** from your terminal (make sure your `(venv)` is active):
#    ```bash
#    python process_alphabet.py
    
