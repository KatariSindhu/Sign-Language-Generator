import cv2
import mediapipe as mp
import numpy as np
import json # Changed from pandas
import os
from tqdm import tqdm

# --- Configuration & Constants ---
RAW_DATASET_DIR = "raw_video_dataset"
PROCESSED_DATASET_DIR = "processed_pose_dataset"
# Updated to point to the correct JSON file
METADATA_FILE = os.path.join(RAW_DATASET_DIR, "WLASL_v0.3.json") 

os.makedirs(PROCESSED_DATASET_DIR, exist_ok=True)

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5)

# --- Main Processing Logic ---

def process_video(video_path, output_path):
    """
    Processes a single video file to extract pose landmarks and render them
    onto a new video with a black background.
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file {video_path}")
            return

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # Handle cases where FPS might be 0
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            fps = 30 # Default to 30 fps if not available

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break

            black_background = np.zeros_like(image)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    black_background,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2),
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
                )
            
            out.write(black_background)

        cap.release()
        out.release()

    except Exception as e:
        print(f"An error occurred while processing {video_path}: {e}")

# --- Script Execution ---

if __name__ == "__main__":
    print("Starting dataset processing...")

    # Load the metadata from the JSON file
    try:
        with open(METADATA_FILE, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Metadata file not found at {METADATA_FILE}.")
        exit()

    # To process the full dataset, this line should be commented out.
    # data = data[:50] 

    # Use tqdm for a progress bar
    for entry in tqdm(data, desc="Processing Videos"):
        gloss = entry['gloss']
        
        if not entry['instances']:
            continue
            
        video_id = entry['instances'][0]['video_id']

        source_video_path = os.path.join(RAW_DATASET_DIR, "videos", f"{video_id}.mp4")
        
        output_filename = f"{gloss.lower().replace(' ', '-')}.mp4"
        destination_video_path = os.path.join(PROCESSED_DATASET_DIR, output_filename)

        # --- THE PROFESSIONAL UPGRADE ---
        # If the processed video already exists, skip it to save time.
        if os.path.exists(destination_video_path):
            continue # Move to the next iteration of the loop

        if os.path.exists(source_video_path):
            process_video(source_video_path, destination_video_path)
        else:
            print(f"Warning: Source video not found for gloss '{gloss}' with id '{video_id}'. Skipping.")

    pose.close()
    print("Dataset processing complete!")
    print(f"Processed pose videos are saved in the '{PROCESSED_DATASET_DIR}' directory.")
    
# ```

# **4. Run the Script:**
# Now, open your terminal, activate your virtual environment, and run the command:
# ```bash
# python process_dataset.py

