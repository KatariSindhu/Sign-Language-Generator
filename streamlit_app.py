import streamlit as st
import whisper
import os
import re
import subprocess
import json
import requests
from moviepy.editor import VideoFileClip, ColorClip
from stqdm import stqdm
import time
import cv2

# --- UI Configuration ---
st.set_page_config(page_title="Sign Language Generator", layout="wide")
st.title("Sign Language Generator")
st.markdown(
    "This application converts a YouTube video's audio into a complete American Sign Language (ASL) pose-animated video — "
    "with clear captions and transcript display."
)

# --- API Key Input ---
st.sidebar.header("Configuration")
api_key = st.sidebar.text_input(
    "Enter your Google AI API Key:",
    type="password",
    help="Get your key from https://ai.google.dev/"
)

# --- Constants ---
PROCESSED_DATASET_DIR = "processed_pose_dataset"
OUTPUT_VIDEO_DIR = "output_videos"
PAUSE_DURATION_SECONDS = 0.5
PAUSE_CLIP_PATH = os.path.join(PROCESSED_DATASET_DIR, "pause.mp4")

os.makedirs(OUTPUT_VIDEO_DIR, exist_ok=True)

# --- Helper Functions ---

@st.cache_data
def download_audio(url):
    """Downloads audio from YouTube using yt-dlp."""
    output_path = "downloads"
    os.makedirs(output_path, exist_ok=True)
    fixed_output_filename = os.path.join(output_path, "downloaded_audio.mp3")

    try:
        if os.path.exists(fixed_output_filename):
            os.remove(fixed_output_filename)
        command = ["yt-dlp", url, "-x", "--audio-format", "mp3", "--audio-quality", "0", "-o", fixed_output_filename]
        subprocess.run(command, check=True, capture_output=True, text=True)
        if os.path.exists(fixed_output_filename):
            st.success("Audio downloaded successfully!")
            return fixed_output_filename
    except subprocess.CalledProcessError as e:
        st.error("Error downloading audio.")
        st.code(e.stderr)
        return None


@st.cache_data
def load_whisper_model():
    """Loads Whisper model once."""
    st.info("Loading Whisper model...")
    model = whisper.load_model("base")
    st.success("Whisper model loaded successfully!")
    return model


def transcribe_audio(model, audio_path):
    """Transcribes YouTube audio to text."""
    try:
        st.info("Transcribing audio...")
        result = model.transcribe(audio_path, fp16=False)
        st.success("Transcription complete!")
        return result['text']
    except Exception as e:
        st.error(f"Error during transcription: {e}")
        return None


def generate_asl_gloss(transcript, key):
    """Converts English transcript to ASL gloss using Gemini API."""
    st.info("Generating ASL Gloss using Gemini API...")
    if not key:
        st.error("API Key not provided.")
        return None

    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key={key}"
    system_prompt = (
        "You are an ASL linguist. Convert English text to ASL gloss (uppercase, Time-Topic-Comment, fs- for fingerspelling). "
        "Return JSON with key 'glossed_sentences'."
    )
    user_prompt = f"Here is the transcript: \"{transcript}\""
    payload = {
        "contents": [{"parts": [{"text": user_prompt}]}],
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "generationConfig": {"responseMimeType": "application/json"}
    }

    try:
        response = requests.post(api_url, headers={'Content-Type': 'application/json'}, json=payload, timeout=300)
        response.raise_for_status()
        gloss_text = response.json()['candidates'][0]['content']['parts'][0]['text']
        gloss_data = json.loads(gloss_text)
        if 'glossed_sentences' in gloss_data:
            st.success("ASL Gloss generation complete!")
            return gloss_data['glossed_sentences']
    except Exception as e:
        st.error(f"Gloss generation failed: {e}")
        return None

def process_glossed_sentences(sentence_list):
    """Tokenizes gloss sentences into words and pauses."""
    all_tokens = []
    for i, sentence in enumerate(sentence_list):
        tokens = re.findall(r"[A-Z0-9-]+", sentence)
        all_tokens.extend(tokens)
        if i < len(sentence_list) - 1:
            all_tokens.append("<PAUSE>")
    return all_tokens

@st.cache_data
def get_or_create_pause_clip():
    """Creates a black pause clip if missing."""
    if os.path.exists(PAUSE_CLIP_PATH):
        return PAUSE_CLIP_PATH
    try:
        ref_path = os.path.join(PROCESSED_DATASET_DIR, "a.mp4")
        ref_clip = VideoFileClip(ref_path)
        w, h = ref_clip.size
        pause_clip = ColorClip(size=(w, h), color=(0, 0, 0), duration=PAUSE_DURATION_SECONDS)
        pause_clip.write_videofile(PAUSE_CLIP_PATH, fps=30, codec="libx264")
        return PAUSE_CLIP_PATH
    except Exception as e:
        st.error(f"Error creating pause clip: {e}")
        return None

def get_video_clip_path(token, missing_assets_list, pause_clip_path):
    """Gets correct clip path for each token."""
    if token == "<PAUSE>":
        return pause_clip_path
    if token.upper().startswith("FS-"):
        return fingerspell_word(token, missing_assets_list, pause_clip_path)
    video_path = os.path.join(PROCESSED_DATASET_DIR, f"{token.lower()}.mp4")
    if os.path.exists(video_path):
        return video_path
    else:
        missing_assets_list.append(f"WORD-{token}")
        return pause_clip_path

def fingerspell_word(word, missing_assets_list, pause_clip_path):
    """Handles fingerspelling by combining alphabet clips."""
    paths = []
    cleaned = re.sub(r"^fs-", "", word).lower()
    for letter in cleaned:
        path = os.path.join(PROCESSED_DATASET_DIR, f"{letter}.mp4")
        if os.path.exists(path):
            paths.append(path)
        else:
            missing_assets_list.append(f"LETTER-{letter.upper()}")
            paths.append(pause_clip_path)
    return paths

def create_sign_language_video(tokens):
    """Creates final ASL video with improved readable captions."""
    st.info("Generating video with captions...")

    pause_clip_path = get_or_create_pause_clip()
    if not pause_clip_path:
        st.error("Pause clip missing.")
        return None, []

    missing_assets = []
    captioned_segments = []

    for token in stqdm(tokens, desc="Processing tokens"):
        clip_path_or_list = get_video_clip_path(token, missing_assets, pause_clip_path)
        if isinstance(clip_path_or_list, list):
            for sub_path in clip_path_or_list:
                captioned_segments.append((sub_path, token.upper()))
        else:
            captioned_segments.append((clip_path_or_list, token.upper()))

    concat_list_path = os.path.join(OUTPUT_VIDEO_DIR, "concat_list.txt")
    with open(concat_list_path, "w", encoding="utf-8") as f:
        for path, _ in captioned_segments:
            abs_path = os.path.abspath(path)
            safe = abs_path.replace("\\", "/").replace("'", "'\\''")
            f.write(f"file '{safe}'\n")

    total_time = 0
    caption_cmds = []

    for i, (path, caption) in enumerate(captioned_segments):
        cap = cv2.VideoCapture(path)
        duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / max(cap.get(cv2.CAP_PROP_FPS), 1)
        cap.release()

        # Avoid showing duplicate pause captions
        if caption == "<PAUSE>" or "pause" in path.lower():
            if i > 0 and (captioned_segments[i - 1][1] == "<PAUSE>" or "pause" in captioned_segments[i - 1][0].lower()):
                display_caption = ""  # skip consecutive pauses
            else:
                display_caption = "<PAUSE>"
        else:
            display_caption = caption

        if display_caption:
            caption_cmds.append(
                f"drawtext=text='{display_caption}':fontcolor=white:fontsize=26:line_spacing=12:"
                f"font='Arial':box=1:boxcolor=black@0.6:boxborderw=4:"
                f"x=(w-text_w)/2:y=h-(text_h*3.5):alpha='if(lt(t,{total_time+0.2}),0,1)':"
                f"enable='between(t,{total_time},{total_time+duration})'"
            )

        total_time += duration

    drawtext_filter = ",".join(caption_cmds)
    output_filename = f"output_captioned_{int(time.time())}.mp4"
    output_path = os.path.join(OUTPUT_VIDEO_DIR, output_filename)

    command = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", concat_list_path,
        "-vf", drawtext_filter,
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", "30",
        output_path
    ]

    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
        st.success("✅ Final video created with captions!")
        return output_path, list(set(missing_assets))
    except subprocess.CalledProcessError as e:
        st.error("Error while generating video with captions.")
        st.code(e.stderr)
        return None, missing_assets

# --- MAIN APP ---

youtube_url = st.text_input("Enter YouTube URL:", placeholder="https://www.youtube.com/watch?v=...")

if st.button("Generate Sign Language Video"):
    if not api_key:
        st.warning("Please enter your Google API Key in the sidebar.")
    elif youtube_url:
        audio_path = download_audio(youtube_url)
        if audio_path:
            model = load_whisper_model()
            transcript = transcribe_audio(model, audio_path)
            os.remove(audio_path)

            if transcript:
                st.subheader("🎙️ Original Transcribed Text")
                st.write(transcript)

                glossed = generate_asl_gloss(transcript, api_key)
                if glossed:
                    tokens = process_glossed_sentences(glossed)
                    video_path, missing = create_sign_language_video(tokens)

                    if video_path:
                        st.header("🎥 Final ASL Pose Video with Captions")

                        # ✅ Centered smaller video display
                        col1, col2, col3 = st.columns([1, 2, 1])
                        with col2:
                            st.video(video_path, format="video/mp4", start_time=0)

                        st.download_button(
                            "⬇️ Download Video",
                            data=open(video_path, "rb"),
                            file_name=os.path.basename(video_path),
                            mime="video/mp4"
                        )

                        st.subheader("🧩 Generated Tokens")
                        with st.expander("See Token Sequence"):
                            st.dataframe(tokens, width=700, height=300)

                        if missing:
                            st.warning("⚠️ Missing Assets Detected:")
                            st.json(missing)
    else:
        st.warning("Please enter a YouTube URL.")