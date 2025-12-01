# Sign Language Generator 🤟📹

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io/)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![OpenAI Whisper](https://img.shields.io/badge/OpenAI-Whisper-412991?style=for-the-badge&logo=openai&logoColor=white)](https://openai.com/research/whisper)
[![Google Gemini](https://img.shields.io/badge/Google%20Gemini-LLM-8E75B2?style=for-the-badge&logo=google&logoColor=white)](https://deepmind.google/technologies/gemini/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Pose-00A8E8?style=for-the-badge&logo=google&logoColor=white)](https://developers.google.com/mediapipe)

> **An end-to-end AI application that automatically translates spoken English from YouTube videos into grammatically correct, pose-animated American Sign Language (ASL) videos.**

This project bridges the digital accessibility gap for the Deaf and Hard of Hearing community by providing native-language translations for online content.

---

## 🚀 Key Features

* **YouTube to ASL:** Takes any YouTube URL as input and generates a downloadable sign language video.
* **AI-Powered Translation:** Uses **Google Gemini LLM** to translate English text into true **ASL Gloss** (Time-Topic-Comment structure), ensuring grammatically correct signing rather than word-for-word translation.
* **High-Accuracy Transcription:** Utilizes **OpenAI Whisper** for robust speech-to-text conversion.
* **Uniform Pose Avatar:** Solves visual inconsistency by generating a standardized "pose skeleton" avatar using **MediaPipe**, ensuring a smooth viewing experience regardless of the original signer's appearance.
* **Smart Fallback System:**
    * *Fingerspelling:* Automatically fingerspells proper nouns and names (e.g., "fs-ALEX").
    * *Stand-in Logic:* Intelligently handles out-of-vocabulary words using a pause mechanism to prevent failure.
* **Memory-Safe Assembly:** Engineered a custom video concatenation pipeline using **FFmpeg** to stitch 1,000+ video clips without memory crashes.
* **Dynamic Captions:** Burns timestamped captions directly onto the final video for verification and learning.

---

## 🏗️ System Architecture

The project follows a **Two-Pipeline Architecture** to ensure performance and scalability.

### 1. Offline Asset Factory (Pre-processing)
*Before the app runs, we "manufacture" our assets to avoid slow real-time processing.*
* **Input:** WLASL-2000 Dataset & Custom Alphabet Dataset.
* **Process:** `process_dataset.py` runs MediaPipe on every video to extract 3D landmarks and renders them onto a black canvas.
* **Output:** A library of ~2,000 uniform pose videos (`hello.mp4`, `world.mp4`, `a.mp4`...).

### 2. Online Web Application (Real-time)
*The user-facing app orchestrates the translation on demand.*
1.  **Audio Download:** `yt-dlp` fetches audio from the URL.
2.  **Transcription:** OpenAI Whisper converts audio to text.
3.  **Translation:** Google Gemini converts text to **ASL Gloss**.
4.  **Assembly:** The system maps gloss tokens to the pre-processed asset library and stitches them using FFmpeg.

---

## 🛠️ Tech Stack

| Domain | Technologies |
| :--- | :--- |
| **Frontend** | Streamlit |
| **AI & ML** | OpenAI Whisper (ASR), Google Gemini API (LLM), MediaPipe (Pose Estimation) |
| **Video Processing** | OpenCV, FFmpeg, MoviePy |
| **Utilities** | yt-dlp (Audio Download), tqdm (Progress Tracking) |

---

## ⚙️ Installation & Setup

### Prerequisites
* Python 3.8+
* **FFmpeg** must be installed and added to your system PATH.

### Steps

1.  **Clone the repository**
    ```bash
    git clone [https://github.com/your-username/sign-language-generator.git](https://github.com/your-username/sign-language-generator.git)
    cd sign-language-generator
    ```

2.  **Create a virtual environment**
    ```bash
    # Windows
    python -m venv venv
    venv\Scripts\activate

    # Mac/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Prepare the Dataset**
    * Download the **WLASL dataset** (or use your own) and place it in the `raw_video_dataset` folder.
    * Run the factory scripts to generate the pose assets:
    ```bash
    python process_dataset.py
    python process_alphabet.py
    ```

5.  **Run the Application**
    ```bash
    streamlit run streamlit_app.py
    ```

---

## 🧠 Challenges & Solutions

| Challenge | Solution |
| :--- | :--- |
| **Memory Overload (OOM)** | Initial attempts to stitch 1,000+ clips using MoviePy crashed the RAM. We re-architected the backend to use **FFmpeg's "concat" demuxer**, enabling disk-based stitching with near-zero RAM usage. |
| **Linguistic Accuracy** | Direct translation resulted in "Signed English." We engineered a system prompt for **Gemini LLM** to act as an "ASL Linguist," strictly enforcing **Time-Topic-Comment** grammar. |
| **Video Playback** | Generated videos failed in browsers due to encoding issues. We implemented a strict re-encoding step using `libx264` and `yuv420p` pixel format for universal compatibility. |

---

## 🔮 Future Scope

* **MediaPipe Holistic:** Integrate facial expression and hand-shape tracking for more expressive ASL.
* **Expanded Vocabulary:** Scale the dataset processing to 10,000+ words.
* **Real-Time Translation:** Optimize the pipeline for live-streaming input.

---

## 👥 Contributors

* **K Sindhu** 
* **K Akshita** 
* **K Asritha** 

---
