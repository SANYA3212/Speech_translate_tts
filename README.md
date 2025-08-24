# Real-Time Speech Translator

This project is an offline-first, real-time speech translation GUI for Windows. It captures audio from a microphone, transcribes it to text, translates the text, and synthesizes it back to speech, all locally on your machine.

---

## Features

-   **Real-Time Streaming:** Transcribes speech as you talk using VAD-based endpointing.
-   **Single-Shot Mode:** Record a single phrase and have it translated.
-   **Offline-First:** Works entirely locally. Models are downloaded once during setup.
-   **High-Quality Models:**
    -   **ASR:** `faster-whisper` (large-v3-turbo) for fast and accurate transcription.
    -   **Translation:** `Ollama` with the `gemma:3b` model.
    -   **TTS:** `Coqui XTTS-v2` for high-quality voice synthesis.
-   **Voice Cloning:** Use any `.wav` file to clone a voice for the TTS output.
-   **Advanced Audio Routing:** Output synthesized speech to multiple audio devices simultaneously, including virtual audio cables for use in other applications.

## Tech Stack

-   **GUI:** Python + Tkinter
-   **ASR (Speech-to-Text):** `faster-whisper`
-   **Translation LLM:** `Ollama`
-   **TTS (Text-to-Speech):** `Coqui TTS`
-   **Audio I/O:** `sounddevice`

---

## Prerequisites

Before you begin, ensure you have the following installed and configured on your Windows 10/11 machine:

1.  **NVIDIA GPU:** A powerful GPU (like an RTX 30-series or 40-series) is highly recommended for good performance.
2.  **CUDA Drivers:** Ensure your NVIDIA drivers and CUDA Toolkit (version 12.1 or compatible) are installed.
3.  **Python 3.11:** The application is pinned to Python 3.11.x. Make sure it is installed and the `py` launcher is available in your system's PATH.
4.  **Ollama:** You must have [Ollama](https://ollama.com/) installed and running.
5.  **(Optional) Virtual Audio Cable:** For routing audio to other applications, it's recommended to install a virtual audio cable program like [VB-CABLE](https://vb-audio.com/Cable/).

---

## Installation & Setup

Follow these steps to get the application running. The setup script will handle creating a virtual environment, installing all dependencies, and downloading the necessary AI models.

1.  **Clone the Repository**
    ```bash
    git clone <repository-url>
    cd <repository-folder>
    ```

2.  **Prepare Ollama Model**
    Open a terminal or PowerShell and pull the required translation model:
    ```bash
    ollama pull gemma3:1b
    ```

3.  **Run the Setup Script**
    Simply double-click on `setup.bat`.

    -   This script will create a local Python virtual environment (`venv311`).
    -   It will install all required Python packages.
    -   It will then run `download_models.py` to download the Whisper and TTS models (several gigabytes).

    > **IMPORTANT:** The first run of this script can be very long (10-20 minutes or more) depending on your internet speed. Please be patient and **do not interrupt the process**, even if the console window seems "stuck". Wait for the `Setup complete!` message.

---

## Usage

1.  **Launch the Application**
    Double-click `run_gui.bat` to start the GUI.

2.  **Using the Interface**
    -   **Load Models:** Click the "Load Whisper" and "Load XTTS" buttons first. Wait for the status to change to "Loaded OK".
    -   **Select Devices:**
        -   Choose your primary microphone from the "Mic" dropdown.
        -   Select one or more output devices from the "Output Devices" list. You can use `Ctrl+Click` to select multiple devices.
    -   **Configure Options:**
        -   **Auto-speak:** Automatically plays the translated text.
        -   **Virtual Mic Checkbox:** If you have a virtual audio cable named "Speech-Translate-TTS", check this box to automatically include it in the output.
        -   **Speaker WAV:** Select a `.wav` file to clone a voice for the TTS. If none is selected, a default voice will be used.
    -   **Start Translating:**
        -   **Single-Shot:** Click "Start Recording", speak a phrase, then click "Stop Recording".
        -   **Streaming:** Click "Start Streaming" and begin speaking. The app will detect when you pause and process the phrase. Click "Stop Streaming" to end the session.

---

## Folder Structure

```
.
├── logs/                 # Log files are saved here
├── models--.../          # Whisper ASR model files
├── tts_models--.../      # Coqui TTS model files
├── venv311/              # Python virtual environment
├── .hf_cache/            # Hugging Face cache
├── download_models.py    # Script to download AI models
├── requirements.txt      # List of Python dependencies
├── speech_translate_tts.py # The main application source code
├── setup.bat             # The main setup script
├── run_gui.bat           # The script to launch the application
└── README.md             # This file
```
