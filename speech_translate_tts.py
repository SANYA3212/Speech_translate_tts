# -*- coding: utf-8 -*-
"""
Speech-Translate-TTS: Offline-first, real-time speech translation GUI.
Microphone -> Whisper (ASR) -> Ollama (Translate) -> Coqui (TTS)
"""

# 0. Environment Setup (CRITICAL!)
import os
import sys
import platform

# --- START: Desperate fix for stubborn ModuleNotFoundError ---
# This block manually adds the venv's site-packages to the path.
# This should not be necessary, but it's a fallback for broken environments.
try:
    # Assuming standard venv structure: <venv_root>/Scripts/python.exe
    venv_scripts_dir = os.path.dirname(sys.executable)
    site_packages = os.path.join(venv_scripts_dir, '..', 'Lib', 'site-packages')
    site_packages = os.path.normpath(site_packages)
    if os.path.exists(site_packages) and site_packages not in sys.path:
        sys.path.insert(0, site_packages)
        print(f"DEBUG: Manually added '{site_packages}' to sys.path to fix import errors.")
except Exception as e:
    print(f"DEBUG: Could not manually add site-packages. Error: {e}")
# --- END: Desperate fix ---

# Add torch/lib to PATH on Windows to help find DLLs like cuDNN
if platform.system() == 'Windows':
    try:
        torch_path = os.path.join(os.path.dirname(sys.executable), 'Lib', 'site-packages', 'torch', 'lib')
        if os.path.exists(torch_path):
            os.add_dll_directory(torch_path)
    except Exception as e:
        print(f"Warning: Could not add torch/lib to PATH. Error: {e}")

# Set environment variables for local, offline-first operation
os.environ['HF_HOME'] = os.path.join(os.getcwd(), '.hf_cache')
os.environ['TRANSFORMERS_CACHE'] = os.path.join(os.getcwd(), '.hf_cache', 'hub')
os.environ['TTS_HOME'] = os.getcwd()
os.environ['COQUI_TOS_AGREED'] = "1"

# --- Main Imports ---
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import queue
import time
import requests
import json
import logging
from pathlib import Path
import numpy as np
import collections

# --- Dependency Checks and Graceful Imports ---
try:
    import torch
    import torchaudio
    import sounddevice as sd
    import soundfile as sf
except ImportError as e:
    messagebox.showerror("Core Dependency Error", f"Module not found: {e.name}. Please run setup.bat to install dependencies.")
    sys.exit(1)

try:
    from faster_whisper import WhisperModel
except ImportError:
    messagebox.showerror("Dependency Error", "Module 'faster_whisper' not found. Please run setup.bat.")
    sys.exit(1)

try:
    from TTS.api import TTS
except ImportError:
    messagebox.showerror("Dependency Error", "Module 'TTS' from Coqui not found. Please run setup.bat.")
    sys.exit(1)


# --- Configuration ---
APP_TITLE = "Real-Time Speech Translator"
OFFLINE_ONLY = True  # Models should be downloaded by setup.bat
DEFAULT_WHISPER_MODEL_ID = "mobiuslabsgmbh/faster-whisper-large-v3-turbo"
DEFAULT_TTS_MODEL_ID = "coqui/XTTS-v2"
DEFAULT_WHISPER_PATH = f"models--{DEFAULT_WHISPER_MODEL_ID.replace('/', '--')}"
DEFAULT_TTS_PATH = "tts_models--multilingual--multi-dataset--xtts_v2"
DEFAULT_OLLAMA_MODEL = "gemma:2b"
OLLAMA_BASE_URL = "http://127.0.0.1:11434"

# Audio streaming settings
SAMPLE_RATE = 16000  # Whisper requires 16kHz
STREAM_WINDOW_S = 15  # 15-second audio window for streaming
STREAM_STEP_S = 0.75 # Process audio every 0.75 seconds
BUFFER_SIZE_SAMPLES = int(STREAM_WINDOW_S * SAMPLE_RATE)

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# --- Helper Functions ---
def find_model_path(local_path, repo_id):
    """
    Checks if a model exists at the standardized local path.
    Returns the local path if found, otherwise returns the repo_id.
    """
    path = Path(local_path)
    if path.exists() and (path / "config.json").exists():
        logging.info(f"Found model at standardized path: {path}")
        return str(path)

    logging.warning(f"Model not found at '{local_path}'. Falling back to repo ID '{repo_id}'.")
    # This will only work if the user manually sets OFFLINE_ONLY=False
    return repo_id

def check_ollama_status(model_name):
    """Checks if Ollama is running and has the specified model."""
    try:
        response = requests.get(OLLAMA_BASE_URL, timeout=3)
        response.raise_for_status()
    except requests.RequestException:
        return False, "Ollama server is not accessible at " + OLLAMA_BASE_URL

    try:
        response = requests.post(f"{OLLAMA_BASE_URL}/api/show", json={"name": model_name}, timeout=5)
        if response.status_code == 404:
            return True, f"Model '{model_name}' not found. Trying to pull..."
        response.raise_for_status()
        return True, f"Ollama is running and model '{model_name}' is available."
    except requests.RequestException as e:
        return True, f"Ollama is running, but could not verify model '{model_name}': {e}"

def pull_ollama_model(model_name, app_instance):
    """Pulls an Ollama model, streaming progress to the GUI."""
    try:
        app_instance.update_status(f"Pulling Ollama model '{model_name}'. This may take a while...")
        req = requests.post(f"{OLLAMA_BASE_URL}/api/pull", json={"name": model_name, "stream": True}, stream=True)
        req.raise_for_status()
        for line in req.iter_lines():
            if line:
                data = json.loads(line)
                if 'status' in data:
                    app_instance.update_status(data['status'])
                if data.get('error'):
                    raise ConnectionError(data['error'])
        app_instance.update_status(f"Ollama model '{model_name}' pulled successfully.")
        return True
    except (requests.RequestException, ConnectionError, json.JSONDecodeError) as e:
        error_msg = f"Failed to pull Ollama model '{model_name}': {e}"
        app_instance.update_status(error_msg)
        messagebox.showerror("Ollama Error", error_msg)
        return False

def get_audio_devices():
    """Returns lists of input and output audio devices."""
    try:
        devices = sd.query_devices()
        input_devices = [dev['name'] for dev in devices if dev['max_input_channels'] > 0]
        output_devices = [dev['name'] for dev in devices if dev['max_output_channels'] > 0]
        return input_devices, output_devices
    except Exception as e:
        logging.error(f"Could not query audio devices: {e}")
        return [], []

def longest_common_prefix(s1, s2):
    """Finds the longest common prefix string between two strings."""
    i = 0
    while i < len(s1) and i < len(s2) and s1[i] == s2[i]:
        i += 1
    return s1[:i]


# --- Worker Threads ---

class ASRWorker(threading.Thread):
    def __init__(self, mode, input_device_idx, vad, result_queue, status_queue):
        super().__init__(daemon=True)
        self.mode = mode # 'single_shot' or 'streaming'
        self.input_device_idx = input_device_idx
        self.vad = vad
        self.result_queue = result_queue
        self.status_queue = status_queue
        self.stop_event = threading.Event()
        self.audio_buffer = collections.deque(maxlen=BUFFER_SIZE_SAMPLES)
        self.full_transcript = ""
        self.last_stable_transcript = ""

    def stop(self):
        self.stop_event.set()

    def _process_transcription(self, model, audio_data):
        try:
            segments, info = model.transcribe(
                audio_data,
                beam_size=5,
                vad_filter=self.vad,
                vad_parameters=dict(min_silence_duration_ms=500),
                language=None # Auto-detect
            )

            if self.mode == 'streaming':
                current_transcript = "".join(seg.text for seg in segments).strip()

                if not self.last_stable_transcript:
                    # First run
                    self.last_stable_transcript = current_transcript
                    newly_confirmed = current_transcript
                else:
                    # Find stable prefix
                    stable_prefix = longest_common_prefix(self.last_stable_transcript, current_transcript)
                    newly_confirmed = self.last_stable_transcript[len(stable_prefix):]
                    self.last_stable_transcript = stable_prefix

                live_text = current_transcript[len(self.last_stable_transcript):]

                if newly_confirmed:
                    self.full_transcript += newly_confirmed

                self.result_queue.put({
                    "type": "asr_result",
                    "confirmed_text": newly_confirmed,
                    "live_text": live_text,
                    "source_lang": info.language
                })

            else: # single_shot
                full_text = "".join(seg.text for seg in segments).strip()
                self.result_queue.put({
                    "type": "asr_result",
                    "confirmed_text": full_text,
                    "live_text": "",
                    "source_lang": info.language
                })

        except Exception as e:
            self.status_queue.put(f"ASR Error: {e}")
            logging.error(f"Exception in ASR transcription: {e}", exc_info=True)


    def run(self):
        global whisper_model
        if whisper_model is None:
            self.status_queue.put("Whisper model not loaded.")
            return

        self.status_queue.put(f"Starting {self.mode} recording...")

        try:
            if self.mode == 'streaming':
                self.audio_buffer.clear()
                self.full_transcript = ""
                self.last_stable_transcript = ""

                def audio_callback(indata, frames, time, status):
                    if status:
                        self.status_queue.put(f"Audio Warning: {status}")
                    self.audio_buffer.extend(indata[:, 0])

                with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='float32',
                                    device=self.input_device_idx, callback=audio_callback):
                    while not self.stop_event.is_set():
                        time.sleep(STREAM_STEP_S)
                        if len(self.audio_buffer) > 0:
                            audio_data = np.array(self.audio_buffer)
                            self._process_transcription(whisper_model, audio_data)

            else: # single_shot
                recorded_audio = []
                def audio_callback(indata, frames, time, status):
                    if status:
                        self.status_queue.put(f"Audio Warning: {status}")
                    recorded_audio.append(indata.copy())

                with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='float32',
                                    device=self.input_device_idx, callback=audio_callback):
                    while not self.stop_event.is_set():
                        sd.sleep(100)

                if recorded_audio:
                    audio_data = np.concatenate(recorded_audio, axis=0).flatten()
                    # Normalize audio if clipping
                    if np.max(np.abs(audio_data)) > 1.0:
                        audio_data = audio_data / np.max(np.abs(audio_data))
                    self.status_queue.put("Transcription started...")
                    self._process_transcription(whisper_model, audio_data)

        except Exception as e:
            error_msg = f"Audio stream error: {e}"
            self.status_queue.put(error_msg)
            logging.error(error_msg, exc_info=True)
            messagebox.showerror("Audio Error", error_msg)

        self.status_queue.put("Recording stopped.")


class TTSWorker(threading.Thread):
    def __init__(self, task_queue, status_queue, output_device_idx):
        super().__init__(daemon=True)
        self.task_queue = task_queue
        self.status_queue = status_queue
        self.output_device_idx = output_device_idx

    def run(self):
        global tts_model
        if tts_model is None:
            self.status_queue.put("TTS model not loaded.")
            return

        while True:
            try:
                task = self.task_queue.get(block=True)
                text, lang, speaker_wav = task['text'], task['lang'], task['speaker_wav']

                if not text:
                    continue

                self.status_queue.put(f"Synthesizing speech for: '{text[:30]}...'")

                # Check if output device is valid before synthesis
                try:
                    sd.check_output_settings(device=self.output_device_idx, samplerate=24000) # XTTS default rate
                except Exception as e:
                    self.status_queue.put(f"TTS Error: Invalid output device. {e}")
                    continue

                wav = tts_model.tts(text=text, language=lang, speaker_wav=speaker_wav, split_sentences=True)

                # Play audio using sounddevice
                sd.play(np.array(wav), samplerate=tts_model.synthesizer.output_sample_rate, device=self.output_device_idx)
                sd.wait()

                self.status_queue.put("Speech synthesis finished.")
                self.task_queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                error_msg = f"TTS synthesis failed: {e}"
                self.status_queue.put(error_msg)
                logging.error(error_msg, exc_info=True)
                self.task_queue.task_done()


# --- Main Application GUI ---

class SpeechTranslatorApp:
    def __init__(self, root):
        self.root = root
        self.root.title(APP_TITLE)
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Global models
        global whisper_model, tts_model
        whisper_model = None
        tts_model = None

        # State variables
        self.input_devices, self.output_devices = [], []
        self.asr_worker = None
        self.tts_worker = None
        self.is_recording = False
        self.is_streaming = False

        # Queues for thread communication
        self.result_queue = queue.Queue()
        self.status_queue = queue.Queue()
        self.tts_queue = queue.Queue()

        # Tkinter variables
        self.source_lang_var = tk.StringVar(value="auto")
        self.target_lang_var = tk.StringVar(value="en")
        self.ollama_model_var = tk.StringVar(value=DEFAULT_OLLAMA_MODEL)
        self.mic_var = tk.StringVar()
        self.speaker_var = tk.StringVar()
        self.vad_var = tk.BooleanVar(value=True)
        self.auto_speak_var = tk.BooleanVar(value=True)
        self.auto_save_var = tk.BooleanVar(value=False)
        self.speaker_wav_var = tk.StringVar()
        self.status_var = tk.StringVar(value="Welcome! Load models to begin.")

        self.create_widgets()
        self.update_audio_devices()
        self.process_queues()

    def create_widgets(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        # --- Models and Devices Frame ---
        models_frame = ttk.LabelFrame(main_frame, text="Setup", padding="10")
        models_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E))
        models_frame.columnconfigure(1, weight=1)
        models_frame.columnconfigure(3, weight=1)

        ttk.Button(models_frame, text="Load Whisper", command=self.load_whisper_model).grid(row=0, column=0, padx=5, pady=5)
        self.whisper_status = ttk.Label(models_frame, text="Whisper: Not Loaded")
        self.whisper_status.grid(row=0, column=1, sticky=tk.W)

        ttk.Button(models_frame, text="Load XTTS", command=self.load_xtts_model).grid(row=1, column=0, padx=5, pady=5)
        self.xtts_status = ttk.Label(models_frame, text="XTTS: Not Loaded")
        self.xtts_status.grid(row=1, column=1, sticky=tk.W)

        ttk.Label(models_frame, text="Mic:").grid(row=0, column=2, padx=5, sticky=tk.E)
        self.mic_combo = ttk.Combobox(models_frame, textvariable=self.mic_var, state="readonly")
        self.mic_combo.grid(row=0, column=3, sticky=(tk.W, tk.E))

        ttk.Label(models_frame, text="Speaker:").grid(row=1, column=2, padx=5, sticky=tk.E)
        self.speaker_combo = ttk.Combobox(models_frame, textvariable=self.speaker_var, state="readonly")
        self.speaker_combo.grid(row=1, column=3, sticky=(tk.W, tk.E))

        ttk.Button(models_frame, text="Refresh Devices", command=self.update_audio_devices).grid(row=0, column=4, rowspan=2, padx=5)

        # --- Translation and Language Frame ---
        lang_frame = ttk.LabelFrame(main_frame, text="Translation", padding="10")
        lang_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        lang_frame.columnconfigure(1, weight=1)
        lang_frame.columnconfigure(3, weight=1)

        ttk.Label(lang_frame, text="Source Lang:").grid(row=0, column=0, padx=5, sticky=tk.W)
        ttk.Entry(lang_frame, textvariable=self.source_lang_var, width=10).grid(row=0, column=1, sticky=tk.W)

        ttk.Label(lang_frame, text="Target Lang:").grid(row=0, column=2, padx=5, sticky=tk.W)
        ttk.Entry(lang_frame, textvariable=self.target_lang_var, width=10).grid(row=0, column=3, sticky=tk.W)

        ttk.Label(lang_frame, text="Ollama Model:").grid(row=1, column=0, padx=5, sticky=tk.W)
        ttk.Entry(lang_frame, textvariable=self.ollama_model_var).grid(row=1, column=1, columnspan=3, sticky=(tk.W, tk.E))

        # --- Controls and Options Frame ---
        controls_frame = ttk.LabelFrame(main_frame, text="Controls", padding="10")
        controls_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)

        self.record_btn = ttk.Button(controls_frame, text="Start Recording", command=self.toggle_single_shot)
        self.record_btn.pack(fill=tk.X, pady=2)
        self.stream_btn = ttk.Button(controls_frame, text="Start Streaming", command=self.toggle_streaming)
        self.stream_btn.pack(fill=tk.X, pady=2)
        ttk.Button(controls_frame, text="Clear Text", command=self.clear_text).pack(fill=tk.X, pady=2)

        options_frame = ttk.LabelFrame(main_frame, text="Options", padding="10")
        options_frame.grid(row=2, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)

        ttk.Checkbutton(options_frame, text="VAD (Voice Activity Detection)", variable=self.vad_var).pack(anchor=tk.W)
        ttk.Checkbutton(options_frame, text="Auto-speak Translation", variable=self.auto_speak_var).pack(anchor=tk.W)
        ttk.Checkbutton(options_frame, text="Auto-save Transcript", variable=self.auto_save_var).pack(anchor=tk.W)

        speaker_wav_frame = ttk.Frame(options_frame)
        speaker_wav_frame.pack(fill=tk.X, pady=5)
        ttk.Button(speaker_wav_frame, text="Speaker WAV...", command=self.select_speaker_wav).pack(side=tk.LEFT)
        ttk.Entry(speaker_wav_frame, textvariable=self.speaker_wav_var, state="readonly").pack(side=tk.LEFT, fill=tk.X, expand=True)

        # --- Text Widgets ---
        text_frame = ttk.Frame(main_frame)
        text_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        text_frame.rowconfigure(1, weight=1)
        text_frame.columnconfigure(0, weight=1)

        ttk.Label(text_frame, text="Live Transcription:", font=("Segoe UI", 10, "bold")).grid(row=0, column=0, sticky=tk.W)
        self.live_text = tk.Text(text_frame, height=3, wrap=tk.WORD, state=tk.DISABLED)
        self.live_text.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        ttk.Label(text_frame, text="Confirmed Transcript & Translation:", font=("Segoe UI", 10, "bold")).grid(row=2, column=0, sticky=tk.W, pady=(10,0))
        self.confirmed_text = tk.Text(text_frame, height=10, wrap=tk.WORD)
        self.confirmed_text.grid(row=3, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        main_frame.rowconfigure(3, weight=1)

        # --- Status Bar ---
        status_bar = ttk.Frame(self.root, relief=tk.SUNKEN, padding=(2, 5))
        status_bar.grid(row=1, column=0, sticky=(tk.W, tk.E))
        ttk.Label(status_bar, textvariable=self.status_var).pack(fill=tk.X)

    def update_status(self, message):
        self.status_var.set(message)
        logging.info(message)

    def process_queues(self):
        try:
            while not self.status_queue.empty():
                msg = self.status_queue.get_nowait()
                self.update_status(msg)

            while not self.result_queue.empty():
                res = self.result_queue.get_nowait()
                if res['type'] == 'asr_result':
                    self.handle_asr_result(res)

        except queue.Empty:
            pass
        finally:
            self.root.after(100, self.process_queues)

    def handle_asr_result(self, result):
        confirmed = result.get('confirmed_text', '')
        live = result.get('live_text', '')
        source_lang = result.get('source_lang', 'auto')

        self.source_lang_var.set(source_lang)

        self.live_text.config(state=tk.NORMAL)
        self.live_text.delete('1.0', tk.END)
        self.live_text.insert(tk.END, live)
        self.live_text.config(state=tk.DISABLED)

        if confirmed:
            self.confirmed_text.insert(tk.END, f"[SRC: {source_lang}] {confirmed}\n")
            self.confirmed_text.see(tk.END)
            self.translate_and_speak(confirmed, source_lang)

    def translate_and_speak(self, text, source_lang):
        target_lang = self.target_lang_var.get()
        ollama_model = self.ollama_model_var.get()

        def do_translation():
            prompt = f"Translate the following text to {target_lang}. Output only the translated text, without any explanations or conversational fluff. Text to translate: \"{text}\""
            try:
                self.update_status(f"Translating with {ollama_model}...")
                response = requests.post(
                    f"{OLLAMA_BASE_URL}/api/generate",
                    json={"model": ollama_model, "prompt": prompt, "stream": False},
                    timeout=20
                )
                response.raise_for_status()
                translated_text = response.json().get('response', '').strip()

                if translated_text:
                    self.confirmed_text.insert(tk.END, f"  -> [TRG: {target_lang}] {translated_text}\n\n")
                    self.confirmed_text.see(tk.END)

                    if self.auto_save_var.get():
                        with open("transcript.txt", "a", encoding="utf-8") as f:
                            f.write(f"[SRC] {text}\n[TRG] {translated_text}\n\n")

                    if self.auto_speak_var.get():
                        self.tts_queue.put({
                            "text": translated_text,
                            "lang": target_lang,
                            "speaker_wav": self.speaker_wav_var.get() or None
                        })
                self.update_status("Translation complete.")
            except requests.RequestException as e:
                self.update_status(f"Ollama Error: {e}")

        threading.Thread(target=do_translation, daemon=True).start()

    def _load_model_thread(self, model_type):
        try:
            if model_type == 'whisper':
                self.update_status("Finding Whisper model...")
                model_path = find_model_path(DEFAULT_WHISPER_PATH, DEFAULT_WHISPER_MODEL_ID)

                self.update_status(f"Loading Whisper from '{model_path}'...")
                global whisper_model
                whisper_model = WhisperModel(model_path, device="cuda", compute_type="int8_float16", download_root=DEFAULT_WHISPER_PATH)

                self.root.after(0, lambda: self.whisper_status.config(text="Whisper: Loaded OK"))
                self.update_status("Whisper model loaded successfully.")

            elif model_type == 'tts':
                self.update_status("Finding XTTS model...")
                # For TTS, the model path and config path are often the same directory
                model_path = find_model_path(DEFAULT_TTS_PATH, DEFAULT_TTS_MODEL_ID)

                self.update_status(f"Loading XTTS from '{model_path}'...")
                global tts_model
                tts_model = TTS(model_path=model_path, config_path=os.path.join(model_path, 'config.json'), progress_bar=True).to("cuda")

                self.root.after(0, lambda: self.xtts_status.config(text="XTTS: Loaded OK"))
                self.update_status("XTTS model loaded successfully. Starting TTS worker...")

                if self.tts_worker is None or not self.tts_worker.is_alive():
                    self.tts_worker = TTSWorker(self.tts_queue, self.status_queue, self.speaker_combo.current())
                    self.tts_worker.start()

        except Exception as e:
            error_msg = f"Failed to load {model_type} model. Please ensure it was downloaded correctly with setup.bat. Error: {e}"
            self.update_status(error_msg)
            logging.error(error_msg, exc_info=True)
            messagebox.showerror("Model Loading Error", error_msg)
            if model_type == 'whisper':
                self.root.after(0, lambda: self.whisper_status.config(text="Whisper: Error"))
            else:
                self.root.after(0, lambda: self.xtts_status.config(text="XTTS: Error"))

    def load_whisper_model(self):
        self.whisper_status.config(text="Whisper: Loading...")
        threading.Thread(target=self._load_model_thread, args=('whisper',), daemon=True).start()

    def load_xtts_model(self):
        self.xtts_status.config(text="XTTS: Loading...")
        threading.Thread(target=self._load_model_thread, args=('tts',), daemon=True).start()

    def update_audio_devices(self):
        self.input_devices, self.output_devices = get_audio_devices()
        self.mic_combo['values'] = self.input_devices
        if self.input_devices:
            self.mic_combo.current(0)
        self.speaker_combo['values'] = self.output_devices
        if self.output_devices:
            self.speaker_combo.current(0)
        self.update_status("Audio devices updated.")

    def select_speaker_wav(self):
        filepath = filedialog.askopenfilename(
            title="Select Speaker WAV file",
            filetypes=(("WAV files", "*.wav"), ("All files", "*.*"))
        )
        if filepath:
            self.speaker_wav_var.set(filepath)

    def clear_text(self):
        self.live_text.config(state=tk.NORMAL)
        self.live_text.delete('1.0', tk.END)
        self.live_text.config(state=tk.DISABLED)
        self.confirmed_text.delete('1.g', tk.END)

    def toggle_single_shot(self):
        if self.is_streaming:
            messagebox.showwarning("Warning", "Stop streaming before starting a single-shot recording.")
            return

        if self.is_recording:
            self.is_recording = False
            self.record_btn.config(text="Start Recording")
            if self.asr_worker:
                self.asr_worker.stop()
                self.asr_worker = None
        else:
            self.is_recording = True
            self.record_btn.config(text="Stop Recording")
            self.asr_worker = ASRWorker(
                'single_shot', self.mic_combo.current(), self.vad_var.get(),
                self.result_queue, self.status_queue
            )
            self.asr_worker.start()

    def toggle_streaming(self):
        if self.is_recording:
            messagebox.showwarning("Warning", "Stop single-shot recording before starting to stream.")
            return

        if self.is_streaming:
            self.is_streaming = False
            self.stream_btn.config(text="Start Streaming")
            if self.asr_worker:
                self.asr_worker.stop()
                self.asr_worker = None
        else:
            self.is_streaming = True
            self.stream_btn.config(text="Stop Streaming")
            self.asr_worker = ASRWorker(
                'streaming', self.mic_combo.current(), self.vad_var.get(),
                self.result_queue, self.status_queue
            )
            self.asr_worker.start()

    def on_closing(self):
        if self.asr_worker and self.asr_worker.is_alive():
            self.asr_worker.stop()
        # TTS worker is a daemon, will exit automatically
        self.root.destroy()


if __name__ == "__main__":
    # Initial checks
    logging.info(f"Torch version: {torch.__version__}, CUDA available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        messagebox.showwarning("CUDA Warning", "PyTorch CUDA is not available. The application will run on CPU, which will be very slow.")

    is_ollama_up, ollama_msg = check_ollama_status(DEFAULT_OLLAMA_MODEL)
    if not is_ollama_up:
        messagebox.showwarning("Ollama Warning", f"{ollama_msg}. Translation will be disabled.")
    else:
        logging.info(ollama_msg)

    # Create and run main window
    root = tk.Tk()
    app = SpeechTranslatorApp(root)
    root.mainloop()
