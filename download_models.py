import os
import sys
from pathlib import Path

try:
    from huggingface_hub import snapshot_download, HfFolder
    from huggingface_hub.utils import HfHubHTTPError
except ImportError:
    print("Error: huggingface-hub is not installed. Please run 'pip install -r requirements.txt' first.")
    sys.exit(1)

# --- Configuration ---
# Set a token if you have one, or login via CLI `huggingface-cli login`
# HfFolder.save_token('YOUR_HF_TOKEN_HERE')

WHISPER_MODEL = "mobiuslabsgmbh/faster-whisper-large-v3-turbo"
# The local path should match the format the GUI expects
LOCAL_WHISPER_PATH = f"models--{WHISPER_MODEL.replace('/', '--')}"

TTS_MODEL = "coqui/XTTS-v2"
# The local path for TTS model is expected in this format by the GUI's find function
LOCAL_TTS_PATH = "tts_models--multilingual--multi-dataset--xtts_v2"


HF_CACHE_DIR = Path(os.getcwd()) / ".hf_cache"
os.environ['HF_HOME'] = str(HF_CACHE_DIR)
os.environ['TRANSFORMERS_CACHE'] = str(HF_CACHE_DIR / "hub")

def download_model(repo_id, local_path, model_name):
    """Downloads a model from Hugging Face Hub if it doesn't exist locally."""
    target_path = Path(local_path)
    if (target_path / "config.json").exists():
        print(f"✅ {model_name} model already exists at: {target_path}")
        return

    print(f"🚀 Downloading {model_name} model: {repo_id}")
    print(f"   This may take a while depending on your internet connection...")
    print(f"   Saving to: {target_path}")

    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(target_path),
            cache_dir=str(HF_CACHE_DIR / "hub"),
            local_dir_use_symlinks=False, # Use False for better Windows compatibility
            resume_download=True,
            # For XTTS, we don't need the massive .wav files
            ignore_patterns=["*.wav"] if "xtts" in repo_id.lower() else None
        )
        print(f"✅ Successfully downloaded {model_name} model.")
    except HfHubHTTPError as e:
        print(f"❌ ERROR: Failed to download {model_name} model.")
        print(f"   Please check your internet connection and if you have access to the model repo.")
        print(f"   HF Error: {e}")
        # Clean up partial download
        if target_path.exists():
            import shutil
            shutil.rmtree(target_path)
        sys.exit(1)
    except Exception as e:
        print(f"❌ ERROR: An unexpected error occurred while downloading {model_name}.")
        print(f"   Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    print("--- Starting Model Download ---")

    # Download Whisper model
    download_model(WHISPER_MODEL, LOCAL_WHISPER_PATH, "Whisper")

    # Download XTTS model
    download_model(TTS_MODEL, LOCAL_TTS_PATH, "XTTS")

    print("\n--- Model download process finished. ---")
    print("Please also ensure you have the Ollama model available.")
    print("You can get it by running: ollama pull gemma:2b")
    print("------------------------------------------")
