import argparse
import whisper
import json
import os
from pydub import AudioSegment
from pydub.utils import make_chunks
from moviepy.editor import VideoFileClip
import numpy as np

SUPPORTED_AUDIO = ["aac", "aiff", "amr", "flac", "mp3", "m4a", "ogg", "wav", "wma"]
SUPPORTED_VIDEO = ["avi", "mkv", "mov", "mpeg", "mp4", "webm", "wmv"]

# -----------------------
# Global model variable
# -----------------------
_model = None  # internal global variable


def get_model(model_name="base"):
    """Lazy-load and return the Whisper model."""
    global _model
    if _model is None:
        print(f"Loading Whisper model '{model_name}' ...")
        _model = whisper.load_model(model_name)
    return _model


def is_silent(filepath, silence_threshold=-50.0, chunk_ms=1000):
    """Return True if no meaningful audio exists."""
    try:
        audio = AudioSegment.from_file(filepath)
    except Exception as e:
        print(f"Error loading audio for silence check: {e}")
        return True  # treat as silence

    # No audio frames
    if len(audio) == 0:
        return True

    # Process in chunks and check max dBFS
    chunks = make_chunks(audio, chunk_ms)

    # Ensure chunks is a list
    if not isinstance(chunks, (list, tuple)):
        return True

    # If no valid chunks => silence
    if len(chunks) == 0:
        return True

    # Check each chunk
    for ck in chunks:
        if ck.dBFS > silence_threshold:
            return False

    return True

def detect_language(content_path, model_name="base"):
    """Detect language in a single file and return JSON-compatible result."""
    
    model = get_model(model_name)

    file_path, file_name = os.path.split(content_path)
    ext = file_name.split(".")[-1].lower()

    # Unsupported file type
    if ext not in SUPPORTED_AUDIO + SUPPORTED_VIDEO:
        return {
            "status": "1",
            "errormessage": "Unsupported extension. Allowed: "
                            + ", ".join(SUPPORTED_AUDIO + SUPPORTED_VIDEO),
            "result": {"filename": file_name, "langcode": "", "probability": "null"}
        }

    # Silence detection
    if is_silent(content_path):
        return {
            "status": "1",
            "errormessage": "No speech was detected to classify.",
            "result": {"filename": file_name, "langcode": "", "probability": "null"}
        }

    temp_path = None
    try:
        # Convert to mono 16 kHz WAV for Whisper
        tmp_name, _ = os.path.splitext(file_name)
        temp_path = os.path.join(file_path, tmp_name + ".wav")

        AudioSegment.from_file(content_path).set_channels(1).set_frame_rate(16000)\
            .export(temp_path, format="wav")

        # Load audio for Whisper
        audio = whisper.load_audio(temp_path)
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio).to(model.device)

        # Detect language
        _, probs = model.detect_language(mel)
        lang = max(probs, key=probs.get)
        prob = float(probs[lang])

        return {
            "status": "0",
            "errormessage": "",
            "result": {"filename": file_name, "langcode": lang, "probability": prob}
        }

    finally:
        # Cleanup temporary file
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

def process_folder(folder_path, model_name="base"):
    """Process all supported files in a folder."""
    results = []
    file_no = 1
    for fname in sorted(os.listdir(folder_path)):
        path = os.path.join(folder_path, fname)
        if not os.path.isfile(path):
            continue
        res = detect_language(path, model_name)
        if res["status"] == "0":  # only keep valid results
            r = res["result"]
            r["fileno"] = file_no
            results.append(r)
            file_no += 1
    return {"result": results}


def main():
    parser = argparse.ArgumentParser(description="Whisper-based audio/video language detector")
    parser.add_argument("--file", type=str, help="Path to a single audio/video file")
    parser.add_argument("--folder", type=str, help="Path to a folder of audio/video files")
    parser.add_argument("--model", type=str, default="base",
                        choices=["tiny", "base", "small", "medium", "large"],
                        help="Whisper model to use (default: base)")
    args = parser.parse_args()

    # Load model globally once
    get_model(args.model)

    if args.file:
        result = detect_language(args.file, args.model)
        print(json.dumps(result, ensure_ascii=False))
    elif args.folder:
        result = process_folder(args.folder, args.model)
        print(json.dumps(result, ensure_ascii=False))
    else:
        parser.error("You must specify either --file or --folder")


if __name__ == "__main__":
    main()
