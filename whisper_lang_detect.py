import argparse
import whisper
import json
import os
import subprocess
from pydub import AudioSegment
from pydub.utils import make_chunks

SUPPORTED_AUDIO = ["aac", "aiff", "amr", "flac", "mp3", "m4a", "ogg", "wav", "wma"]
SUPPORTED_VIDEO = ["avi", "mkv", "mov", "mpeg", "mp4", "webm", "wmv"]

# -----------------------
# Global model variable
# -----------------------
_model = None  # internal global variable


def get_model(model_name="base"):
    global _model
    if _model is None:
        print(f"Loading Whisper model '{model_name}' ...")
        _model = whisper.load_model(model_name)
    return _model


def ffmpeg_trim(input_path, output_path, max_seconds):
    """Trim audio/video using FFmpeg without loading entire file."""
    try:
        cmd = [
            "ffmpeg",
            "-y",
            "-i", input_path,
            "-t", str(max_seconds),
            "-ac", "1",
            "-ar", "16000",
            output_path
        ]
        subprocess.run(cmd, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
        return True
    except Exception:
        return False


def is_silent(filepath, silence_threshold=-50.0, chunk_ms=1000):
    """Check silence safely using pydub (only works for normal files)."""
    try:
        audio = AudioSegment.from_file(filepath)
    except Exception as e:
        print(f"Error loading audio for silence check: {e}")
        return True

    if len(audio) == 0:
        return True

    chunks = make_chunks(audio, chunk_ms)
    if len(chunks) == 0:
        return True

    for ck in chunks:
        if ck.dBFS > silence_threshold:
            return False

    return True


def detect_language(content_path, model_name="base", max_minutes=None):
    model = get_model(model_name)

    file_path, file_name = os.path.split(content_path)
    ext = file_name.split(".")[-1].lower()

    if ext not in SUPPORTED_AUDIO + SUPPORTED_VIDEO:
        return {
            "status": "1",
            "errormessage": "Unsupported extension.",
            "result": {"filename": file_name, "langcode": "", "probability": "null"}
        }

    # Handle max minutes
    max_seconds = max_minutes * 60 if max_minutes else None

    # Temp output
    trimmed_wav = os.path.join(file_path, f"__trimmed_{file_name}.wav")

    if max_seconds:
        # Trim file before any processing
        ok = ffmpeg_trim(content_path, trimmed_wav, max_seconds)
        if not ok:
            return {
                "status": "1",
                "errormessage": "Failed to trim file.",
                "result": {"filename": file_name, "langcode": "", "probability": "null"}
            }
        audio_path = trimmed_wav
    else:
        # Convert entire file to WAV (may fail for >4GB)
        audio_path = trimmed_wav
        ffmpeg_trim(content_path, audio_path, 99999999)  # basically full
        

    # Silence detection on trimmed audio only
    if is_silent(audio_path):
        os.remove(audio_path)
        return {
            "status": "1",
            "errormessage": "No speech was detected to classify.",
            "result": {"filename": file_name, "langcode": "", "probability": "null"}
        }

    try:
        # Whisper language detection
        audio = whisper.load_audio(audio_path)
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio).to(model.device)
        _, probs = model.detect_language(mel)

        lang = max(probs, key=probs.get)
        prob = float(probs[lang])

        return {
            "status": "0",
            "errormessage": "",
            "result": {"filename": file_name, "langcode": lang, "probability": prob}
        }

    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)


def process_folder(folder_path, model_name="base", max_minutes=None):
    results = []
    file_no = 1
    for fname in sorted(os.listdir(folder_path)):
        path = os.path.join(folder_path, fname)
        if not os.path.isfile(path):
            continue
        res = detect_language(path, model_name, max_minutes)
        if res["status"] == "0":
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
    parser.add_argument("--max_minutes", type=int, help="Analyze only first X minutes (positive integer)")

    args = parser.parse_args()

    get_model(args.model)

    if args.file:
        result = detect_language(args.file, args.model, args.max_minutes)
        print(json.dumps(result, ensure_ascii=False))
    elif args.folder:
        result = process_folder(args.folder, args.model, args.max_minutes)
        print(json.dumps(result, ensure_ascii=False))
    else:
        parser.error("Specify --file or --folder")


if __name__ == "__main__":
    main()
