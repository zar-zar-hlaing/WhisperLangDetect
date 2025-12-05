import argparse
import whisper
import json
import os
from pydub import AudioSegment
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


def is_silent(content_path, threshold=-50):
    """Check if audio/video file is silent or has no speech."""
    ext = content_path.split(".")[-1].lower()
    try:
        if ext in SUPPORTED_AUDIO:
            sound = AudioSegment.from_file(content_path)
            if sound is None:
                return True
            samples = np.array(sound.get_array_of_samples())
        elif ext in SUPPORTED_VIDEO:
            clip = VideoFileClip(content_path)
            if clip.audio is None:
                return True
            # Convert audio to mono and fixed sample rate
            samples = clip.audio.to_soundarray(fps=16000)
            if samples.size == 0:
                return True
            if len(samples.shape) == 2:  # stereo → mono
                samples = samples.mean(axis=1)
        else:
            return True  # unsupported extension treated as silent

        rms = np.sqrt(np.mean(np.square(samples)))
        db = 20 * np.log10(rms) if rms > 0 else -100
        return db < threshold
    except Exception as e:
        print(f"Error checking silence in {content_path}: {e}")
        return True


def detect_language(content_path, model_name="base"):
    """Detect language in a single file and return JSON-compatible result."""
    model = get_model(model_name)  # ensure model is loaded

    file_path, file_name = os.path.split(content_path)
    ext = file_name.split(".")[-1].lower()

    if ext not in SUPPORTED_AUDIO + SUPPORTED_VIDEO:
        return {
            "status": "1",
            "errormessage": "Unsupported extension. Allowed: "
                            + ", ".join(SUPPORTED_AUDIO + SUPPORTED_VIDEO),
            "result": {"filename": file_name, "langcode": "", "probability": "null"}
        }

    if is_silent(content_path):
        return {
            "status": "1",
            "errormessage": "No speech was detected to classify.",
            "result": {"filename": file_name, "langcode": "", "probability": "null"}
        }

    # Convert video/audio to a temporary WAV file
    temp_path = None
    try:
        tmp_name, _ = os.path.splitext(file_name)
        temp_path = os.path.join(file_path, tmp_name + ".wav")
        AudioSegment.from_file(content_path).set_channels(1).set_frame_rate(16000).export(temp_path, format="wav")
        audio_path = temp_path

        audio = whisper.load_audio(audio_path)
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio).to(model.device)
        _, probs = model.detect_language(mel)
        lang = max(probs, key=probs.get)
        prob = probs[lang]

        result = {
            "status": "0",
            "errormessage": "",
            "result": {"filename": file_name, "langcode": lang, "probability": float(prob)}
        }
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

    return result



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
