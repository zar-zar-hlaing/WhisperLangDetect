# 🎙 Whisper Language Detection Tool  

[![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)  
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)  
[![Dependencies](https://img.shields.io/badge/dependencies-whisper%20%7C%20pydub%20%7C%20moviepy%20%7C%20numpy-orange)](https://pypi.org/)  
[![FFmpeg](https://img.shields.io/badge/ffmpeg-required-red.svg)](https://ffmpeg.org/)  

Automatically detects the spoken language of audio or video files using **OpenAI’s Whisper model**.  
Supports both **single-file** and **batch folder processing**, handles a wide range of formats, and gracefully skips files with no speech.  

---

## ✨ Key Features  

- **Automatic Language Detection** – Uses Whisper’s `detect_language()` function.  
- **Model Selection** – Supports `tiny`, `base`, `small`, `medium`, `large`.  
- **Multi-format Support** – Audio: `.mp3, .aac, .aiff, .amr, .flac, .m4a, .ogg, .wav, .wma`; Video: `.avi, .mkv, .mov, .mpeg, .mp4, .webm, .wmv`.  
- **Silence Detection** – Skips silent files automatically.  
- **Batch Folder Processing** – Process multiple files at once with structured JSON output.  

---

## 📦 Requirements  

- **Python**: 3.9+  

### Dependencies
```bash
pip install openai-whisper pydub moviepy==1.0.3 numpy
```  

Install **ffmpeg**:
```bash
sudo apt update && sudo apt install ffmpeg
```  

---

## 🚀 Usage

### Single file detection:
```bash
python3 whisper_lang_detect.py --file /path/to/sample.mp4
```

### Single file with specific model:
```bash
python3 whisper_lang_detect.py --file /path/to/sample.mp4 --model small
```

### Folder detection:
```bash
python3 whisper_lang_detect.py --folder /path/to/audio_video_folder/
```

### Folder detection with specific model:
```bash
python3 whisper_lang_detect.py --folder /path/to/audio_video_folder/ --model medium
```

---

## 🔹 Example JSON Output  

**Single File Example:**
```json
{
  "status": "0",
  "errormessage": "",
  "result": {
    "filename": "sample.mp4",
    "langcode": "en",
    "probability": 0.92
  }
}
```

**If no speech is detected:**
```json
{
  "status": "1",
  "errormessage": "No speech was detected to classify.",
  "result": {
    "filename": "silent.mp4",
    "langcode": "",
    "probability": "null"
  }
}
```

**Batch Folder Example:**
```json
{
  "result": [
    {
      "fileno": 1,
      "filename": "clip1.mp3",
      "langcode": "es",
      "probability": 0.87
    },
    {
      "fileno": 2,
      "filename": "clip2.mp4",
      "langcode": "fr",
      "probability": 0.90
    }
  ]
}
```

---

## 📚 Use Cases  

- **Multilingual Media Processing** – Detect language before transcription/translation.  
- **Dataset Preprocessing** – Label large datasets with language info.  
- **Content Compliance** – Verify uploaded media matches supported languages.  

---

## 📜 License  

This project is licensed under **MIT License** – see [LICENSE](LICENSE).  

---

