# 🎙 Whisper Language Detection Tool  

[![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)  
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)  
[![Dependencies](https://img.shields.io/badge/dependencies-whisper%20%7C%20pydub%20%7C%20moviepy%20%7C%20numpy-orange)](https://pypi.org/)  
[![FFmpeg](https://img.shields.io/badge/ffmpeg-required-red.svg)](https://ffmpeg.org/)  

Automatically detects the spoken language of audio or video files using **OpenAI’s Whisper model**.  
Supports both **single-file** and **batch folder processing**, handles a wide range of formats, and gracefully skips files with no speech.  

---

## Key Features  

- **Automatic Language Detection** – Uses Whisper’s `detect_language()` function.  
- **Model Selection** – Supports `tiny`, `base`, `small`, `medium`, `large`.  
- **Multi-format Support** – Audio: `.mp3, .aac, .aiff, .amr, .flac, .m4a, .ogg, .wav, .wma`; Video: `.avi, .mkv, .mov, .mpeg, .mp4, .webm, .wmv`.  
- **Silence Detection** – Skips silent files automatically.  
- **Batch Folder Processing** – Process multiple files at once with structured JSON output.  

---
## Supported Languages

<details>
<summary>Click to expand/collapse the full table</summary>

| No. | Code | Language           |
|-----|------|--------------------|
| 1   | af   | Afrikaans          |
| 2   | am   | Amharic            |
| 3   | ar   | Arabic             |
| 4   | as   | Assamese           |
| 5   | az   | Azerbaijani        |
| 6   | ba   | Bashkir            |
| 7   | be   | Belarusian         |
| 8   | bg   | Bulgarian          |
| 9   | bn   | Bengali            |
| 10  | bo   | Tibetan            |
| 11  | br   | Breton             |
| 12  | bs   | Bosnian            |
| 13  | ca   | Catalan            |
| 14  | cs   | Czech              |
| 15  | cy   | Welsh              |
| 16  | da   | Danish             |
| 17  | de   | German             |
| 18  | el   | Greek              |
| 19  | en   | English            |
| 20  | es   | Spanish            |
| 21  | et   | Estonian           |
| 22  | eu   | Basque             |
| 23  | fa   | Persian            |
| 24  | fi   | Finnish            |
| 25  | fo   | Faroese            |
| 26  | fr   | French             |
| 27  | gl   | Galician           |
| 28  | gu   | Gujarati           |
| 29  | ha   | Hausa              |
| 30  | haw  | Hawaiian           |
| 31  | he   | Hebrew             |
| 32  | hi   | Hindi              |
| 33  | hr   | Croatian           |
| 34  | ht   | Haitian Creole     |
| 35  | hu   | Hungarian          |
| 36  | hy   | Armenian           |
| 37  | id   | Indonesian         |
| 38  | is   | Icelandic          |
| 39  | it   | Italian            |
| 40  | ja   | Japanese           |
| 41  | jw   | Javanese           |
| 42  | ka   | Georgian           |
| 43  | kk   | Kazakh             |
| 44  | km   | Khmer              |
| 45  | kn   | Kannada            |
| 46  | ko   | Korean             |
| 47  | la   | Latin              |
| 48  | lb   | Luxembourgish      |
| 49  | ln   | Lingala            |
| 50  | lo   | Lao                |
| 51  | lt   | Lithuanian         |
| 52  | lv   | Latvian            |
| 53  | mg   | Malagasy           |
| 54  | mi   | Maori              |
| 55  | mk   | Macedonian         |
| 56  | ml   | Malayalam          |
| 57  | mn   | Mongolian          |
| 58  | mr   | Marathi            |
| 59  | ms   | Malay              |
| 60  | mt   | Maltese            |
| 61  | my   | Myanmar            |
| 62  | ne   | Nepali             |
| 63  | nl   | Dutch              |
| 64  | nn   | Norwegian Nynorsk  |
| 65  | no   | Norwegian          |
| 66  | oc   | Occitan            |
| 67  | pa   | Punjabi            |
| 68  | pl   | Polish             |
| 69  | ps   | Pashto             |
| 70  | pt   | Portuguese         |
| 71  | ro   | Romanian           |
| 72  | ru   | Russian            |
| 73  | sa   | Sanskrit           |
| 74  | sd   | Sindhi             |
| 75  | si   | Sinhala            |
| 76  | sk   | Slovak             |
| 77  | sl   | Slovenian          |
| 78  | sn   | Shona              |
| 79  | so   | Somali             |
| 80  | sq   | Albanian           |
| 81  | sr   | Serbian            |
| 82  | su   | Sundanese          |
| 83  | sv   | Swedish            |
| 84  | sw   | Swahili            |
| 85  | ta   | Tamil              |
| 86  | te   | Telugu             |
| 87  | tg   | Tajik              |
| 88  | th   | Thai               |
| 89  | tk   | Turkmen            |
| 90  | tl   | Tagalog            |
| 91  | tr   | Turkish            |
| 92  | tt   | Tatar              |
| 93  | uk   | Ukrainian          |
| 94  | ur   | Urdu               |
| 95  | uz   | Uzbek              |
| 96  | vi   | Vietnamese         |
| 97  | yi   | Yiddish            |
| 98  | yo   | Yoruba             |
| 99  | zh   | Chinese            |
| 100 | yue  | Cantonese          |


</details>

---

## Requirements  

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

## Usage

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

## Use Cases  

- **Multilingual Media Processing** – Detect language before transcription/translation.  
- **Dataset Preprocessing** – Label large datasets with language info.  
- **Content Compliance** – Verify uploaded media matches supported languages.  

---

## References

- [OpenAI Whisper GitHub Repository](https://github.com/openai/whisper)  
- [Whisper: Robust Speech Recognition via Large-Scale Weak Supervision (arXiv, 2022)](https://arxiv.org/pdf/2212.04356)  
- [Whisper API Documentation – Supported Languages](https://whisper-api.com/docs/languages/#_top)  

---

## License  

This project is licensed under **MIT License** – see [LICENSE](LICENSE).  

---

