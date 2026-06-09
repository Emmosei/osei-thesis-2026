# Speaker Time Tracker

Estimates how long each participant speaks in an audio or video recording.
Works with 2–4+ speakers. Detects overlapping speech and silence.

## Install

```bash
pip install -r requirements.txt
```

You also need **ffmpeg** for video files or non-WAV audio:
- Windows: https://ffmpeg.org/download.html  (add to PATH)
- Mac: `brew install ffmpeg`
- Linux: `sudo apt install ffmpeg`

## One-Time Setup (Hugging Face)

The diarization model requires a free account:

1. Sign up at https://huggingface.co
2. Accept model terms at https://huggingface.co/pyannote/speaker-diarization-3.1
3. Accept model terms at https://huggingface.co/pyannote/segmentation-3.0
4. Get your token at https://huggingface.co/settings/tokens

The model (~300MB) downloads automatically on first run and is cached locally.

## Usage

```bash
# Basic
python speaker_tracker.py --audio conversation.wav --token 

# Video file (audio is extracted automatically)
python speaker_tracker.py --audio meeting.mp4 --token hf_xxxx

# If you know the number of speakers (improves accuracy)
python speaker_tracker.py --audio conversation.wav --token hf_xxxx --speakers 3

# With visual timeline plot
python speaker_tracker.py --audio conversation.wav --token hf_xxxx --plot

# Save full report as JSON
python speaker_tracker.py --audio conversation.wav --token hf_xxxx --json

# Set token as environment variable instead (more convenient)
# Windows:  set HF_TOKEN=hf_xxxx
# Mac/Linux: export HF_TOKEN=hf_xxxx
python speaker_tracker.py --audio conversation.wav
```

## Example Output

```
── Speaker Report ────────────────────────────────────────────
  Total duration   : 120.0s
  Total speaking   : 108.3s
  Silence          : 11.7s
  Overlapping talk : 3.2s
  Speakers found   : 3

  SPEAKER_00
    █████████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 42.0%
    Speaking : 50.4s over 18 turns
    Longest  : 12.3s   Average: 2.8s

  SPEAKER_01
    ████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 24.5%
    Speaking : 29.4s over 11 turns
    Longest  : 8.1s    Average: 2.67s

  SPEAKER_02
    █████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 19.0%
    Speaking : 22.8s over 9 turns
    Longest  : 6.4s    Average: 2.53s
──────────────────────────────────────────────────────────────
```

## How It Works

1. **Audio extraction** — if the input is a video, ffmpeg pulls out the audio track as a 16kHz WAV
2. **Voice activity detection** — the model finds when anyone is speaking vs silence
3. **Speaker segmentation** — finds the exact moments one speaker hands off to another
4. **Speaker embedding** — each segment is converted to a voice "fingerprint" vector
5. **Clustering** — similar voice vectors are grouped into the same speaker identity
6. **Aggregation** — speaking time, turn counts, overlaps, and silences are calculated

## Tips

- **Known speaker count** (`--speakers N`) always improves accuracy — use it when you can
- **Clean audio** (low background noise, no music) gives the best results
- **Overlapping speech** is detected but attributed to the dominant speaker in that segment
- The model works best on recordings where speakers take clear turns
- For very long recordings (1hr+), expect a few minutes of processing time
