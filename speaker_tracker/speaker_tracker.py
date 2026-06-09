"""
speaker_tracker.py
------------------
Estimates speaking time of each participant in an audio or video recording.
Supports 3-4 speakers (works for 2 or more).

Dependencies:
    pip install -r requirements.txt

Setup (one-time):
    1. Create a free account at https://huggingface.co
    2. Accept model terms at https://huggingface.co/pyannote/speaker-diarization-3.1
    3. Accept model terms at https://huggingface.co/pyannote/segmentation-3.0
    4. Generate an access token at https://huggingface.co/settings/tokens
    5. Pass it via --token or set env variable HF_TOKEN

Usage:
    python speaker_tracker.py --audio conversation.wav --token hf_xxxx
    python speaker_tracker.py --audio conversation.mp4 --token hf_xxxx  # video too
    python speaker_tracker.py --audio conversation.wav --token hf_xxxx --speakers 4
    python speaker_tracker.py --audio conversation.wav --token hf_xxxx --plot
"""

import os
import sys
import argparse
import wave
import json
import subprocess
from dataclasses import dataclass, field
from typing import Optional
from collections import defaultdict


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class Config:
    # Number of speakers — set if known for better accuracy, None to auto-detect
    num_speakers: Optional[int] = None
    min_speakers: int = 2
    max_speakers: int = 4

    # Hugging Face token (required for pyannote models)
    hf_token: Optional[str] = None

    # Output
    show_plot: bool = False
    save_json: bool = False


# ---------------------------------------------------------------------------
# Layer 1: Audio Extraction
# Handles both audio files and video files (extracts audio track)
# ---------------------------------------------------------------------------

class AudioExtractor:
    """
    Ensures we always have a clean .wav file to work with.
    If the input is a video, ffmpeg extracts the audio track.
    If the input is already a .wav, it's used directly.
    Other audio formats (mp3, m4a, etc.) are converted via ffmpeg.
    """

    SUPPORTED_AUDIO = {'.wav', '.mp3', '.m4a', '.flac', '.ogg', '.aac'}
    SUPPORTED_VIDEO = {'.mp4', '.mov', '.avi', '.mkv', '.webm'}

    def __init__(self, input_path: str):
        self.input_path = input_path
        self.temp_wav   = None

    def get_wav_path(self) -> str:
        ext = os.path.splitext(self.input_path)[1].lower()

        if ext == '.wav':
            return self.input_path  # already good

        if ext in self.SUPPORTED_AUDIO or ext in self.SUPPORTED_VIDEO:
            return self._convert_to_wav()

        raise ValueError(f"Unsupported file format: {ext}")

    def _convert_to_wav(self) -> str:
        out = self.input_path.rsplit('.', 1)[0] + '_extracted.wav'
        print(f"Converting {os.path.basename(self.input_path)} → WAV ...")

        cmd = [
            'ffmpeg', '-y',
            '-i', self.input_path,
            '-vn',                   # drop video track
            '-acodec', 'pcm_s16le',  # standard 16-bit PCM
            '-ar', '16000',          # 16kHz — optimal for speech models
            '-ac', '1',              # mono
            out,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"ffmpeg failed. Is it installed?\n"
                f"Install: https://ffmpeg.org/download.html\n"
                f"Error: {result.stderr[-300:]}"
            )
        self.temp_wav = out
        return out

    def get_duration(self, wav_path: str) -> float:
        with wave.open(wav_path, 'r') as wf:
            return wf.getnframes() / wf.getframerate()

    def cleanup(self):
        if self.temp_wav and os.path.exists(self.temp_wav):
            os.remove(self.temp_wav)


# ---------------------------------------------------------------------------
# Layer 2: Speaker Diarization
# Wraps pyannote.audio — handles "who spoke when"
# ---------------------------------------------------------------------------

class DiarizationEngine:
    """
    Runs pyannote speaker diarization pipeline.
    Returns a list of segments: (start, end, speaker_label).
    """

    MODEL = "pyannote/speaker-diarization-3.1"

    def __init__(self, config: Config):
        self.cfg = config
        self._pipeline = None

    def _load_pipeline(self):
        if self._pipeline is not None:
            return

        try:
            from pyannote.audio import Pipeline
        except ImportError:
            raise ImportError("Run: pip install pyannote.audio")

        token = self.cfg.hf_token or os.environ.get("HF_TOKEN")
        if not token:
            raise ValueError(
                "A Hugging Face token is required.\n"
                "  1. Sign up at https://huggingface.co\n"
                "  2. Accept terms at https://huggingface.co/pyannote/speaker-diarization-3.1\n"
                "  3. Accept terms at https://huggingface.co/pyannote/segmentation-3.0\n"
                "  4. Get token at https://huggingface.co/settings/tokens\n"
                "  5. Pass via --token hf_xxxx or set env variable HF_TOKEN"
            )

        print("Loading diarization model (first run downloads ~300MB) ...")
        self._pipeline = Pipeline.from_pretrained(self.MODEL, token=token)
        print("Model loaded.")

    def run(self, wav_path: str) -> list[tuple]:
        self._load_pipeline()

        print("Running speaker diarization ...")

        # Pre-load audio as waveform tensor to bypass torchcodec/AudioDecoder issues
        import torch
        import soundfile as sf

        waveform, sample_rate = sf.read(wav_path, dtype="float32", always_2d=True)
        # soundfile returns (samples, channels) — torch expects (channels, samples)
        waveform_tensor = torch.from_numpy(waveform.T)
        audio_input = {"waveform": waveform_tensor, "sample_rate": sample_rate}

        # Build kwargs — if num_speakers is known, pass it for better accuracy
        kwargs = {}
        if self.cfg.num_speakers:
            kwargs["num_speakers"] = self.cfg.num_speakers
        else:
            kwargs["min_speakers"] = self.cfg.min_speakers
            kwargs["max_speakers"] = self.cfg.max_speakers

        diarization = self._pipeline(audio_input, **kwargs)

        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append((
                round(turn.start, 3),
                round(turn.end, 3),
                speaker
            ))

        return sorted(segments, key=lambda x: x[0])


# ---------------------------------------------------------------------------
# Layer 3: Speaking Time Tracker
# Aggregates segments into per-speaker statistics
# ---------------------------------------------------------------------------

@dataclass
class SpeakerStats:
    label: str
    total_duration_s: float     = 0.0
    num_turns: int              = 0
    longest_turn_s: float       = 0.0
    turns: list                 = field(default_factory=list)

    @property
    def average_turn_s(self) -> float:
        return round(self.total_duration_s / self.num_turns, 2) if self.num_turns else 0.0


class SpeakingTimeTracker:
    """
    Takes raw diarization segments and builds per-speaker statistics.
    Also detects overlapping speech (interruptions).
    """

    def __init__(self, total_duration_s: float):
        self.total_duration = total_duration_s
        self.speakers: dict[str, SpeakerStats] = {}
        self.segments: list[tuple] = []
        self.overlap_duration_s: float = 0.0

    def process(self, segments: list[tuple]):
        self.segments = segments

        # Accumulate per-speaker stats
        for start, end, speaker in segments:
            duration = round(end - start, 3)
            if speaker not in self.speakers:
                self.speakers[speaker] = SpeakerStats(label=speaker)

            s = self.speakers[speaker]
            s.total_duration_s = round(s.total_duration_s + duration, 3)
            s.num_turns        += 1
            s.longest_turn_s   = round(max(s.longest_turn_s, duration), 3)
            s.turns.append({"start": start, "end": end, "duration": duration})

        # Detect overlapping speech (two speakers at same time)
        self.overlap_duration_s = self._compute_overlap(segments)

    def _compute_overlap(self, segments: list[tuple]) -> float:
        """Find total seconds where two or more speakers overlap."""
        if len(segments) < 2:
            return 0.0

        overlap = 0.0
        sorted_segs = sorted(segments, key=lambda x: x[0])

        for i in range(len(sorted_segs) - 1):
            a_start, a_end, a_spk = sorted_segs[i]
            b_start, b_end, b_spk = sorted_segs[i + 1]

            if a_spk != b_spk and b_start < a_end:
                overlap += round(min(a_end, b_end) - b_start, 3)

        return round(overlap, 2)

    def report(self) -> dict:
        total_speaking = sum(s.total_duration_s for s in self.speakers.values())
        silence_s      = round(self.total_duration - total_speaking + self.overlap_duration_s, 2)

        speakers_out = []
        for label, stats in sorted(self.speakers.items()):
            pct = round(100 * stats.total_duration_s / self.total_duration, 1)
            speakers_out.append({
                "speaker":          label,
                "total_speaking_s": round(stats.total_duration_s, 2),
                "percentage":       pct,
                "num_turns":        stats.num_turns,
                "longest_turn_s":   stats.longest_turn_s,
                "average_turn_s":   stats.average_turn_s,
                "turns":            stats.turns,
            })

        # Sort by speaking time descending
        speakers_out.sort(key=lambda x: x["total_speaking_s"], reverse=True)

        return {
            "total_duration_s":   round(self.total_duration, 2),
            "total_speaking_s":   round(total_speaking, 2),
            "silence_s":          max(0.0, silence_s),
            "overlap_s":          self.overlap_duration_s,
            "num_speakers":       len(self.speakers),
            "speakers":           speakers_out,
        }


# ---------------------------------------------------------------------------
# Layer 4: Visualizer (optional terminal + matplotlib plot)
# ---------------------------------------------------------------------------

class Visualizer:

    @staticmethod
    def print_report(report: dict):
        print("\n── Speaker Report " + "─" * 44)
        print(f"  Total duration   : {report['total_duration_s']}s")
        print(f"  Total speaking   : {report['total_speaking_s']}s")
        print(f"  Silence          : {report['silence_s']}s")
        print(f"  Overlapping talk : {report['overlap_s']}s")
        print(f"  Speakers found   : {report['num_speakers']}")
        print()

        for sp in report["speakers"]:
            bar_len = int(sp["percentage"] / 2)
            bar     = "█" * bar_len + "░" * (50 - bar_len)
            print(f"  {sp['speaker']}")
            print(f"    {bar} {sp['percentage']}%")
            print(f"    Speaking : {sp['total_speaking_s']}s over {sp['num_turns']} turns")
            print(f"    Longest  : {sp['longest_turn_s']}s   Average: {sp['average_turn_s']}s")
            print()

        print("─" * 62 + "\n")

    @staticmethod
    def plot_timeline(report: dict, total_duration: float):
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
        except ImportError:
            print("pip install matplotlib to enable timeline plot.")
            return

        speakers = [sp["speaker"] for sp in report["speakers"]]
        colors   = plt.cm.Set2.colors
        cmap     = {spk: colors[i % len(colors)] for i, spk in enumerate(speakers)}

        fig, (ax_timeline, ax_bar) = plt.subplots(
            2, 1, figsize=(14, 5),
            gridspec_kw={"height_ratios": [3, 1]}
        )
        fig.suptitle("Speaker Analysis", fontsize=14, fontweight="bold")

        # --- Timeline ---
        for i, sp in enumerate(report["speakers"]):
            for turn in sp["turns"]:
                ax_timeline.barh(
                    i, turn["duration"],
                    left=turn["start"],
                    color=cmap[sp["speaker"]],
                    edgecolor="white", linewidth=0.3,
                    height=0.6,
                )
        ax_timeline.set_yticks(range(len(speakers)))
        ax_timeline.set_yticklabels(speakers)
        ax_timeline.set_xlabel("Time (seconds)")
        ax_timeline.set_xlim(0, total_duration)
        ax_timeline.set_title("Speaking Timeline")
        ax_timeline.grid(axis="x", alpha=0.3)

        # --- Bar chart ---
        labels = [sp["speaker"] for sp in report["speakers"]]
        times  = [sp["total_speaking_s"] for sp in report["speakers"]]
        bars   = ax_bar.barh(labels, times,
                             color=[cmap[l] for l in labels], height=0.5)
        ax_bar.set_xlabel("Total speaking time (seconds)")
        ax_bar.set_title("Speaking Time per Speaker")
        for bar, sp in zip(bars, report["speakers"]):
            ax_bar.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                        f"{sp['percentage']}%", va="center", fontsize=9)

        plt.tight_layout()
        plt.savefig("speaker_timeline.png", dpi=150, bbox_inches="tight")
        print("  Timeline saved → speaker_timeline.png")
        plt.show()


# ---------------------------------------------------------------------------
# Layer 5: Orchestrator
# ---------------------------------------------------------------------------

def process_audio(input_path: str, config: Config) -> dict:
    # Step 1 — ensure we have a wav file
    extractor = AudioExtractor(input_path)
    wav_path  = extractor.get_wav_path()
    duration  = extractor.get_duration(wav_path)
    print(f"Audio duration: {duration:.1f}s")

    try:
        # Step 2 — run diarization
        engine   = DiarizationEngine(config)
        segments = engine.run(wav_path)

        print(f"Found {len(segments)} speaking turns.")

        # Step 3 — aggregate stats
        tracker = SpeakingTimeTracker(total_duration_s=duration)
        tracker.process(segments)
        report = tracker.report()

        # Step 4 — display
        Visualizer.print_report(report)

        if config.show_plot:
            Visualizer.plot_timeline(report, duration)

        if config.save_json:
            out_path = os.path.splitext(input_path)[0] + "_speaker_report.json"
            with open(out_path, "w") as f:
                json.dump(report, f, indent=2)
            print(f"  Report saved → {out_path}")

        return report

    finally:
        extractor.cleanup()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Estimate speaking time of each participant in an audio/video recording."
    )
    parser.add_argument("--audio",    required=True,
                        help="Path to audio or video file")
    parser.add_argument("--token",    default=None,
                        help="Hugging Face token (or set HF_TOKEN env var)")
    parser.add_argument("--speakers", type=int, default=None,
                        help="Number of speakers if known (improves accuracy)")
    parser.add_argument("--min",      type=int, default=2,
                        help="Minimum number of speakers to detect (default: 2)")
    parser.add_argument("--max",      type=int, default=4,
                        help="Maximum number of speakers to detect (default: 4)")
    parser.add_argument("--plot",     action="store_true",
                        help="Show and save a visual timeline")
    parser.add_argument("--json",     action="store_true",
                        help="Save full report as JSON")
    args = parser.parse_args()

    config = Config(
        num_speakers = args.speakers,
        min_speakers = args.min,
        max_speakers = args.max,
        hf_token     = args.token,
        show_plot    = args.plot,
        save_json    = args.json,
    )

    process_audio(args.audio, config)


if __name__ == "__main__":
    main()
