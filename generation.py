import sys
sys.path.insert(0, r"D:\python_packages")

import json
import re
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel


# ============================================================
# CONFIG
# ============================================================

# Specify the base path where voice and script JSON files are stored, and where all outputs will be saved
BASE_PATH = Path(__file__).parent / "Muse7/Chapter8"

VOICE_FILE = BASE_PATH / "voices.json"
SCRIPT_FILE = BASE_PATH / "script.json"

OUT_DIR = BASE_PATH / "audio_output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda:0"
DTYPE = torch.bfloat16
LANGUAGE = "English"

PAUSE_MS_BETWEEN_SEGMENTS = 300


# ============================================================
# LOAD JSON DATA
# ============================================================

with open(VOICE_FILE, "r", encoding="utf-8") as f:
    SPEAKER_STYLES = json.load(f)

with open(SCRIPT_FILE, "r", encoding="utf-8") as f:
    segments = json.load(f)


# ============================================================
# HELPERS
# ============================================================

def sanitize_filename(name: str) -> str:
    name = name.strip().lower()
    name = re.sub(r"[^a-z0-9._-]+", "_", name)
    name = re.sub(r"_+", "_", name)
    return name.strip("_")


def write_wav(path: Path, audio: np.ndarray, sr: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, sr)


def concat_audio(chunks, sr, pause_ms=300):
    pause = np.zeros(int(sr * pause_ms / 1000.0), dtype=np.float32)
    out = []
    for i, c in enumerate(chunks):
        out.append(c.astype(np.float32))
        if i < len(chunks) - 1:
            out.append(pause)
    return np.concatenate(out) if out else np.zeros(1, dtype=np.float32)


# ============================================================
# LOAD MODELS
# ============================================================

design_model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    device_map=DEVICE,
    dtype=DTYPE,
)

clone_model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    device_map=DEVICE,
    dtype=DTYPE,
)


# ============================================================
# CREATE VOICE PROMPTS
# ============================================================

voice_clone_prompts = {}
sample_rate_map = {}

for speaker, cfg in SPEAKER_STYLES.items():

    print(f"Designing voice: {speaker}")

    ref_wavs, sr = design_model.generate_voice_design(
        text=cfg["ref_text"],
        language=LANGUAGE,
        instruct=cfg["ref_instruct"],
    )

    ref_audio = ref_wavs[0]

    ref_path = OUT_DIR / "references" / f"{sanitize_filename(speaker)}.wav"
    write_wav(ref_path, ref_audio, sr)

    voice_prompt = clone_model.create_voice_clone_prompt(
        ref_audio=(ref_audio, sr),
        ref_text=cfg["ref_text"],
    )

    voice_clone_prompts[speaker] = voice_prompt
    sample_rate_map[speaker] = sr

print("Voices ready.")


# ============================================================
# GENERATE AUDIO
# ============================================================

all_audio = []
global_sr = None

for idx, seg in enumerate(segments, start=1):

    speaker = seg["speaker"]
    text = seg["text"]

    if speaker not in voice_clone_prompts:
        raise ValueError(f"Speaker '{speaker}' not defined in voices.json")

    print(f"Segment {idx}/{len(segments)} | {speaker}")

    wavs, sr = clone_model.generate_voice_clone(
        text=text,
        language=LANGUAGE,
        voice_clone_prompt=voice_clone_prompts[speaker],
    )

    wav = wavs[0].astype(np.float32)

    if global_sr is None:
        global_sr = sr
    elif sr != global_sr:
        raise ValueError("Sample rate mismatch")

    seg_dir = OUT_DIR / "segments" / sanitize_filename(speaker)
    seg_path = seg_dir / f"{idx:03d}_{sanitize_filename(speaker)}.wav"

    write_wav(seg_path, wav, sr)

    all_audio.append(wav)


print("Segments generated.")


# ============================================================
# COMBINE CHAPTER
# ============================================================

chapter_audio = concat_audio(
    all_audio,
    sr=global_sr,
    pause_ms=PAUSE_MS_BETWEEN_SEGMENTS,
)

write_wav(OUT_DIR / "podcast_full.wav", chapter_audio, global_sr)

print("Full podcast written.")