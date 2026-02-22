# Kokoro TTS Integration Guide for OpenClaw

> This document explains exactly how the Kokoro TTS pipeline works, how it differs
> from a standard sherpa-onnx VITS setup, and how to integrate it into an existing
> chatbot that may already have its own sherpa-onnx code.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [How Kokoro Differs from Standard VITS in sherpa-onnx](#2-how-kokoro-differs-from-standard-vits-in-sherpa-onnx)
3. [Model Files Explained](#3-model-files-explained)
4. [The Minimal Code Path (Text → WAV)](#4-the-minimal-code-path-text--wav)
5. [Configuration Parameters Reference](#5-configuration-parameters-reference)
6. [Integration Strategies](#6-integration-strategies)
7. [Common Pitfalls](#7-common-pitfalls)
8. [Async Usage](#8-async-usage)
9. [Quick Copy-Paste Minimal Example](#9-quick-copy-paste-minimal-example)

---

## 1. Architecture Overview

```
Text Input
    │
    ▼
┌──────────────────────┐
│  Text Preprocessing   │  Fix contractions, clean markup
│  (Python, in-code)    │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│   sherpa-onnx         │  C++ engine with Python bindings
│   OfflineTts          │
│                       │
│  ┌─────────────────┐  │
│  │ espeak-ng data   │  │  text → phoneme conversion
│  │ + lexicon .txt   │  │  (lexicon overrides espeak for listed words)
│  │ + tokens.txt     │  │  phoneme → token ID mapping
│  └────────┬────────┘  │
│           │            │
│  ┌────────▼────────┐  │
│  │ model.onnx       │  │  Kokoro neural net (token IDs → mel → audio)
│  │ + voices.bin     │  │  voice style embeddings (selects speaker via sid)
│  └────────┬────────┘  │
│           │            │
│  Output: samples[]     │  raw float32 PCM audio + sample_rate
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│   soundfile.write()   │  Write to .wav file
└──────────────────────┘
```

**Key point:** sherpa-onnx handles the entire text→phonemes→tokens→audio pipeline
internally in C++. Your Python code only needs to:

1. Configure it (file paths)
2. Call `tts.generate(text, sid=N)`
3. Write the result to a file

---

## 2. How Kokoro Differs from Standard VITS in sherpa-onnx

If OpenClaw already has a sherpa-onnx VITS implementation, here are the **critical differences**:

### Config class is different

```python
# ❌ VITS config (what you might already have)
sherpa_onnx.OfflineTtsModelConfig(
    vits=sherpa_onnx.OfflineTtsVitsModelConfig(
        model="path/to/model.onnx",
        lexicon="path/to/lexicon.txt",
        tokens="path/to/tokens.txt",
        data_dir="path/to/espeak-ng-data",
    ),
)

# ✅ Kokoro config (what this repo uses)
sherpa_onnx.OfflineTtsModelConfig(
    kokoro=sherpa_onnx.OfflineTtsKokoroModelConfig(
        model="path/to/model.onnx",
        voices="path/to/voices.bin",       # ← Kokoro-specific, VITS doesn't have this
        tokens="path/to/tokens.txt",
        lexicon="path/to/lexicon.txt",
        data_dir="path/to/espeak-ng-data",
        dict_dir="path/to/dict",           # ← optional, for Chinese text (Jieba)
        length_scale=1.0,                  # ← Kokoro uses length_scale, NOT speed
    ),
)
```

### Key differences table

| Feature | VITS setup | Kokoro setup |
|---------|-----------|-------------|
| Config class | `OfflineTtsVitsModelConfig` | `OfflineTtsKokoroModelConfig` |
| Voice file | Not needed | **`voices.bin` required** |
| Speed control | `speed` param in `generate()` | `length_scale` in config (not in generate) |
| Lexicon | Optional | Optional but recommended for quality |
| `data_dir` | espeak-ng-data | Same espeak-ng-data |
| `dict_dir` | Jieba dict (optional) | Jieba dict (optional) |
| Generate call | `tts.generate(text, sid=N, speed=S)` | `tts.generate(text, sid=N)` — no speed param |

### Speed control is set at init, not at generate time

```python
# VITS: speed is per-call
audio = tts.generate("Hello", sid=0, speed=1.2)

# Kokoro: speed is set once via length_scale in config
# Lower = faster. 1.0 = normal. 0.8 = faster. 1.2 = slower.
audio = tts.generate("Hello", sid=6)  # ← no speed param here
```

---

## 3. Model Files Explained

All files live in `models/kokoro-multi-lang-v1_0/`:

| File | Size | What it does | Modifiable? |
|------|------|-------------|-------------|
| `model.onnx` | ~300MB | The neural network. Takes token IDs + voice embedding → audio. | No |
| `voices.bin` | ~5MB | Contains 53 voice style embeddings (sid 0–52). Each is a vector that shapes the voice timbre. | No |
| `tokens.txt` | ~10KB | Maps phoneme symbols → integer IDs that the model understands. | No |
| `lexicon-us-en.txt` | ~5.8MB | Maps English words → phoneme sequences. sherpa-onnx looks up each word here FIRST, falls back to espeak-ng if not found. **This is the file you can customize** to control pronunciation. | **Yes** |
| `espeak-ng-data/` | ~3MB | Pre-compiled phoneme rules for 100+ languages. Used as fallback for words not in the lexicon. | No |
| `dict/` | Small | Jieba Chinese segmentation dictionary. Only needed if you generate Chinese speech. | No |

### How phoneme lookup works at runtime:

```
Word "hello"
    │
    ├─ 1. Check lexicon-us-en.txt → found: "h ə l ˈO" → use this
    │
    └─ 2. If not found → espeak-ng-data generates phonemes automatically
```

This is why the lexicon matters — it's the **override layer** for pronunciation quality.

---

## 4. The Minimal Code Path (Text → WAV)

Strip away the class hierarchy and this is what actually happens:

```python
import sherpa_onnx
import soundfile as sf

# 1. Configure
config = sherpa_onnx.OfflineTtsConfig(
    model=sherpa_onnx.OfflineTtsModelConfig(
        kokoro=sherpa_onnx.OfflineTtsKokoroModelConfig(
            model="models/kokoro-multi-lang-v1_0/model.onnx",
            voices="models/kokoro-multi-lang-v1_0/voices.bin",
            tokens="models/kokoro-multi-lang-v1_0/tokens.txt",
            lexicon="models/kokoro-multi-lang-v1_0/lexicon-us-en.txt",
            data_dir="models/kokoro-multi-lang-v1_0/espeak-ng-data",
            dict_dir="models/kokoro-multi-lang-v1_0/dict",
            length_scale=1.0,
        ),
        provider="cpu",
        num_threads=1,
        debug=False,
    ),
    max_num_sentences=1,
)

# 2. Validate
assert config.validate(), "Invalid config — check file paths"

# 3. Create engine (loads model into memory, ~2-3 seconds)
tts = sherpa_onnx.OfflineTts(config)

# 4. Generate (fast, ~100-300ms for a sentence on CPU)
audio = tts.generate("Hello, how are you today?", sid=6)

# 5. Save
sf.write("output.wav", audio.samples, samplerate=audio.sample_rate, subtype="PCM_16")
```

**That's it.** 25 lines of real code. Everything else in `sherpa_onnx_tts.py` is:
- Contraction preprocessing (fixing "don 't" → "don't")
- Auto-download logic
- File path management
- VITS fallback support (you don't need this)

---

## 5. Configuration Parameters Reference

### OfflineTtsKokoroModelConfig

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `model` | str | **Yes** | - | Path to `model.onnx` |
| `voices` | str | **Yes** | - | Path to `voices.bin` |
| `tokens` | str | **Yes** | - | Path to `tokens.txt` |
| `lexicon` | str | No | "" | Path to lexicon file(s). Multiple files: comma-separated |
| `data_dir` | str | No | "" | Path to `espeak-ng-data/` directory |
| `dict_dir` | str | No | "" | Path to `dict/` (Jieba, for Chinese) |
| `length_scale` | float | No | 1.0 | Speed: 1.0=normal, <1.0=faster, >1.0=slower |

### OfflineTtsModelConfig

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `provider` | str | No | "cpu" | `"cpu"`, `"cuda"`, or `"coreml"` |
| `num_threads` | int | No | 1 | CPU threads for inference |
| `debug` | bool | No | False | Print debug info from C++ layer |

### OfflineTtsConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_num_sentences` | int | 1 | Sentences per batch. 1 is safest. -1 for all at once. |
| `rule_fsts` | str | "" | Path to FST rules for number/date expansion (optional) |

### tts.generate()

| Parameter | Type | Description |
|-----------|------|-------------|
| `text` | str | The text to speak |
| `sid` | int | Speaker voice ID (0–54 for this model) |

Returns an object with:
- `.samples` → `list[float]` — raw PCM audio
- `.sample_rate` → `int` — always 24000 for Kokoro

---

## 6. Integration Strategies

### Strategy A: Replace existing sherpa-onnx VITS with Kokoro (Recommended)

If OpenClaw already creates a `sherpa_onnx.OfflineTts` object somewhere, you just need to change the config:

```python
# Find where OpenClaw does something like this:
config = sherpa_onnx.OfflineTtsConfig(
    model=sherpa_onnx.OfflineTtsModelConfig(
        vits=sherpa_onnx.OfflineTtsVitsModelConfig(...)  # ← OLD
    )
)

# Replace with:
config = sherpa_onnx.OfflineTtsConfig(
    model=sherpa_onnx.OfflineTtsModelConfig(
        kokoro=sherpa_onnx.OfflineTtsKokoroModelConfig(  # ← NEW
            model="models/kokoro-multi-lang-v1_0/model.onnx",
            voices="models/kokoro-multi-lang-v1_0/voices.bin",
            tokens="models/kokoro-multi-lang-v1_0/tokens.txt",
            lexicon="models/kokoro-multi-lang-v1_0/lexicon-us-en.txt",
            data_dir="models/kokoro-multi-lang-v1_0/espeak-ng-data",
            dict_dir="models/kokoro-multi-lang-v1_0/dict",
            length_scale=1.0,
        )
    ),
    max_num_sentences=1,
)
```

Then also update any `tts.generate()` calls:
```python
# OLD (VITS):
audio = tts.generate(text, sid=0, speed=1.0)

# NEW (Kokoro):
audio = tts.generate(text, sid=6)  # No speed param
```

### Strategy B: Use the TTSEngine wrapper class from this repo

Copy `src/sherpa_onnx_tts.py`, `src/tts_interface.py`, and `src/tts_model_utils.py` into OpenClaw's codebase, then:

```python
from sherpa_onnx_tts import TTSEngine

tts = TTSEngine(
    model_type="kokoro",
    kokoro_model="models/kokoro-multi-lang-v1_0/model.onnx",
    kokoro_voices="models/kokoro-multi-lang-v1_0/voices.bin",
    kokoro_tokens="models/kokoro-multi-lang-v1_0/tokens.txt",
    kokoro_lexicon="models/kokoro-multi-lang-v1_0/lexicon-us-en.txt",
    kokoro_data_dir="models/kokoro-multi-lang-v1_0/espeak-ng-data",
    kokoro_dict_dir="models/kokoro-multi-lang-v1_0/dict",
    sid=6,
    speed=1.0,
)

wav_path = tts.generate_audio("Hello world")  # → "cache/temp.wav"
```

### Strategy C: Bare minimum — just the 25 lines

If OpenClaw has its own audio handling and you just need the sherpa-onnx calls, use the code from [Section 4](#4-the-minimal-code-path-text--wav) directly. No wrapper classes needed.

---

## 7. Common Pitfalls

### 1. Using `vits=` instead of `kokoro=` in config

```python
# ❌ WRONG — will fail or produce garbage
sherpa_onnx.OfflineTtsModelConfig(
    vits=sherpa_onnx.OfflineTtsVitsModelConfig(model="kokoro-model.onnx", ...)
)

# ✅ RIGHT
sherpa_onnx.OfflineTtsModelConfig(
    kokoro=sherpa_onnx.OfflineTtsKokoroModelConfig(model="kokoro-model.onnx", ...)
)
```

The model architectures are different. A Kokoro .onnx **cannot** be loaded as VITS.

### 2. Passing `speed=` to `tts.generate()` with Kokoro

```python
# ❌ WRONG — Kokoro ignores this, or it errors
audio = tts.generate("Hello", sid=6, speed=1.2)

# ✅ RIGHT — speed is set via length_scale in config
audio = tts.generate("Hello", sid=6)
```

### 3. Forgetting `voices.bin`

VITS models don't need a voices file. Kokoro **requires** it. Missing `voices.bin` → config validation failure. 

### 4. File paths must be relative to CWD or absolute

sherpa-onnx resolves paths from the current working directory, not from the Python file location. If your process starts from a different directory, use absolute paths:

```python
import os
BASE = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE, "models", "kokoro-multi-lang-v1_0", "model.onnx")
```

### 5. `config.validate()` returns False

Always check this. It means one or more file paths are wrong. Enable `debug=True` to see which file sherpa-onnx can't find:

```python
sherpa_onnx.OfflineTtsModelConfig(
    kokoro=sherpa_onnx.OfflineTtsKokoroModelConfig(...),
    debug=True,  # ← prints file lookup details
)
```

### 6. sherpa-onnx version mismatch

This repo is pinned to `sherpa-onnx==1.12.12`. The `OfflineTtsKokoroModelConfig` class was added in a specific version. If OpenClaw uses an older version, that class won't exist. Check with:

```python
import sherpa_onnx
print(sherpa_onnx.__version__)  # Needs to be >= 1.10.x (Kokoro support)
```

If upgrading isn't possible, you would need the alternative `onnxruntime` approach (not recommended — much more complex).

### 7. Audio output is empty (0 samples)

Usually means the text was too short or the phoneme conversion failed. Preprocess:

```python
# Ensure text has proper sentence endings
if not text.endswith(('.', '!', '?')):
    text += '.'

# Ensure text isn't empty
if not text.strip():
    return None
```

### 8. Contractions sound garbled

The espeak-ng phonemizer sometimes splits contractions wrong. That's why `sherpa_onnx_tts.py` includes `_preprocess_text_for_contractions()` — it normalizes "don ' t" → "don't" before sending to the engine.

---

## 8. Async Usage

sherpa-onnx is a synchronous C++ library. For async frameworks (FastAPI, etc.):

```python
import asyncio

async def speak(tts, text: str, output: str) -> str:
    """Run TTS in a thread pool to avoid blocking the event loop."""
    return await asyncio.to_thread(tts.generate, text, 6)  # sid=6

# Then write the file from the result:
audio = await speak(tts, "Hello", "output.wav")
sf.write("output.wav", audio.samples, samplerate=audio.sample_rate, subtype="PCM_16")
```

Or using the wrapper class:

```python
path = await asyncio.to_thread(tts.generate_audio, "Hello world", "output")
```

---

## 9. Quick Copy-Paste Minimal Example

Zero-dependency-on-this-repo example. Just needs `pip install sherpa-onnx soundfile`:

```python
"""
Minimal Kokoro TTS — copy this into any project.
Requires: pip install sherpa-onnx==1.12.12 soundfile
Model: run setup.sh first, or download manually from
  https://github.com/K-Jadeja/tts-models/releases/download/v1.1/kokoro-multi-lang-v1_0.tar.bz2
"""
import sherpa_onnx
import soundfile as sf

MODEL = "models/kokoro-multi-lang-v1_0"

config = sherpa_onnx.OfflineTtsConfig(
    model=sherpa_onnx.OfflineTtsModelConfig(
        kokoro=sherpa_onnx.OfflineTtsKokoroModelConfig(
            model=f"{MODEL}/model.onnx",
            voices=f"{MODEL}/voices.bin",
            tokens=f"{MODEL}/tokens.txt",
            lexicon=f"{MODEL}/lexicon-us-en.txt",
            data_dir=f"{MODEL}/espeak-ng-data",
            dict_dir=f"{MODEL}/dict",
            length_scale=1.0,
        ),
        provider="cpu",
        num_threads=1,
    ),
    max_num_sentences=1,
)

assert config.validate(), "Bad config — check model file paths!"
tts = sherpa_onnx.OfflineTts(config)

def speak(text: str, filename: str = "output.wav", sid: int = 6) -> str:
    """Generate a WAV file from text. Returns the filename."""
    if not text.strip():
        return None
    if not text.rstrip().endswith(('.', '!', '?')):
        text = text.rstrip() + '.'
    audio = tts.generate(text, sid=sid)
    if len(audio.samples) == 0:
        return None
    sf.write(filename, audio.samples, samplerate=audio.sample_rate, subtype="PCM_16")
    return filename

# Usage:
speak("Hello! This is Kokoro TTS running locally.", "hello.wav")
```

---

## Summary for the OpenClaw Agent

1. **Run `setup.sh`** to download the model files (~333MB) into `models/kokoro-multi-lang-v1_0/`
2. **The key difference from VITS**: use `OfflineTtsKokoroModelConfig` not `OfflineTtsVitsModelConfig`, and provide `voices.bin`
3. **The custom `lexicon-us-en.txt`** in this repo is tracked in git — it may differ from the download
4. **Speed is set at config time** via `length_scale`, not at `generate()` time
5. **sherpa-onnx >= 1.12.12** is required for Kokoro support
6. **Output is always 24kHz WAV**, PCM float32 samples
7. **sid=6** is the default voice (af_nicole, American female). Range 0–52 available.

---

## 10. Voice ID Reference (Kokoro v1.0 — 53 speakers)

| SID | Voice | Language | Gender | Notes |
|-----|-------|----------|--------|---------|
| 0 | af_alloy | American English | female | |
| 1 | af_aoede | American English | female | |
| 2 | af_bella | American English | female | Soft, warm tone |
| 3 | af_heart | American English | female | |
| 4 | af_jessica | American English | female | |
| 5 | af_kore | American English | female | |
| 6 | af_nicole | American English | female | **Default — clear, articulate** |
| 7 | af_nova | American English | female | |
| 8 | af_river | American English | female | |
| 9 | af_sarah | American English | female | |
| 10 | af_sky | American English | female | Bright, energetic |
| 11 | am_adam | American English | male | |
| 12 | am_echo | American English | male | |
| 13 | am_eric | American English | male | |
| 14 | am_fenrir | American English | male | |
| 15 | am_liam | American English | male | |
| 16 | am_michael | American English | male | |
| 17 | am_onyx | American English | male | |
| 18 | am_puck | American English | male | |
| 19 | am_santa | American English | male | |
| 20 | bf_alice | British English | female | |
| 21 | bf_emma | British English | female | |
| 22 | bf_isabella | British English | female | British accent |
| 23 | bf_lily | British English | female | |
| 24 | bm_daniel | British English | male | |
| 25 | bm_fable | British English | male | |
| 26 | bm_george | British English | male | |
| 27 | bm_lewis | British English | male | |
| 28 | ef_dora | Spanish | female | |
| 29 | em_alex | Spanish | male | |
| 30 | ff_siwis | French | female | |
| 31 | hf_alpha | Hindi | female | |
| 32 | hf_beta | Hindi | female | |
| 33 | hm_omega | Hindi | male | |
| 34 | hm_psi | Hindi | male | |
| 35 | if_sara | Italian | female | |
| 36 | im_nicola | Italian | male | |
| 37 | jf_alpha | Japanese | female | |
| 38 | jf_gongitsune | Japanese | female | |
| 39 | jf_nezumi | Japanese | female | |
| 40 | jf_tebukuro | Japanese | female | |
| 41 | jm_kumo | Japanese | male | |
| 42 | pf_dora | Brazilian Portuguese | female | |
| 43 | pm_alex | Brazilian Portuguese | male | |
| 44 | pm_santa | Brazilian Portuguese | male | |
| 45 | zf_xiaobei | Mandarin Chinese | female | |
| 46 | zf_xiaoni | Mandarin Chinese | female | |
| 47 | zf_xiaoxiao | Mandarin Chinese | female | |
| 48 | zf_xiaoyi | Mandarin Chinese | female | |
| 49 | zm_yunjian | Mandarin Chinese | male | |
| 50 | zm_yunxi | Mandarin Chinese | male | |
| 51 | zm_yunxia | Mandarin Chinese | male | |
| 52 | zm_yunyang | Mandarin Chinese | male | |

> Source: [sherpa-onnx official docs](https://k2-fsa.github.io/sherpa/onnx/tts/pretrained_models/kokoro.html#kokoro-multi-lang-v1-0-chinese-english-53-speakers)
