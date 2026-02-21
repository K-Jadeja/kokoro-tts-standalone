# kokoro-tts-standalone

Standalone Kokoro TTS engine extracted from [Open-LLM-VTuber](https://github.com/t41372/Open-LLM-VTuber).

Uses **sherpa-onnx** to run the Kokoro multilingual ONNX model fully offline on CPU.

## Setup (on VPS / Linux)

```bash
# 1. Install Python dependencies
pip install -r requirements.txt

# 2. Download the model (~333MB, one-time)
chmod +x setup.sh && ./setup.sh

# 3. Test it works
python tts_demo.py
# → cache/test_output.wav
```

## Usage

```python
import sys
sys.path.insert(0, "src")
from sherpa_onnx_tts import TTSEngine

tts = TTSEngine(
    model_type="kokoro",
    kokoro_model="models/kokoro-multi-lang-v1_0/model.onnx",
    kokoro_voices="models/kokoro-multi-lang-v1_0/voices.bin",
    kokoro_tokens="models/kokoro-multi-lang-v1_0/tokens.txt",
    kokoro_lexicon="models/kokoro-multi-lang-v1_0/lexicon-us-en.txt",
    kokoro_data_dir="models/kokoro-multi-lang-v1_0/espeak-ng-data",
    kokoro_dict_dir="models/kokoro-multi-lang-v1_0/dict",
    sid=6,    # Voice ID: 0-54. sid=6 = af_sky (American female)
    speed=1.0,
)

path = tts.generate_audio("Hello world", "my_output")
# → cache/my_output.wav
```

## Voice IDs (sid)

| sid | Voice | Style |
|-----|-------|-------|
| 0 | af | American Female (default) |
| 1 | af_bella | American Female |
| 6 | af_sky | American Female ← **default used** |
| 10 | am_adam | American Male |
| 11 | am_michael | American Male |
| 20 | bf_emma | British Female |
| 30 | bm_george | British Male |

## File layout

```
src/
  sherpa_onnx_tts.py     ← TTS engine
  tts_interface.py       ← abstract base
  tts_model_utils.py     ← auto-download helpers
models/
  kokoro-multi-lang-v1_0/
    lexicon-us-en.txt    ← custom lexicon (tracked in git)
    model.onnx           ← downloaded by setup.sh (not in git)
    voices.bin           ← downloaded by setup.sh
    tokens.txt           ← downloaded by setup.sh
    espeak-ng-data/      ← downloaded by setup.sh
requirements.txt
setup.sh
tts_demo.py
```
