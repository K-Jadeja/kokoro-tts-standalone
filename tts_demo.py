"""Quick sanity test for Kokoro TTS. Run after setup.sh completes."""
import sys
import os

sys.path.insert(0, "src")

from sherpa_onnx_tts import TTSEngine  # noqa: E402

MODEL_DIR = "models/kokoro-multi-lang-v1_0"

tts = TTSEngine(
    model_type="kokoro",
    kokoro_model=f"{MODEL_DIR}/model.onnx",
    kokoro_voices=f"{MODEL_DIR}/voices.bin",
    kokoro_tokens=f"{MODEL_DIR}/tokens.txt",
    kokoro_lexicon=f"{MODEL_DIR}/lexicon-us-en.txt",
    kokoro_data_dir=f"{MODEL_DIR}/espeak-ng-data",
    kokoro_dict_dir=f"{MODEL_DIR}/dict",
    sid=6,      # af_sky voice
    speed=1.0,
)

os.makedirs("cache", exist_ok=True)
out = tts.generate_audio("Hello! Your Kokoro TTS is working correctly.", "test_output")
print(f"✅ Generated: {out}")
