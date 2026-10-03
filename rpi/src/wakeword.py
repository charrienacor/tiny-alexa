"""Wake-word detection with openWakeWord (local, lightweight, no ASR, no cloud).

Streams 80 ms microphone chunks into a small openWakeWord ONNX model and
returns as soon as the wake word is detected.
Default wake word: "hey alexa" (open model that ships with openWakeWord).
"""

import numpy as np
import sounddevice as sd
from openwakeword.model import Model

SAMPLE_RATE = 16000
CHUNK = 1280  # 80 ms at 16 kHz — the chunk size openWakeWord expects


def load_wake_word_model(model_name="alexa"):
    """Load the wake-word model once and reuse it (downloads on first use)."""
    return Model(wakeword_models=[model_name], inference_framework="onnx")


def wait_for_wake_word(model=None, model_name="alexa",
                       threshold=0.5, should_stop=None):
    """Block until the wake word is heard.

    Returns the detection score, or None if `should_stop()` became True.
    """
    if model is None:
        model = load_wake_word_model(model_name)
    print("listening for wake word...")
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                        dtype="int16", blocksize=CHUNK) as stream:
        while True:
            if should_stop is not None and should_stop():
                return None
            chunk, _ = stream.read(CHUNK)
            scores = model.predict(np.squeeze(chunk))
            score = scores[model_name] if isinstance(scores, dict) else float(scores)
            if score >= threshold:
                print(f"wake word detected (score {score:.2f})")
                try:
                    model.reset()  # clear internal buffers for the next detection
                except AttributeError:
                    pass
                return score