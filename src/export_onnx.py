"""Export the trained two-head CRNN to ONNX for Raspberry Pi inference.

Run from the repo root:
    python src/export_onnx.py            # models/crnn.onnx (FP32, verified)
    python src/export_onnx.py --int8     # additionally models/crnn_int8.onnx

The exported model has two outputs: intent_logits and slot_logits.
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from features import load_intents_slots
from model import CRNN

ROOT = Path(__file__).resolve().parents[1]


def export_model(model, dummy, out_path):
    """Export to ONNX, silencing two expected, harmless warnings:

    1. torch >= 2.9 deprecates the legacy TorchScript exporter (we use it
       on purpose: identical behavior across torch versions).
    2. The generic GRU batch-size caution. Inference always uses a fixed
       batch of 1 with shape (1, 64, 480), so it does not apply here.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", message=".*batch_size other than 1.*")
        try:
            torch.onnx.export(model, dummy, out_path,
                              input_names=["logmel"],
                              output_names=["intent_logits", "slot_logits"],
                              opset_version=17, dynamo=False)
        except TypeError:  # older torch versions have no 'dynamo' argument
            torch.onnx.export(model, dummy, out_path,
                              input_names=["logmel"],
                              output_names=["intent_logits", "slot_logits"],
                              opset_version=17)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--int8", action="store_true",
                        help="also create a dynamically quantized model")
    args = parser.parse_args()

    intents, slots, _ = load_intents_slots(ROOT / "data")
    model = CRNN(n_intents=len(intents), n_slots=len(slots))
    model.load_state_dict(torch.load(ROOT / "models" / "crnn.pt",
                                     map_location="cpu"))
    model.eval()

    dummy = torch.randn(1, 1, 64, 480)          # one log-mel spectrogram
    out_path = ROOT / "models" / "crnn.onnx"

    export_model(model, dummy, out_path)
    print(f"exported {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")

    # quick check: the ONNX model must match the PyTorch model (both heads)
    session = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    torch_i, torch_s = model(dummy)
    onnx_i, onnx_s = session.run(None, {"logmel": dummy.numpy()})
    print(f"max |torch - onnx| intent = "
          f"{np.abs(torch_i.detach().numpy() - onnx_i).max():.2e}")
    print(f"max |torch - onnx| slot   = "
          f"{np.abs(torch_s.detach().numpy() - onnx_s).max():.2e}")

    if args.int8:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            from onnxruntime.quantization import quantize_dynamic, QuantType
        int8_path = ROOT / "models" / "crnn_int8.onnx"
        quantize_dynamic(str(out_path), str(int8_path), weight_type=QuantType.QInt8)
        print(f"exported {int8_path} ({int8_path.stat().st_size / 1e6:.1f} MB)")
        print("check its accuracy on the test set before deploying it")


if __name__ == "__main__":
    main()