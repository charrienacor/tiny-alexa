"""Export the trained two-head TCN to ONNX for Raspberry Pi inference.

Run from the repo root:
    python src/export_onnx.py            # models/TCN.onnx (FP32, verified)
    python src/export_onnx.py --int8     # additionally models/TCN_int8.onnx

Outputs: intent_logits and slot_logits.
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from features import load_intents_slots
from model import TCN

ROOT = Path(__file__).resolve().parents[1]


def export_model(model, dummy, out_path):
    """Export to ONNX, silencing two expected, harmless warnings
    (TorchScript-exporter deprecation; the generic batch-size caution,
    which does not apply — inference is always batch 1).
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
    model = TCN(n_intents=len(intents), n_slots=len(slots))
    model.load_state_dict(torch.load(ROOT / "models" / "TCN.pt",
                                     map_location="cpu"))
    model.eval()

    dummy = torch.randn(1, 1, 64, 480)          # one log-mel spectrogram
    out_path = ROOT / "models" / "TCN.onnx"

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
        int8_path = ROOT / "models" / "TCN_int8.onnx"
        quantize_dynamic(str(out_path), str(int8_path), weight_type=QuantType.QInt8)
        print(f"exported {int8_path} ({int8_path.stat().st_size / 1e6:.1f} MB)")
        print("check its accuracy on the test set before deploying it")


if __name__ == "__main__":
    main()