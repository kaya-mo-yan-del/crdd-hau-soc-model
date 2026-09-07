import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from configs import config
from src.models.builder import build_model


def main():
    pth_path = config.MODEL_DIR / "crdd_model_v1.pth"
    onnx_path = config.MODEL_DIR / "crdd_model_v1.onnx"

    if not pth_path.exists():
        print(f"ERROR: {pth_path} not found. Train the model first (02_train.py).")
        sys.exit(1)

    # Export runs on CPU, matching the Pi's runtime environment
    device = torch.device("cpu")
    model = build_model(config.NUM_CLASSES, pretrained=False)
    model.load_state_dict(torch.load(pth_path, map_location=device))
    model.eval()  # freeze dropout/batchnorm before export

    # Dummy input defines the graph's fixed shape: (1, 3, 224, 224)
    dummy = torch.randn(1, 3, config.IMAGE_SIZE, config.IMAGE_SIZE)
    torch.onnx.export(
        model, dummy, str(onnx_path),
        input_names=["input"],
        output_names=["logits"],
        opset_version=17,
        do_constant_folding=True,
    )
    size_mb = onnx_path.stat().st_size / 1e6
    print(f"Exported: {onnx_path}  ({size_mb:.1f} MB)")

    # Verify: PyTorch output vs ONNX output on the same input
    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    with torch.no_grad():
        torch_out = model(dummy).numpy()
    onnx_out = sess.run(["logits"], {"input": dummy.numpy()})[0]

    max_diff = float(np.abs(torch_out - onnx_out).max())
    print(f"Max output difference PyTorch vs ONNX: {max_diff:.2e}")
    if max_diff < 1e-4:
        print("VERIFIED: ONNX model matches the PyTorch model. Safe to deploy.")
    else:
        print("WARNING: outputs differ more than expected — do not deploy; "
              "check torch/onnx versions.")

    print("\nNext: copy models/saved/crdd_model_v1.onnx and the pi_deploy/ folder "
          "to the Raspberry Pi, then follow pi_deploy/README_PI_SETUP.md.")


if __name__ == "__main__":
    main()
