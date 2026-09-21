from __future__ import annotations

from pathlib import Path

import torch

from .config import CHECKPOINT_PATH, MODEL_INPUT_SHAPE, ONNX_PATH
from .model.mobilenetv2 import MobileNetV2


def _load_state_dict(path: Path) -> dict[str, torch.Tensor]:
    try:
        loaded = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        loaded = torch.load(path, map_location="cpu")

    if isinstance(loaded, dict) and "state_dict" in loaded:
        loaded = loaded["state_dict"]

    if not isinstance(loaded, dict):
        raise TypeError("checkpoint did not contain a state dict")

    cleaned = {}
    for key, value in loaded.items():
        cleaned[key.removeprefix("module.")] = value
    return cleaned


def export() -> Path:
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"checkpoint not found: {CHECKPOINT_PATH}")

    state = _load_state_dict(CHECKPOINT_PATH)
    linear_weight = state.get("linear.weight")
    if linear_weight is None or linear_weight.ndim != 2:
        raise RuntimeError("checkpoint is missing a valid linear.weight tensor")

    num_classes, checkpoint_features = linear_weight.shape
    model = MobileNetV2(
        num_classes=int(num_classes),
        input_layer=1,
        input_shape=MODEL_INPUT_SHAPE,
    )

    if model.linear.in_features != int(checkpoint_features):
        raise RuntimeError(
            "checkpoint classifier does not match the configured input shape: "
            f"checkpoint expects {checkpoint_features} features, "
            f"model produces {model.linear.in_features}"
        )

    model.load_state_dict(state, strict=True)
    model.eval()

    ONNX_PATH.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(1, 1, *MODEL_INPUT_SHAPE, dtype=torch.float32)

    torch.onnx.export(
        model,
        dummy,
        ONNX_PATH,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )

    return ONNX_PATH


if __name__ == "__main__":
    path = export()
    print(f"Exported {path}")
