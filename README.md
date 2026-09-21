# RoHDE Real-Time EMG

Real-time EMG inference from a Myo armband using a robustness-trained MobileNetV2 checkpoint and ONNX Runtime.

This repository owns the deployment integration. It is no longer a copy of the earlier research trees or a storage location for raw datasets and intermediate checkpoints.

## Lineage

Two earlier code paths feed into this project:

- [RoHDE](https://github.com/whoIsStella/IEEE-NER-2023-RoHDE) contains the robustness work around disturbed high-density EMG, including WGAN-GP augmentation and classifier experiments.
- [EffiE](https://github.com/whoIsStella/IEEE-NER-2023-EffiE) contains the earlier real-time Myo acquisition path and BLE-based gesture-recognition workflow.
- **RoHDE-integrated** owns the live deployment path: Myo BLE input, preprocessing, checkpoint-preserving ONNX export, inference, and prediction smoothing.

The earlier repositories remain the source for training and research history. Their code and datasets are not duplicated here.

## Runtime pipeline

```text
Myo armband
    |
    | BLE
    v
two 8-channel EMG samples per notification
    |
    v
24-sample rolling window
    |
    v
per-channel normalization
    |
    v
[1, 1, 8, 24] tensor
    |
    v
ONNX Runtime
    |
    v
class prediction
    |
    v
5-prediction majority vote
```

## Model contract

The selected checkpoint is stored at:

```text
models/rohde-lc.pt
```

The current deployment contract is an **8 x 24** EMG window.

A previous integration attempt tiled 8 Myo channels to 192 channels and rebuilt the final classifier layer during ONNX export. That made the tensor shape fit but did not preserve the trained classifier head. The current export path removes that behavior.

`runtime/export_onnx.py` now:

1. reads the classifier dimensions directly from the checkpoint;
2. builds the model at the native 8 x 24 input shape;
3. checks that the feature count matches the saved classifier weights;
4. loads the checkpoint with `strict=True`;
5. exports only if the trained head can be preserved.

If the checkpoint and runtime shape do not agree, export fails instead of silently creating a new head.

## Class labels

The selected robustness checkpoint exposes class outputs, but the repository does not currently contain a validated mapping from those output indexes to the live gesture-name list used by the earlier Myo workflow.

The runtime therefore prints `class_0`, `class_1`, and so on instead of assigning an unverified gesture name.

That mapping should be added only after it can be traced back to the training labels for this checkpoint.

## Setup

Python 3.10 or 3.11 is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r runtime/requirements.txt
```

Export the selected checkpoint to ONNX:

```bash
python -m runtime.export_onnx
```

Then connect a Myo armband and run:

```bash
python -m runtime.realtime
```

The runtime needs direct access to the host Bluetooth LE stack.

## Repository layout

```text
runtime/
  config.py            paths and runtime shape constants
  preprocessing.py     packet decoding, normalization, input preparation
  realtime.py          BLE acquisition and ONNX inference
  export_onnx.py       checkpoint-preserving ONNX export
  model/               MobileNetV2 implementation
  scaling_params.json  per-channel normalization values

models/
  rohde-lc.pt          selected checkpoint

tests/
  hardware-independent preprocessing tests

docs/
  model-contract.md    deployment assumptions and unresolved label mapping
```

Raw EMG datasets, generated logs, caches, editor state, duplicated research trees, and intermediate checkpoints are intentionally not kept in the current repository tree.

## Verification

The CI surface is intentionally hardware-independent:

- Python syntax/compile checks
- preprocessing unit tests
- repository hygiene checks

BLE connectivity and live Myo behavior require physical hardware and are not claimed as CI-verified.

## Status

The repository now has a coherent runtime boundary and a reproducible export contract.

Remaining work:

- validate the checkpoint's class-index-to-gesture mapping;
- run the cleaned export against the selected checkpoint and record the ONNX model metadata;
- perform a hardware smoke test with a Myo armband;
- add latency measurements once the live path is revalidated.
