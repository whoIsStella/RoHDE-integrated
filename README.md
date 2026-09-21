# RoHDE Real-Time EMG Classification

RoHDE-integrated is a real-time EMG inference prototype that connects a Myo armband over BLE to an ONNX model derived from the RoHDE robustness work.

The project bridges two earlier code paths:

- **RoHDE** focused on robust high-density EMG classification, including disturbance-aware training and WGAN-GP generated samples.
- **EffiE** provided the real-time Myo acquisition path: BLE streaming, short EMG windows, preprocessing, and live gesture-recognition workflow.
- **RoHDE-integrated** combines those ideas into one runtime: live Myo input, preprocessing, shape adaptation for the HD-EMG model, ONNX Runtime inference, and rolling prediction smoothing.

## Runtime pipeline

```text
Myo armband
    |
    | BLE via Bleak
    v
8-channel sEMG stream
    |
    | signed conversion + 24-sample window
    v
per-channel normalization
    |
    | compatibility adapter
    v
192 x 24 model input
    |
    v
ONNX Runtime
    |
    v
gesture prediction
    |
    v
5-sample majority vote
```

## What the live path does

`RoHDE-new/realtime.py` handles the current inference path.

1. Scans for a Myo armband and connects over BLE.
2. Reads the four EMG characteristics exposed by the device.
3. Converts the incoming bytes to signed EMG values.
4. Builds an 8-channel window of 24 samples.
5. Normalizes each channel using `scaling_params.json`.
6. Repeats the 8-channel window across the channel axis to produce a `192 x 24` tensor.
7. Adds batch and input-channel dimensions to produce `[1, 1, 192, 24]`.
8. Runs the tensor through ONNX Runtime.
9. Smooths predictions with a rolling five-prediction majority vote.

## Why the 8-to-192 channel adapter exists

The robustness model was built around 192-channel HD-EMG input, while the Myo armband exposes 8 EMG channels.

The live runtime uses `np.tile` to repeat the 8-channel Myo window until it matches the model's 192-channel input shape. This is a shape-compatibility adapter for deployment experiments. It does **not** make 8-channel Myo sensing equivalent to a true 192-channel HD-EMG array, and it does not recreate the missing spatial information.

That distinction matters: this repository explores whether a model built for the HD-EMG pipeline can be exercised against a much smaller live sensor interface without rewriting the entire inference stack.

## Relationship to the earlier code

### RoHDE

The RoHDE code path contains the robustness experiments: HD-EMG classifiers, disturbance conditions such as contact artifacts and loose contacts, and WGAN-GP tooling for generating synthetic disturbed EMG samples.

`RoHDE-new/EMG-Classifier.py`, `RoHDE-new/RoHDE.py`, and `RoHDE-new/WGAN-GP-train.py` preserve that side of the project.

### EffiE

The EffiE code path contains the real-time Myo workflow and the earlier 8-channel sEMG acquisition approach. A copy is retained under `IEEE-NER-2023-EffiE/` as reference material for the live acquisition lineage.

### Integration layer

The integration work lives primarily under `RoHDE-new/`:

- `realtime.py`: BLE acquisition, preprocessing, ONNX inference, and prediction smoothing
- `export_onnx.py`: PyTorch-to-ONNX export path
- `model/mobilenetv2.py`: MobileNetV2 classifier architecture
- `dataset.py`: HD-EMG loading and input-shape adaptation utilities
- `scaling_params.json`: per-channel normalization values for the live Myo input
- `weight/`: trained and exported model artifacts

## Running the live prototype

Requirements:

- Python 3.8 to 3.11
- Myo armband
- host Bluetooth LE access
- `bleak==0.20.2`
- `onnxruntime`
- `numpy`
- `torch`

Install the core dependencies:

```bash
pip install bleak==0.20.2 onnxruntime numpy torch
```

Then run from `RoHDE-new/`:

```bash
python realtime.py
```

Direct BLE access is required. WSL and containerized environments may not expose the host Bluetooth stack cleanly.

## Exporting a model to ONNX

```bash
cd RoHDE-new
python export_onnx.py
```

The export target is a `[batch, 1, 192, 24]` input tensor.

## Current integration debt

Two model-contract issues still need cleanup before this should be treated as a finished deployment path:

1. The robustness classifier/export path is configured for 8 output classes, while the current live gesture-name list contains 7 labels. The runtime detects an unmapped output instead of silently assigning the wrong gesture, but the class mapping needs to be reconciled.
2. The current ONNX export script rebuilds the final linear layer to fit the `192 x 24` feature shape. That changes the classifier head rather than preserving the trained head, so new exports need to be revalidated against the intended checkpoint before they are treated as equivalent to the trained model.

These are integration issues, not hidden assumptions. The BLE acquisition, preprocessing path, channel-shape adapter, ONNX runtime wiring, and prediction smoothing are all explicit in the repository.

## Status

This repository is an integration prototype for real-time EMG inference. The next cleanup pass should reconcile the class mapping and make the ONNX export preserve a validated trained classifier head.
