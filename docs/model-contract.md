# Model contract

## Input

The deployment runtime uses one normalized EMG window with shape:

```text
[batch, input_channels, emg_channels, samples]
[1,     1,              8,            24]
```

The eight EMG channels come directly from the Myo armband. No synthetic channel tiling is performed.

## Normalization

`runtime/scaling_params.json` contains one mean and standard deviation pair for each live EMG channel.

`runtime.preprocessing.prepare_model_input` applies channel-wise z-score normalization and adds the batch/input-channel dimensions expected by ONNX Runtime.

## Export

`runtime.export_onnx` reads `linear.weight` from the selected PyTorch checkpoint to determine the saved classifier dimensions.

The export is rejected if the configured 8 x 24 input produces a feature count that does not match the checkpoint's classifier input width.

The classifier head is never replaced during export.

## Output labels

The output class count comes from the checkpoint.

A validated human-readable label map for this exact checkpoint is not currently present in the repository. The live runtime therefore reports generic class IDs.

This is deliberate. The older real-time gesture list and the robustness classifier configuration do not agree closely enough to infer a mapping safely.
