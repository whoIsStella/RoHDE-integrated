# Model artifacts

`rohde-lc.pt` is the selected checkpoint carried forward from the previous robustness-model path:

```text
weight/ICELab/Mobilenet/Training_noise_testnoise/LC_LC/98.6816
```

The old repository tree contained many intermediate checkpoints. Only the selected checkpoint is kept here.

The historical filename suggests a recorded model score, but the current integration repository does not contain enough experiment metadata to independently reconstruct that number. It is therefore treated as an artifact identifier, not a verified benchmark claim.

ONNX files are generated locally with:

```bash
python -m runtime.export_onnx
```

Generated ONNX artifacts are not committed until the cleaned export path is revalidated.
