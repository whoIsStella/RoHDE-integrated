#***RoHDE Real-Time EMG Classification***

Real-time hand gesture classification using the Myo armband and a MobileNetV2 model trained on high-density EMG (HD-EMG) data. 
This system allows robust gesture recognition through ONNX inference on tiled Myo data for compatibility with HD-trained models.

**Features:**

Uses the Myo armband for real-time EMG streaming over BLE

Trained on 192-channel HD-EMG data, but deployable using Myo's 8-channel input

Tiling logic matches training-time input dimensionality

ONNX model inference for fast and portable execution

**System Requirements**

Python 3.8–3.11 (avoid 3.12 for compatibility)

Myo Armband with BLE support

Windows Bluetooth enabled

**Dependencies**

```
pip install bleak==0.20.2 onnxruntime numpy torch
```

On Windows, run this from Command Prompt or PowerShell — not WSL or Docker!

**Project Structure**

RoHDE-new/
├── realtime.py             # Real-time Bluetooth data stream + ONNX inference
├── model/
│   └── mobilenetv2.py      # MobileNetV2 architecture
├── dataset.py              # Training and export utilities
├── export_onnx.py          # Converts trained PyTorch model to ONNX
├── weight/
│   └── ...                 # Trained weights and ONNX models
├── scaling_params.json     # Mean/STD values for real-time standardization
├── WGAN-GP-train.py
├── EMG-Classifier.py
├── RoHDE.py
├── enviournment.yml
├── data/
│   └── ...



**Export your trained model to ONNXRun:**

```
python export_onnx.py
```

Connect the Myo armband and run the real-time inference

```
python realtime.py
```

Perform gesturesThe system will instruct you to perform Rest, Fist, Thumbs Up, and Ok Sign. After recording, real-time predictions will stream via console output.

**Training Compatibility**

Trained on HD-EMG (192 channels, 24 time steps)

Myo provides 8-channel data which is tiled to match the model’s expected 192-channel input



