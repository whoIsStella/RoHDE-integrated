"""
    Description: Compression of the model to ONNX format
    Author: Stella Parker @ SF State MIC Lab
    Date: 2025
"""
import torch
from model.mobilenetv2 import MobileNetV2
import os

num_classes = 8
input_layer = 1
height = 192
width = 24
model_path = "weight/ICELab/Mobilenet/Training_noise_testnoise/LC_LC/98.6816"
onnx_path = "weight/ICELab/Mobilenet/Training_noise_testnoise/LC_LC/98.6816.onnx"

# Make sure output directory exists
os.makedirs(os.path.dirname(onnx_path), exist_ok=True)

# Load model
model = MobileNetV2(num_classes=num_classes, input_layer=input_layer)
state_dict = torch.load(model_path, map_location=torch.device('cpu'))
state_dict.pop('linear.weight', None)
state_dict.pop('linear.bias', None)
model.load_state_dict(state_dict, strict=False)

# Adjust linear layer to match 192x24 input
with torch.no_grad():
    dummy_input = torch.randn(1, input_layer, height, width)
    features = model.bn2(model.conv2(model.layers(model.bn1(model.conv1(dummy_input)))))
    flatten_size = features.view(1, -1).size(1)
    model.linear = torch.nn.Linear(flatten_size, num_classes)

# Export ONNX
model.eval()
torch.onnx.export(model, dummy_input, onnx_path,
                  input_names=['input'], output_names=['output'],
                  dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
                  opset_version=11)

print("✅ Export complete. File saved?", os.path.exists(onnx_path), "\n→", onnx_path)





