"""
    Description: Compression of the model to ONNX format
    Author: Stella Parker @ SF State MIC Lab
    Date: 2025
"""

import torch
from model.mobilenetv2 import MobileNetV2

batch_size = 1
input_layer = 1
#num_sensors = 192
num_classes = 8
window_size = 24



model_path = "weight/ICELab/Mobilenet/Training_noise_testnoise/LC_LC/98.6816"
onnx_path = "weight/ICELab/Mobilenet/Training_noise_testnoise/LC_LC/98.6816.onnx"

model = MobileNetV2(num_classes=num_classes, input_layer=input_layer)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()
dummy_input = torch.randn(1, input_layer, num_classes, window_size)

torch.onnx.export(model, dummy_input, onnx_path, input_names=['input'], output_names=['output'],
                  dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
                  opset_version=11, do_constant_folding=True)

print("ONNX model exported successfully.")



