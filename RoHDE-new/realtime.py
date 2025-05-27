## ##
## @original author: Amir Modan
## @editor: Jimmy L. @ SF State MIC Lab
##  - Date: Summer 2022

## Main Program for Real-Time system which establishes BLE connection,
##     defines GUI, and finetunes realtime samples from a pretrained finetune-base model.
    
## Flow: (running via Async Functions)
##     1. Run the code (python realtime.py)
##     2. Enable bluetooth in setting, and code with automatically pair with armband
##     3. Follow instructions to perform gestures for finetuning
##     4. Finetune training starts
##         - Optional: save finetuned-model
##     5. Real time gesture recognition begins

## Note:
##     1. Should see myo armband in blue lighting if connected.
##     2. Run this on Linux if possible, sometimes Bleak refuses to connect to Myo Armband under Windows environment.
## ##
from config import gesture_names as GESTURES
import asyncio
import time
import json
import nest_asyncio
nest_asyncio.apply()
import torch as pytorch
import numpy as np
import warnings
import platform
from typing import Any
from bleak import BleakClient, discover, BleakScanner
from dataset import dataset
from model.mobilenetv2 import MobileNetV2
import onnxruntime as ort
warnings.filterwarnings("ignore")
from collections import deque
gesture_buffer = deque(maxlen=5)
# new lines 41-44 this can go if it doesn't work
if platform.system() == "Windows":
    from bleak import _logger as logger
    logger.setLevel("DEBUG")

## UUID's for BLE Connection
CONTROL = "d5060401-a904-deb9-4748-2c7f4a124842"
EMG0 = "d5060105-a904-deb9-4748-2c7f4a124842"
EMG1 = "d5060205-a904-deb9-4748-2c7f4a124842"
EMG2 = "d5060305-a904-deb9-4748-2c7f4a124842"
EMG3 = "d5060405-a904-deb9-4748-2c7f4a124842"

## Batch size for realtime fine-tuning
realtime_batch_size = 2

## Epoch for realtime fine-tuning
realtime_epochs = 15

## Samples window
window = 24

## Step Size
step_size = 10

## Samples to be recored for each gesture
SAMPLES_PER_GESTURE = 20 * window #10 * window

## List of Gestures to be used for classification
GESTURES = [
    "Rest", "Fist", "Thumbs Up", "Ok Sign", "Open Hand",
    "Wave In", "Wave Out"
]

delay = 2.0

## Number of sensors Myo Armband contains
num_sensors = 8 #192

## Path to save finetuned model, set NONE if no export
## finetuned_path = None
finetuned_path = "finetuned/checkpoint.ckpt"

## 2D list to store realtime training data
sensors = [[] for _ in range(num_sensors)]

## Bluetooth device for Myo Armband
selected_device = []

## Load MEAN and Standard Deviation for Standarization from Ninapro DB5 sEMG signals.
with open("scaling_params.json", 'r') as f:
    params = json.load(f)

class Connection:
    client: BleakClient = None

    def __init__(
        self,
        EMG0: str,
        EMG1: str,
        EMG2: str,
        EMG3: str,
        CONTROL: str,
    ):
        self.EMG0 = EMG0
        self.EMG1 = EMG1
        self.EMG2 = EMG2
        self.EMG3 = EMG3
        self.CONTROL = CONTROL
        self.connected = False
        self.connected_device = None
        self.ort_session = ort.InferenceSession("weight/ICELab/Mobilenet/Training_noise_testnoise/LC_LC/98.6816.onnx")
        self.current_sample = [[] for _ in range(num_sensors)]
        self.count = 0

    def on_disconnect(self, client: BleakClient):
        self.connected = False
        print(f"Disconnected from {self.connected_device.name}!")

    async def cleanup(self):
        if self.client:
            await asyncio.gather(
            await self.client.stop_notify(EMG0),
            await self.client.stop_notify(EMG1),
            await self.client.stop_notify(EMG2),
            await self.client.stop_notify(EMG3),
            )
            await self.client.disconnect()

    async def manager(self):
        print("Starting connection manager.")
        while True:
            if self.client:
                await self.connect()
            else:
                await self.select_device()
                await asyncio.sleep(1.0)

    async def connect(self):
        if self.connected:
            return
        try:
            await asyncio.sleep(0.01)
            await self.client.connect()
            self.connected = await self.client.is_connected()
            if self.connected:
                print(f"Connected to {self.connected_device.name}")
                self.client.set_disconnected_callback(self.on_disconnect)
                bytes_to_send = bytearray([1, 3, 2, 0, 0])
                await connection.client.write_gatt_char(CONTROL, bytes_to_send)
                for gesture in GESTURES:
                    print("Begining to Perform " + gesture)
                    initial_length = len(sensors[0])
                    await asyncio.sleep(delay)
                    print("Perform " + gesture + " Now!\n")
                    await asyncio.gather(
                    await self.client.start_notify(self.EMG0, self.training_handler),
                    await self.client.start_notify(self.EMG1, self.training_handler),
                    await self.client.start_notify(self.EMG2, self.training_handler),
                    await self.client.start_notify(self.EMG3, self.training_handler),
                    )
                    while (len(sensors[0]) - initial_length) < SAMPLES_PER_GESTURE:
                        await asyncio.sleep(0.05)
                    
                    await asyncio.gather(
                    await self.client.stop_notify(EMG0),
                    await self.client.stop_notify(EMG1),
                    await self.client.stop_notify(EMG2),
                    await self.client.stop_notify(EMG3),
                    )
                    
                    for channel_idx, sensor_samples in enumerate(sensors):
                        sensors[channel_idx] = sensor_samples[:(SAMPLES_PER_GESTURE + initial_length)]
                while True:
                    if not self.connected:
                        break
                    
                    await asyncio.gather(
                    await self.client.start_notify(self.EMG0, self.prediction_handler),
                    await self.client.start_notify(self.EMG1, self.prediction_handler),
                    await self.client.start_notify(self.EMG2, self.prediction_handler),
                    await self.client.start_notify(self.EMG3, self.prediction_handler),
                    )
            else:
                print(f"Failed to connect to {self.connected_device.name}")
        except Exception as e:
            print(f"Connection failed or droped: {e}")
            self.client = None
            self.connected = False
            print(e)

    async def select_device(self):
        print("Bluetooh LE hardware warming up...")
        await asyncio.sleep(2.0)
        devices = await BleakScanner.discover()
        response = None
        for i, device in enumerate(devices):
            if device.name == "Myo Armband":  #Cyclops
                response = i
        if response is None:
            print("Could not find myo armband. Please Try Again.")
            self.cleanup()
        print(f"Connecting to {devices[response].name}")
        self.connected_device = devices[response]
        self.client = BleakClient(devices[response].address)

    def training_handler(self, sender: str, data: Any):
        sequence_1, sequence_2 = getFeatures(data, twos_complement=True)
        for channel_idx in range(8):
            sensors[channel_idx].append(sequence_1[channel_idx])
            sensors[channel_idx].append(sequence_2[channel_idx])

    async def prediction_handler(self, sender: str, data: Any):
        sequence_1, sequence_2 = getFeatures(data, twos_complement=True)
        for channel_idx in range(8):
            self.current_sample[channel_idx].append(sequence_1[channel_idx])
            self.current_sample[channel_idx].append(sequence_2[channel_idx])
        if len(self.current_sample[0]) >= window:
            sEMG = np.array([samples[-window:] for samples in self.current_sample], dtype=np.float32)
            for channel_idx in range(len(sEMG)):
                mean = params[str(channel_idx)][0]
                std = params[str(channel_idx)][1]
                sEMG[channel_idx] = (sEMG[channel_idx] - mean) / std
            sEMG = np.tile(sEMG, (192 // 8, 1))
            input_data = sEMG[np.newaxis, np.newaxis, :, :]
            #sanity check
            print(f"Explicit input shape check: {input_data.shape}")
            
            pred_out = self.ort_session.run(None, {"input": input_data})
            pred = np.argmax(pred_out[0], axis=1).item()
            if pred >= len(GESTURES):
                print("⚠️ Invalid prediction:", pred, "— output:", pred_out)
            else:
                gesture_buffer.append(pred)
                most_common = max(set(gesture_buffer), key=gesture_buffer.count)
                print(GESTURES[most_common])
            #     print(GESTURES[pred])
            # self.current_sample = [samples[-(window - step_size):] for samples in self.current_sample]

def getFeatures(data, twos_complement=True):
    sequence_1 = []
    sequence_2 = []
    for i in range(8):
        if twos_complement and data[i] > 127:
            sequence_1.append(data[i] - 256)
        else:
            sequence_1.append(data[i])
    for i in range(8, 16):
        if twos_complement and data[i] > 127:
            sequence_2.append(data[i] - 256)
        else:
            sequence_2.append(data[i])
    return sequence_1, sequence_2

##########################
## App Main
##########################
if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    connection = Connection(EMG0, EMG1, EMG2, EMG3, CONTROL)
    try:
        asyncio.ensure_future(connection.manager())
        loop.run_forever()
    except KeyboardInterrupt:
        print("User stopped program.")
    finally:
        print("Disconnecting...")
        loop.run_until_complete(connection.cleanup())
