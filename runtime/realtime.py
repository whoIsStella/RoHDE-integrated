from __future__ import annotations

import asyncio
import json
from collections import deque

import numpy as np
import onnxruntime as ort
from bleak import BleakClient, BleakScanner

from .config import (
    MODEL_INPUT_SHAPE,
    ONNX_PATH,
    SCALING_PARAMS_PATH,
    STEP,
    VOTING_WINDOW,
    WINDOW,
)
from .preprocessing import decode_emg_packet, majority_vote, prepare_model_input

CONTROL = "d5060401-a904-deb9-4748-2c7f4a124842"
EMG_CHARACTERISTICS = (
    "d5060105-a904-deb9-4748-2c7f4a124842",
    "d5060205-a904-deb9-4748-2c7f4a124842",
    "d5060305-a904-deb9-4748-2c7f4a124842",
    "d5060405-a904-deb9-4748-2c7f4a124842",
)


class LiveClassifier:
    def __init__(self) -> None:
        if not ONNX_PATH.exists():
            raise FileNotFoundError(
                f"{ONNX_PATH} does not exist. Run: python -m runtime.export_onnx"
            )

        with SCALING_PARAMS_PATH.open("r", encoding="utf-8") as handle:
            self.scaling_params = json.load(handle)

        self.session = ort.InferenceSession(
            str(ONNX_PATH),
            providers=["CPUExecutionProvider"],
        )
        self.input = self.session.get_inputs()[0]
        self.output = self.session.get_outputs()[0]

        self._validate_model_contract()

        self.channels = [deque(maxlen=WINDOW) for _ in range(MODEL_INPUT_SHAPE[0])]
        self.predictions: deque[int] = deque(maxlen=VOTING_WINDOW)
        self.samples_since_prediction = 0

    def _validate_model_contract(self) -> None:
        shape = self.input.shape
        expected = [None, 1, *MODEL_INPUT_SHAPE]
        if len(shape) != 4:
            raise RuntimeError(f"expected rank-4 ONNX input, got {shape}")

        for actual, wanted in zip(shape, expected, strict=True):
            if wanted is None:
                continue
            if isinstance(actual, int) and actual != wanted:
                raise RuntimeError(
                    f"ONNX input shape {shape} does not match expected {expected}"
                )

    def handle_packet(self, _sender, data: bytearray) -> None:
        samples = decode_emg_packet(data)

        for sample in samples:
            for channel, value in enumerate(sample):
                self.channels[channel].append(int(value))
            self.samples_since_prediction += 1

        if any(len(channel) < WINDOW for channel in self.channels):
            return
        if self.samples_since_prediction < STEP:
            return

        self.samples_since_prediction = 0
        window = np.asarray(self.channels, dtype=np.float32)
        input_data = prepare_model_input(window, self.scaling_params)

        logits = self.session.run([self.output.name], {self.input.name: input_data})[0]
        prediction = int(np.argmax(logits, axis=1).item())

        self.predictions.append(prediction)
        stable = majority_vote(list(self.predictions))
        print(f"class_{stable}")


async def find_myo():
    devices = await BleakScanner.discover(timeout=5.0)
    for device in devices:
        if device.name == "Myo Armband":
            return device
    raise RuntimeError("Myo Armband not found")


async def run() -> None:
    classifier = LiveClassifier()
    device = await find_myo()

    print(f"Connecting to {device.name} ({device.address})")

    async with BleakClient(device) as client:
        if not client.is_connected:
            raise RuntimeError("failed to connect to Myo Armband")

        await client.write_gatt_char(CONTROL, bytearray([1, 3, 2, 0, 0]))

        for characteristic in EMG_CHARACTERISTICS:
            await client.start_notify(characteristic, classifier.handle_packet)

        print("Streaming EMG. Press Ctrl+C to stop.")
        try:
            while client.is_connected:
                await asyncio.sleep(1.0)
        finally:
            for characteristic in EMG_CHARACTERISTICS:
                try:
                    await client.stop_notify(characteristic)
                except Exception:
                    pass


def main() -> None:
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
