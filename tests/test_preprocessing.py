import unittest

import numpy as np

from runtime.preprocessing import (
    decode_emg_packet,
    majority_vote,
    normalize_window,
    prepare_model_input,
)


SCALING = {str(i): [float(i), 2.0] for i in range(8)}


class PreprocessingTests(unittest.TestCase):
    def test_decode_packet_converts_unsigned_bytes_to_signed_samples(self):
        packet = bytes([0, 1, 127, 128, 129, 254, 255, 64] * 2)
        decoded = decode_emg_packet(packet)

        self.assertEqual(decoded.shape, (2, 8))
        self.assertEqual(decoded[0].tolist(), [0, 1, 127, -128, -127, -2, -1, 64])

    def test_decode_packet_rejects_wrong_length(self):
        with self.assertRaises(ValueError):
            decode_emg_packet(b"short")

    def test_prepare_model_input_keeps_native_eight_channels(self):
        window = np.zeros((8, 24), dtype=np.float32)
        prepared = prepare_model_input(window, SCALING)

        self.assertEqual(prepared.shape, (1, 1, 8, 24))
        self.assertEqual(prepared.dtype, np.float32)

    def test_normalization_is_per_channel(self):
        window = np.vstack([np.full(24, i + 2, dtype=np.float32) for i in range(8)])
        normalized = normalize_window(window, SCALING)

        np.testing.assert_allclose(normalized, np.ones((8, 24), dtype=np.float32))

    def test_majority_vote(self):
        self.assertEqual(majority_vote([3, 2, 3, 3, 2]), 3)


if __name__ == "__main__":
    unittest.main()
