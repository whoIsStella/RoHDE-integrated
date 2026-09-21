from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = REPO_ROOT / "models"
CHECKPOINT_PATH = MODEL_DIR / "rohde-lc.pt"
ONNX_PATH = MODEL_DIR / "rohde-lc.onnx"
SCALING_PARAMS_PATH = Path(__file__).with_name("scaling_params.json")

MYO_CHANNELS = 8
WINDOW = 24
STEP = 10
VOTING_WINDOW = 5

MODEL_INPUT_SHAPE = (MYO_CHANNELS, WINDOW)
