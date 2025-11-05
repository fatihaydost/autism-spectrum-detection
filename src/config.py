from pathlib import Path

# Root directory of the project (this file lives in src/).
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Primary dataset location after cleanup.
DATA_ROOT = PROJECT_ROOT / "data" / "processed" / "asd_faces"

# Where trained models, logs, and checkpoints are stored.
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)

# Label canonicalisation table to cope with folder naming inconsistencies.
LABEL_CANONICAL_MAP = {
    "autism": "autistic",
    "autistic": "autistic",
    "tipical": "non_autistic",
    "typical": "non_autistic",
}

CLASS_NAMES = ("non_autistic", "autistic")
NUM_CLASSES = len(CLASS_NAMES)

# Training defaults.
IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 4
SEED = 42
CLASSIFIER_DROPOUT = 0.3

# Default checkpoint path for the fine-tuned classifier.
DEFAULT_CHECKPOINT_PATH = ARTIFACTS_DIR / "resnet18_autism_classifier.pth"
