"""
config.py

Central configuration for Smart Follower Shopping Cart.
"""

from pathlib import Path

# ==========================================================
# Project Paths
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent

MODEL_PATH = PROJECT_ROOT / "models" / "yolov8n.onnx"

LOG_DIR = PROJECT_ROOT / "logs"

TEST_IMAGE_DIR = PROJECT_ROOT / "test_images"

# ==========================================================
# Camera
# ==========================================================

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

CAMERA_ROTATE_180 = True

# ==========================================================
# YOLO
# ==========================================================

YOLO_INPUT_SIZE = 320

YOLO_SKIP_FRAMES = 2

YOLO_INPUT_WIDTH = 320
YOLO_INPUT_HEIGHT = 320

# Confidence required for a detection
CONFIDENCE_THRESHOLD = 0.45

# Non-Maximum Suppression threshold
NMS_THRESHOLD = 0.45

# Only detect persons
PERSON_CLASS_ID = 0

# ==========================================================
# Tracking
# ==========================================================

LOCK_FRAMES_REQUIRED = 5

TARGET_TIMEOUT = 20

IOU_THRESHOLD = 0.30

HSV_DISTANCE_THRESHOLD = 35

# ==========================================================
# Motion
# ==========================================================

MOVE_COMMAND = "MOVE"

STOP_COMMAND = "STOP"

# ==========================================================
# Arduino
# ==========================================================

SERIAL_PORT = "/dev/ttyACM0"

BAUDRATE = 115200

RECONNECT_INTERVAL = 5

# ==========================================================
# Display
# ==========================================================

SHOW_WINDOW = False

SAVE_DEBUG_IMAGES = False

# ==========================================================
# Logging
# ==========================================================

LOG_LEVEL = "INFO"
