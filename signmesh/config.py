"""
Configuration for SignMesh application
Automatically detects environment (Colab, local, production)
"""
import os
from pathlib import Path

# Detect environment
IS_COLAB = os.path.exists('/content')
IS_PRODUCTION = os.environ.get('ENV') == 'production'

# Base paths
if IS_COLAB:
    # Google Colab with mounted Drive
    BASE_PATH = "/content/drive/MyDrive/SignMesh"
elif IS_PRODUCTION:
    # Production server
    BASE_PATH = "/app/data"
else:
    # Local development
    BASE_PATH = "./data"

# Data directories
VIDEOS_DIR = os.path.join(BASE_PATH, "videos")
NPZ_DIR = os.path.join(BASE_PATH, "npz_files")
MODEL_PATH = os.path.join(BASE_PATH, "models")
CACHE_DIR = os.path.join(BASE_PATH, "cache")

# Rendering settings
DEFAULT_IMG_SIZE = 720
DEFAULT_BG_COLOR = (245, 245, 245)  # Light gray
DEFAULT_DEVICE = 'cuda' if IS_PRODUCTION else 'cpu'

# Video library
VIDEO_LIBRARY = {
    'video1_speaking': {
        'title': 'Video 1 - Speaking',
        'filename': 'video1_speaking.mp4',
        'duration': '1:00',
        'fps': 30,
        'frames': 1800
    },
    'video2_gestures': {
        'title': 'Video 2 - Gestures',
        'filename': 'video2_gestures.mp4',
        'duration': '1:00',
        'fps': 30,
        'frames': 1800
    },
    'video3_conversation': {
        'title': 'Video 3 - Conversation',
        'filename': 'video3_conversation.mp4',
        'duration': '1:00',
        'fps': 30,
        'frames': 1800
    },
    'video4_demonstration': {
        'title': 'Video 4 - Demonstration',
        'filename': 'video4_demonstration.mp4',
        'duration': '1:00',
        'fps': 30,
        'frames': 1800
    }
}

# Performance settings
CACHE_SIZE = 30  # Number of frames to cache
BATCH_SIZE = 10  # For batch processing
