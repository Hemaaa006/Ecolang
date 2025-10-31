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
    BASE_PATH = "/content/drive/MyDrive/ecolang"
elif IS_PRODUCTION:
    # Production server
    BASE_PATH = "/app/data"
else:
    # Local development
    BASE_PATH = "./data"

# Data directories
VIDEOS_DIR = os.path.join(BASE_PATH, "videos")
NPZ_DIR = os.path.join(BASE_PATH, "Extracted_parameters")
MODEL_PATH = os.path.join(BASE_PATH, "models")
CACHE_DIR = os.path.join(BASE_PATH, "cache")

# Rendering settings
DEFAULT_IMG_SIZE = 720
DEFAULT_BG_COLOR = (245, 245, 245)  # Light gray
DEFAULT_DEVICE = 'cuda' if IS_PRODUCTION else 'cpu'

# Video library
VIDEO_LIBRARY = {
    'ch07_speakerview_012': {
        'title': 'CH07 Speaker View 012',
        'filename': 'ch07_speakerview_012.mp4',
        'github_url': 'https://drive.google.com/uc?export=download&id=1LLeKFl_MqUdp1EVByHfs0dTJO8Qhyt2D',
        'duration': '1:00',
        'fps': 30,
        'frames': 1800,
    },
    'ch08_speakerview_025': {
        'title': 'CH08 Speaker View 025',
        'filename': 'ch08_speakerview_025.mp4',
        'github_url': 'https://drive.google.com/uc?export=download&id=1G_Mg7n3HhJrz8tBBSMHiDGPft7BZSkL1',
        'duration': '1:00',
        'fps': 30,
        'frames': 1800
    },
    'ch09_speakerview_027': {
        'title': 'CH09 Speaker View 027',
        'filename': 'ch09_speakerview_027.mp4',
        'github_url': 'https://drive.google.com/uc?export=download&id=1HgrEvuiI0ivgOrJIsmzknUhzIUlz2e_a',
        'duration': '1:00',
        'fps': 30,
        'frames': 1800
    },
    'ch11_speakerview_002': {
        'title': 'CH11 Speaker View 002',
        'filename': 'ch11_speakerview_002.mp4',
        'github_url': 'https://drive.google.com/uc?export=download&id=1kU35NXEdfl_VvJEX2oHRiawdeTWzPlN3',
        'duration': '1:00',
        'fps': 30,
        'frames': 1800
    }
}

# Performance settings
CACHE_SIZE = 30  # Number of frames to cache
BATCH_SIZE = 10  # For batch processing
