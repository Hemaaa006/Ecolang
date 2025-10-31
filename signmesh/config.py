"""
Configuration for ECOLANG
API-based architecture with Google Colab backend
"""
import os

# =============================================================================
# Colab API Configuration
# =============================================================================
# The ngrok URL from your running Colab notebook
# This should be set in Streamlit Cloud secrets as COLAB_API_URL
COLAB_API_URL = os.environ.get('COLAB_API_URL', 'http://localhost:8000')

# =============================================================================
# Application Configuration
# =============================================================================
APP_NAME = "ECOLANG"
APP_VERSION = "1.0.0"

# Default video settings
DEFAULT_VIDEO_FPS = 30
DEFAULT_FRAMES_PER_VIDEO = 1800

# Mesh generation settings
MESH_IMG_SIZE = 720
MESH_GENERATION_TIMEOUT = 600  # 10 minutes max for full video

# =============================================================================
# Video Library
# =============================================================================
# Videos are hosted on Google Drive
#
# HOW TO UPDATE THESE URLS:
# 1. Upload videos to: /MyDrive/ecolang/videos/ folder in Google Drive
# 2. For each video: Right-click → Share → "Anyone with the link" can view
# 3. Copy the share link (looks like: https://drive.google.com/file/d/FILE_ID/view?usp=sharing)
# 4. Extract the FILE_ID (the part between /d/ and /view)
# 5. Replace PASTE_FILE_ID_HERE with your actual FILE_ID below
#
# Example conversion:
#   Share link: https://drive.google.com/file/d/1AbCdEfGhIjKlMnOp/view?usp=sharing
#   FILE_ID: 1AbCdEfGhIjKlMnOp
#   Direct URL: https://drive.google.com/uc?export=download&id=1AbCdEfGhIjKlMnOp

VIDEO_LIBRARY = {
    'video1_speaking': {
        'title': 'Video 1 - Speaking',
        'filename': 'video1_speaking.mp4',
        'github_url': 'https://drive.google.com/uc?export=download&id=PASTE_FILE_ID_HERE',
        'frames': 1800,
        'fps': 30,
        'duration': '1:00'
    },
    'video2_gestures': {
        'title': 'Video 2 - Gestures',
        'filename': 'video2_gestures.mp4',
        'github_url': 'https://drive.google.com/uc?export=download&id=PASTE_FILE_ID_HERE',
        'frames': 1800,
        'fps': 30,
        'duration': '1:00'
    },
    'video3_conversation': {
        'title': 'Video 3 - Conversation',
        'filename': 'video3_conversation.mp4',
        'github_url': 'https://drive.google.com/uc?export=download&id=PASTE_FILE_ID_HERE',
        'frames': 1800,
        'fps': 30,
        'duration': '1:00'
    },
    'video4_demonstration': {
        'title': 'Video 4 - Demonstration',
        'filename': 'video4_demonstration.mp4',
        'github_url': 'https://drive.google.com/uc?export=download&id=PASTE_FILE_ID_HERE',
        'frames': 1800,
        'fps': 30,
        'duration': '1:00'
    }
}

# =============================================================================
# Google Drive Structure (For Colab Backend)
# =============================================================================
# This is the expected folder structure in your Google Drive
# /MyDrive/ecolang/
#   ├── Extracted_parameters/
#   │   ├── video1_speaking/
#   │   │   ├── frame_0001_params.npz
#   │   │   ├── frame_0002_params.npz
#   │   │   └── ... (1800 files)
#   │   ├── video2_gestures/
#   │   └── ...
#   └── models/
#       └── SMPLX_NEUTRAL.npz

# Paths used by Colab backend (for reference)
DRIVE_BASE_PATH = "/content/drive/MyDrive/ecolang"
DRIVE_NPZ_PATH = f"{DRIVE_BASE_PATH}/Extracted_parameters"
DRIVE_MODEL_PATH = f"{DRIVE_BASE_PATH}/models"

# =============================================================================
# UI Configuration
# =============================================================================
# Color scheme
PRIMARY_COLOR = "#2E86AB"
SECONDARY_COLOR = "#A23B72"
BACKGROUND_COLOR = "#F8F9FA"
TEXT_COLOR = "#333333"

# Layout
SIDEBAR_ENABLED = False  # No sidebar in ECOLANG
SHOW_EMOJIS = False  # Clean design without emojis
