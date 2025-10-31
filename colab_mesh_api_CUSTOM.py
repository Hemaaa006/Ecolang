# ECOLANG Mesh Rendering API - CUSTOM VERSION FOR YOUR DRIVE STRUCTURE
# ========================================================================
#
# This version is customized for your specific Google Drive folder names:
# - ch07_speakerview_012_parameters
# - ch08_speakerview_025_parameters
# - ch09_speakerview_027_parameters
# - ch11_speakerview_002_parameters
#
# Copy each CELL below into separate cells in Google Colab.
#

# ==============================================================================
# CELL 1: Install Dependencies
# ==============================================================================

# Install all required packages
!pip install -q fastapi uvicorn pyngrok pillow trimesh pyrender smplx torch opencv-python

print("✓ All dependencies installed successfully")


# ==============================================================================
# CELL 2: Mount Google Drive & Verify Structure
# ==============================================================================

from google.colab import drive
import os

# Mount Google Drive
drive.mount('/content/drive')

# Define paths - CUSTOMIZED FOR YOUR STRUCTURE
BASE_PATH = "/content/drive/MyDrive/ecolang"
NPZ_BASE_PATH = os.path.join(BASE_PATH, "Extracted_parameters")

# Your actual folder names from the screenshots
ACTUAL_FOLDERS = [
    "ch07_speakerview_012_parameters",
    "ch08_speakerview_025_parameters",
    "ch09_speakerview_027_parameters",
    "ch11_speakerview_002_parameters"
]

# Mapping: video IDs to actual NPZ parameter folder names
VIDEO_FOLDER_MAPPING = {
    "ch07_speakerview_012": "ch07_speakerview_012_parameters",
    "ch08_speakerview_025": "ch08_speakerview_025_parameters",
    "ch09_speakerview_027": "ch09_speakerview_027_parameters",
    "ch11_speakerview_002": "ch11_speakerview_002_parameters"
}

print("Verifying Google Drive structure...\n")

# Check base folder
if not os.path.exists(BASE_PATH):
    print(f"❌ Base folder not found: {BASE_PATH}")
    print("Please create folder: /MyDrive/ecolang/")
else:
    print(f"✓ Base folder exists: {BASE_PATH}")

# Check NPZ parameters folder
if not os.path.exists(NPZ_BASE_PATH):
    print(f"❌ NPZ folder not found: {NPZ_BASE_PATH}")
    print("Please create folder: /MyDrive/ecolang/Extracted_parameters/")
else:
    print(f"✓ NPZ folder exists: {NPZ_BASE_PATH}\n")

    # Check each video folder
    print("Checking video folders:")
    for app_name, actual_folder in VIDEO_FOLDER_MAPPING.items():
        folder_path = os.path.join(NPZ_BASE_PATH, actual_folder)
        if os.path.exists(folder_path):
            file_count = len([f for f in os.listdir(folder_path) if f.endswith('.npz')])
            print(f"  ✓ {actual_folder}: {file_count} NPZ files")
        else:
            print(f"  ❌ {actual_folder}: NOT FOUND")

# Check if SMPL-X model exists (we'll download it if not)
MODEL_DIR = os.path.join(BASE_PATH, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "SMPLX_NEUTRAL.npz")

if os.path.exists(MODEL_PATH):
    model_size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
    print(f"\n✓ SMPL-X model exists: {model_size_mb:.1f} MB")
else:
    print(f"\n⚠️  SMPL-X model not found at: {MODEL_PATH}")
    print("You'll need to upload SMPLX_NEUTRAL.npz to /MyDrive/ecolang/models/")

print("\n" + "="*70)
print("✓ Verification complete - Ready to load model!")
print("="*70)


# ==============================================================================
# CELL 3: Load SMPL-X Model
# ==============================================================================

import torch
import smplx
import numpy as np

# Select device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Load SMPL-X model
print("\nLoading SMPL-X model...")

# Check if model exists
if not os.path.exists(MODEL_PATH):
    print(f"\n❌ ERROR: SMPL-X model not found!")
    print(f"Please upload SMPLX_NEUTRAL.npz to: {MODEL_DIR}")
    print("\nYou can download it from: https://smpl-x.is.tue.mpg.de/")
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

smplx_model = smplx.create(
    model_path=MODEL_DIR,
    model_type='smplx',
    gender='neutral',
    use_face_contour=False,
    use_pca=True,
    num_pca_comps=12,
    flat_hand_mean=True
).to(device)

print("✓ SMPL-X model loaded successfully!")
print(f"   - Body joints: {smplx_model.NUM_BODY_JOINTS}")
print(f"   - Total vertices: {smplx_model.get_num_verts()}")


# ==============================================================================
# CELL 4: Create Mesh Rendering Function
# ==============================================================================

import pyrender
import trimesh
from PIL import Image
import io
import base64

def render_mesh_from_npz(npz_path, person_id=0):
    """
    Load NPZ parameters and render 3D mesh to 720x720 image

    Args:
        npz_path: Path to NPZ file with SMPL-X parameters
        person_id: Person ID in the NPZ file (default: 0)

    Returns:
        Base64-encoded PNG image string
    """

    # Load parameters from NPZ
    data = np.load(npz_path, allow_pickle=True)
    prefix = f'person_{person_id}_smplx_'

    # Check if person exists
    person_ids = data.get('person_ids', np.array([]))
    if len(person_ids) == 0 or person_id not in person_ids:
        raise ValueError(f"Person {person_id} not found in frame")

    # Extract SMPL-X parameters
    try:
        global_orient = torch.tensor(
            data[prefix + 'root_pose'].reshape(1, 3),
            dtype=torch.float32
        ).to(device)

        body_pose = torch.tensor(
            data[prefix + 'body_pose'].reshape(1, 21, 3),
            dtype=torch.float32
        ).to(device)

        betas = torch.tensor(
            data[prefix + 'shape'].reshape(1, 10),
            dtype=torch.float32
        ).to(device)

        expression = torch.tensor(
            data.get(prefix + 'expr', np.zeros((1, 10))).reshape(1, 10),
            dtype=torch.float32
        ).to(device)

        jaw_pose = torch.tensor(
            data.get(prefix + 'jaw_pose', np.zeros((1, 3))).reshape(1, 3),
            dtype=torch.float32
        ).to(device)

        left_hand_pose = torch.tensor(
            data[prefix + 'lhand_pose'].reshape(1, 12),
            dtype=torch.float32
        ).to(device)

        right_hand_pose = torch.tensor(
            data[prefix + 'rhand_pose'].reshape(1, 12),
            dtype=torch.float32
        ).to(device)

    except Exception as e:
        raise ValueError(f"Error extracting parameters: {str(e)}")

    # Generate mesh using SMPL-X
    with torch.no_grad():
        output = smplx_model(
            global_orient=global_orient,
            body_pose=body_pose,
            betas=betas,
            expression=expression,
            jaw_pose=jaw_pose,
            left_hand_pose=left_hand_pose,
            right_hand_pose=right_hand_pose
        )

        vertices = output.vertices.detach().cpu().numpy()[0]
        faces = smplx_model.faces

    # Create trimesh object
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    # Setup pyrender scene
    scene = pyrender.Scene(ambient_light=[0.3, 0.3, 0.3])

    # Add mesh to scene
    mesh_material = pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.2,
        roughnessFactor=0.8,
        baseColorFactor=[0.3, 0.3, 0.8, 1.0]  # Blue-ish color
    )
    mesh_node = pyrender.Mesh.from_trimesh(mesh, material=mesh_material)
    scene.add(mesh_node)

    # Setup camera
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
    camera_pose = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 2.5],
        [0.0, 0.0, 0.0, 1.0]
    ])
    scene.add(camera, pose=camera_pose)

    # Setup lighting
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
    scene.add(light, pose=camera_pose)

    # Render to image
    renderer = pyrender.OffscreenRenderer(720, 720)
    color, _ = renderer.render(scene)
    renderer.delete()

    # Convert to PIL Image
    img = Image.fromarray(color)

    # Encode as base64
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

    return img_base64


# Test the function with your first video
print("Testing render function...")
test_folder = ACTUAL_FOLDERS[0]  # ch07_speakerview_012_parameters
test_npz = os.path.join(NPZ_BASE_PATH, test_folder, "frame_0001_params.npz")

if os.path.exists(test_npz):
    try:
        test_img = render_mesh_from_npz(test_npz)
        print(f"✓ Render function works! Image size: {len(test_img)} characters")
        print(f"  Tested with: {test_folder}/frame_0001_params.npz")
    except Exception as e:
        print(f"❌ Render test failed: {str(e)}")
else:
    print(f"⚠️  Test file not found: {test_npz}")
    print("Make sure NPZ files are in the correct folders")


# ==============================================================================
# CELL 5: Create FastAPI Server with Folder Mapping
# ==============================================================================

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

app = FastAPI(
    title="ECOLANG Mesh Rendering API",
    description="Backend API for rendering SMPL-X meshes from sign language videos",
    version="1.0.0"
)

class RenderRequest(BaseModel):
    video_id: str
    frame_number: int
    person_id: Optional[int] = 0

class RenderResponse(BaseModel):
    success: bool
    image: Optional[str] = None
    frame_number: int
    error: Optional[str] = None

@app.get("/")
def root():
    return {
        "message": "ECOLANG Mesh Rendering API",
        "status": "online",
        "available_videos": list(VIDEO_FOLDER_MAPPING.keys()),
        "folder_mapping": VIDEO_FOLDER_MAPPING
    }

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "message": "Colab API is running",
        "device": str(device),
        "model_loaded": smplx_model is not None,
        "available_videos": list(VIDEO_FOLDER_MAPPING.keys())
    }

@app.post("/render_frame", response_model=RenderResponse)
async def render_frame(request: RenderRequest):
    """
    Render a single frame from NPZ parameters

    Request body:
    - video_id: Name from app (e.g., "video1_speaking")
    - frame_number: Frame number (1-1800 or however many frames you have)
    - person_id: Person ID in frame (default: 0)

    Returns:
    - success: Boolean indicating if render succeeded
    - image: Base64-encoded PNG image (if success)
    - frame_number: Echo of requested frame number
    - error: Error message (if failed)
    """

    try:
        # Validate frame number
        if request.frame_number < 1:
            raise HTTPException(
                status_code=400,
                detail=f"Frame number must be >= 1, got {request.frame_number}"
            )

        # Map the app's video_id to your actual folder name
        if request.video_id not in VIDEO_FOLDER_MAPPING:
            raise HTTPException(
                status_code=404,
                detail=f"Video '{request.video_id}' not found. Available: {list(VIDEO_FOLDER_MAPPING.keys())}"
            )

        actual_folder = VIDEO_FOLDER_MAPPING[request.video_id]

        # Build NPZ path using YOUR folder structure
        npz_filename = f"frame_{request.frame_number:04d}_params.npz"
        npz_path = os.path.join(NPZ_BASE_PATH, actual_folder, npz_filename)

        # Check if file exists
        if not os.path.exists(npz_path):
            raise HTTPException(
                status_code=404,
                detail=f"NPZ file not found: {actual_folder}/{npz_filename}"
            )

        # Render mesh
        img_base64 = render_mesh_from_npz(npz_path, person_id=request.person_id)

        return RenderResponse(
            success=True,
            image=img_base64,
            frame_number=request.frame_number,
            error=None
        )

    except HTTPException:
        raise
    except Exception as e:
        return RenderResponse(
            success=False,
            image=None,
            frame_number=request.frame_number,
            error=str(e)
        )

print("✓ FastAPI server configured with folder mapping:")
for app_name, actual_folder in VIDEO_FOLDER_MAPPING.items():
    print(f"   - {app_name} → {actual_folder}")


# ==============================================================================
# CELL 6: Start Server with ngrok
# ==============================================================================

from pyngrok import ngrok
import uvicorn
import nest_asyncio

# Allow nested event loops (required for Colab)
nest_asyncio.apply()

# Set your ngrok authtoken
# Get it from: https://dashboard.ngrok.com/get-started/your-authtoken
NGROK_AUTH_TOKEN = "YOUR_NGROK_TOKEN_HERE"  # ← REPLACE THIS WITH YOUR ACTUAL TOKEN

# Authenticate ngrok
ngrok.set_auth_token(NGROK_AUTH_TOKEN)

# Kill any existing ngrok tunnels
ngrok.kill()

# Start ngrok tunnel
public_url = ngrok.connect(8000)
print("\n" + "="*70)
print("🚀 ECOLANG MESH API IS NOW RUNNING!")
print("="*70)
print(f"\n📡 Public URL: {public_url}")
print(f"\n🔗 Test endpoints:")
print(f"   - Health: {public_url}/health")
print(f"   - Docs:   {public_url}/docs")
print("\n" + "="*70)
print("📋 COPY THIS URL FOR STREAMLIT CLOUD:")
print("="*70)
print(f'\nCOLAB_API_URL = "{public_url}"')
print("\n" + "="*70)
print("⚠️  IMPORTANT: Keep this cell running!")
print("    Stopping it will shut down the API server.")
print("\n📁 Video folder mapping:")
for app_name, actual_folder in VIDEO_FOLDER_MAPPING.items():
    print(f"   {app_name} → {actual_folder}")
print("="*70 + "\n")

# Start FastAPI server (this will run indefinitely)
uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")


# ==============================================================================
# USAGE INSTRUCTIONS
# ==============================================================================
#
# This version is customized for YOUR Google Drive structure:
#
# Your folders:
#   - ch07_speakerview_012_parameters  (mapped to: video1_speaking)
#   - ch08_speakerview_025_parameters  (mapped to: video2_gestures)
#   - ch09_speakerview_027_parameters  (mapped to: video3_conversation)
#   - ch11_speakerview_002_parameters  (mapped to: video4_demonstration)
#
# The app will call them as video1_speaking, video2_gestures, etc.
# The API automatically translates to your actual folder names.
#
# After running Cell 6:
# 1. Copy the ngrok URL (e.g., https://abc123.ngrok-free.app)
# 2. Update Streamlit Cloud secrets with this URL
# 3. Keep this notebook running!
#
# ==============================================================================
