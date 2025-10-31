# ECOLANG Mesh Rendering API - Google Colab Backend
# ====================================================
#
# This file contains all the code cells for the Colab notebook.
# Copy each section into separate cells in Google Colab.
#
# Setup Instructions:
# 1. Create new notebook in Google Colab
# 2. Change runtime to GPU: Runtime → Change runtime type → GPU
# 3. Copy each CELL below into separate code cells
# 4. Run cells in order (1 through 6)
# 5. Keep Cell 6 running while using the application
#

# ==============================================================================
# CELL 1: Install Dependencies
# ==============================================================================

# Install all required packages
!pip install -q fastapi uvicorn pyngrok pillow trimesh pyrender smplx torch opencv-python

print("✅ All dependencies installed successfully")


# ==============================================================================
# CELL 2: Mount Google Drive & Verify Structure
# ==============================================================================

from google.colab import drive
import os

# Mount Google Drive
drive.mount('/content/drive')

# Define paths
BASE_PATH = "/content/drive/MyDrive/ecolang"
NPZ_BASE_PATH = os.path.join(BASE_PATH, "Extracted_parameters")
MODEL_PATH = os.path.join(BASE_PATH, "models", "SMPLX_NEUTRAL.npz")
VIDEOS_PATH = os.path.join(BASE_PATH, "videos")

# Verify structure
print("Verifying Google Drive structure...\n")

# Check base folder
assert os.path.exists(BASE_PATH), f"❌ Base folder not found: {BASE_PATH}"
print(f"✅ Base folder exists: {BASE_PATH}")

# Check NPZ parameters
assert os.path.exists(NPZ_BASE_PATH), f"❌ NPZ folder not found: {NPZ_BASE_PATH}"
video_folders = os.listdir(NPZ_BASE_PATH)
print(f"✅ NPZ folder exists with {len(video_folders)} video folders:")
for folder in video_folders:
    folder_path = os.path.join(NPZ_BASE_PATH, folder)
    if os.path.isdir(folder_path):
        file_count = len([f for f in os.listdir(folder_path) if f.endswith('.npz')])
        print(f"   - {folder}: {file_count} NPZ files")

# Check SMPL-X model
assert os.path.exists(MODEL_PATH), f"❌ SMPL-X model not found: {MODEL_PATH}"
model_size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
print(f"✅ SMPL-X model exists: {model_size_mb:.1f} MB")

# Check videos
assert os.path.exists(VIDEOS_PATH), f"❌ Videos folder not found: {VIDEOS_PATH}"
video_files = [f for f in os.listdir(VIDEOS_PATH) if f.endswith('.mp4')]
print(f"✅ Videos folder exists with {len(video_files)} MP4 files:")
for video in video_files:
    print(f"   - {video}")

print("\n" + "="*60)
print("✅ ALL CHECKS PASSED - Ready to start API server!")
print("="*60)


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
smplx_model = smplx.create(
    model_path=os.path.dirname(MODEL_PATH),
    model_type='smplx',
    gender='neutral',
    use_face_contour=False,
    use_pca=True,  # Use PCA for hand poses (12D instead of 45D)
    num_pca_comps=12,
    flat_hand_mean=True
).to(device)

print("✅ SMPL-X model loaded successfully!")
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


# Test the function
print("Testing render function...")
test_npz = os.path.join(NPZ_BASE_PATH, video_folders[0], "frame_0001_params.npz")
if os.path.exists(test_npz):
    try:
        test_img = render_mesh_from_npz(test_npz)
        print(f"✅ Render function works! Image size: {len(test_img)} characters")
    except Exception as e:
        print(f"❌ Render test failed: {str(e)}")
else:
    print(f"⚠️  Test file not found: {test_npz}")


# ==============================================================================
# CELL 5: Create FastAPI Server
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
        "endpoints": ["/health", "/render_frame"]
    }

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "message": "Colab API is running",
        "device": str(device),
        "model_loaded": smplx_model is not None
    }

@app.post("/render_frame", response_model=RenderResponse)
async def render_frame(request: RenderRequest):
    """
    Render a single frame from NPZ parameters

    Request body:
    - video_id: Name of video folder (e.g., "video1_speaking")
    - frame_number: Frame number (1-1800)
    - person_id: Person ID in frame (default: 0)

    Returns:
    - success: Boolean indicating if render succeeded
    - image: Base64-encoded PNG image (if success)
    - frame_number: Echo of requested frame number
    - error: Error message (if failed)
    """

    try:
        # Validate frame number
        if request.frame_number < 1 or request.frame_number > 1800:
            raise HTTPException(
                status_code=400,
                detail=f"Frame number must be between 1 and 1800, got {request.frame_number}"
            )

        # Build NPZ path
        npz_filename = f"frame_{request.frame_number:04d}_params.npz"
        npz_path = os.path.join(NPZ_BASE_PATH, request.video_id, npz_filename)

        # Check if file exists
        if not os.path.exists(npz_path):
            raise HTTPException(
                status_code=404,
                detail=f"NPZ file not found: {npz_filename} for video {request.video_id}"
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

print("✅ FastAPI server configured with endpoints:")
print("   - GET  /         (API info)")
print("   - GET  /health   (Health check)")
print("   - POST /render_frame (Render mesh)")


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
print("="*70 + "\n")

# Start FastAPI server (this will run indefinitely)
uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")


# ==============================================================================
# USAGE INSTRUCTIONS
# ==============================================================================
#
# After running Cell 6:
# 1. Copy the ngrok URL that appears (e.g., https://abc123.ngrok-free.app)
# 2. Go to Streamlit Cloud: https://share.streamlit.io/
# 3. Open your app settings
# 4. Add to Secrets:
#    COLAB_API_URL = "https://your-ngrok-url-here.ngrok-free.app"
# 5. Save and restart your Streamlit app
# 6. Test by selecting a video in the dropdown
#
# Keep this notebook running while using the application!
#
# ==============================================================================
