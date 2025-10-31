"""
ECOLANG Colab Backend - Complete Setup
Run this in Google Colab to create the rendering API
"""

# ============= CELL 1: Setup Environment =============
# Install dependencies
!apt-get install -y xvfb
!pip install -q pyvirtualdisplay smplx trimesh pyrender nest_asyncio fastapi uvicorn ngrok opencv-python pillow

# Setup virtual display for headless rendering
from pyvirtualdisplay import Display
import os
display = Display(visible=0, size=(1400, 900))
display.start()
os.environ['PYOPENGL_PLATFORM'] = 'egl'

print("✓ Environment setup complete!")

# ============= CELL 2: Mount Google Drive =============
from google.colab import drive
drive.mount('/content/drive')

import os
BASE_PATH = "/content/drive/MyDrive/ecolang"

# Verify paths
print(f"Base path: {BASE_PATH}")
print(f"Path exists: {os.path.exists(BASE_PATH)}")

# ============= CELL 3: Load SMPL-X Model =============
import torch
import smplx
import shutil

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Create smplx subfolder if needed
SMPLX_DIR = os.path.join(BASE_PATH, "models", "smplx")
os.makedirs(SMPLX_DIR, exist_ok=True)

# Check if model file is in the right location
source_model = os.path.join(BASE_PATH, "models", "SMPLX_NEUTRAL.npz")
target_model = os.path.join(SMPLX_DIR, "SMPLX_NEUTRAL.npz")

if os.path.exists(source_model) and not os.path.exists(target_model):
    shutil.copy(source_model, target_model)
    print(f"✓ Copied model to: {SMPLX_DIR}")

# Load SMPL-X model
smplx_model = smplx.create(
    model_path=os.path.join(BASE_PATH, "models"),
    model_type='smplx',
    gender='neutral',
    use_face_contour=False,
    use_pca=False,
    flat_hand_mean=False
).to(device)

print("✓ SMPL-X model loaded with full hand articulation!")

# ============= CELL 4: Video Folder Mapping =============
VIDEO_FOLDER_MAPPING = {
    "ch07_speakerview_012": "ch07_speakerview_012_parameters",
    "ch08_speakerview_025": "ch08_speakerview_025_parameters",
    "ch09_speakerview_027": "ch09_speakerview_027_parameters",
    "ch11_speakerview_002": "ch11_speakerview_002_parameters"
}

VIDEO_INFO = {
    'ch07_speakerview_012': {'fps': 30, 'frames': 1800},
    'ch08_speakerview_025': {'fps': 30, 'frames': 1800},
    'ch09_speakerview_027': {'fps': 30, 'frames': 1800},
    'ch11_speakerview_002': {'fps': 30, 'frames': 1800}
}

print("✓ Video mapping configured!")

# ============= CELL 5: Rendering Functions =============
import numpy as np
import trimesh
import pyrender
import cv2
from PIL import Image

def load_frame_params(video_id, frame_num):
    """Load SMPL-X parameters from NPZ file"""
    folder = VIDEO_FOLDER_MAPPING[video_id]
    npz_path = os.path.join(BASE_PATH, "Extracted_parameters", folder, f"frame_{frame_num:04d}.npz")

    if not os.path.exists(npz_path):
        return None, f"NPZ file not found: {npz_path}"

    try:
        data = np.load(npz_path, allow_pickle=True)
        person_ids = data.get('person_ids', np.array([]))

        if len(person_ids) == 0:
            return None, "No person detected"

        prefix = 'person_0_smplx_'

        params = {
            'global_orient': torch.tensor(data[prefix + 'root_pose'].reshape(1, 3), dtype=torch.float32).to(device),
            'body_pose': torch.tensor(data[prefix + 'body_pose'].reshape(1, -1), dtype=torch.float32).to(device),
            'jaw_pose': torch.tensor(data[prefix + 'jaw_pose'].reshape(1, 3), dtype=torch.float32).to(device),
            'betas': torch.tensor(data[prefix + 'shape'].reshape(1, -1), dtype=torch.float32).to(device),
            'expression': torch.tensor(data[prefix + 'expr'].reshape(1, -1), dtype=torch.float32).to(device),
            'left_hand_pose': torch.tensor(data[prefix + 'lhand_pose'].reshape(1, 45), dtype=torch.float32).to(device),
            'right_hand_pose': torch.tensor(data[prefix + 'rhand_pose'].reshape(1, 45), dtype=torch.float32).to(device),
            'leye_pose': torch.zeros((1, 3), dtype=torch.float32).to(device),
            'reye_pose': torch.zeros((1, 3), dtype=torch.float32).to(device)
        }

        cam_trans = data.get('person_0_cam_trans')
        return (params, cam_trans), None

    except Exception as e:
        return None, str(e)

def render_mesh(vertices, faces, img_size=720):
    """Render 3D mesh to image"""
    try:
        # Create mesh
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

        # Center and scale
        bounds = mesh.bounds
        center = bounds.mean(axis=0)
        mesh.vertices -= center
        scale = 2.0 / (bounds[1] - bounds[0]).max()
        mesh.vertices *= scale

        # Material
        material = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=[0.8, 0.8, 0.8, 1.0],
            metallicFactor=0.0,
            roughnessFactor=0.7
        )
        py_mesh = pyrender.Mesh.from_trimesh(mesh, material=material)

        # Scene
        scene = pyrender.Scene(bg_color=[0.96, 0.96, 0.96, 1.0])
        scene.add(py_mesh)

        # Camera
        camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
        camera_pose = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 2.5],
            [0.0, 0.0, 0.0, 1.0]
        ])
        scene.add(camera, pose=camera_pose)

        # Lights
        light1 = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
        light2 = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.0)
        scene.add(light1, pose=camera_pose)
        scene.add(light2, pose=camera_pose)

        # Render
        renderer = pyrender.OffscreenRenderer(img_size, img_size)
        color, _ = renderer.render(scene)
        renderer.delete()

        return color, None

    except Exception as e:
        return None, str(e)

def render_frame(video_id, frame_num):
    """Render a single frame"""
    # Load parameters
    result, error = load_frame_params(video_id, frame_num)
    if error:
        return None, error

    params, cam_trans = result

    # Generate mesh
    with torch.no_grad():
        output = smplx_model(**params)
        vertices = output.vertices[0].cpu().numpy()
        faces = smplx_model.faces

    if cam_trans is not None:
        vertices += cam_trans

    # Render
    img, error = render_mesh(vertices, faces)
    return img, error

print("✓ Rendering functions loaded!")

# ============= CELL 6: FastAPI Server =============
from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
import base64
import io
from typing import Optional
import asyncio

app = FastAPI(title="ECOLANG Rendering API")

# Store rendering progress
render_progress = {}

class RenderRequest(BaseModel):
    video_id: str

@app.get("/")
async def root():
    return {"message": "ECOLANG Rendering API", "status": "running"}

@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "message": "Colab API is running",
        "device": str(device),
        "model_loaded": True
    }

@app.post("/render_video")
async def render_video_endpoint(request: RenderRequest):
    """Render full video and return URL"""
    video_id = request.video_id

    if video_id not in VIDEO_INFO:
        return JSONResponse(
            status_code=404,
            content={"success": False, "error": f"Video {video_id} not found"}
        )

    try:
        # Get video info
        info = VIDEO_INFO[video_id]
        total_frames = info['frames']
        fps = info['fps']

        # Output video path
        output_dir = os.path.join(BASE_PATH, "rendered_videos")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{video_id}_rendered.mp4")

        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (720, 720))

        render_progress[video_id] = {"current": 0, "total": total_frames, "status": "rendering"}

        # Render each frame
        for frame_num in range(1, total_frames + 1):
            img, error = render_frame(video_id, frame_num)

            if img is not None:
                # Convert RGB to BGR for OpenCV
                frame_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                video_writer.write(frame_bgr)
            else:
                print(f"Warning: Frame {frame_num} failed: {error}")

            # Update progress
            render_progress[video_id]["current"] = frame_num

            if frame_num % 100 == 0:
                print(f"Rendered {frame_num}/{total_frames} frames")

        video_writer.release()
        render_progress[video_id]["status"] = "complete"

        # Upload to Google Drive and get shareable link
        video_url = f"https://drive.google.com/file/d/{video_id}/preview"  # You'll need to implement actual upload

        return {
            "success": True,
            "video_url": output_path,  # For now, return local path
            "frames_rendered": total_frames
        }

    except Exception as e:
        render_progress[video_id]["status"] = "error"
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

@app.get("/render_progress/{video_id}")
async def get_progress(video_id: str):
    """Get rendering progress"""
    if video_id in render_progress:
        return render_progress[video_id]
    return {"current": 0, "total": 0, "status": "not_started"}

print("✓ FastAPI server configured!")

# ============= CELL 7: Start Server with ngrok =============
import nest_asyncio
nest_asyncio.apply()

# Start ngrok tunnel
!ngrok authtoken YOUR_NGROK_TOKEN  # Replace with your token
from pyngrok import ngrok

# Create tunnel
public_url = ngrok.connect(8000)
print(f"\n🌐 Public URL: {public_url}")
print(f"📋 Add this URL to Streamlit secrets as COLAB_API_URL\n")

# Start server
import uvicorn
config = uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="info")
server = uvicorn.Server(config)
await server.serve()
