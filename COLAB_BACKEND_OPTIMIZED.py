"""
ECOLANG Colab Backend - OPTIMIZED for Fast Rendering
Run this in Google Colab to create the rendering API
"""

# ============= CELL 1: Setup Environment =============
!apt-get install -y xvfb
!pip install -q pyvirtualdisplay smplx trimesh pyrender nest_asyncio fastapi uvicorn pyngrok opencv-python pillow

from pyvirtualdisplay import Display
import os
display = Display(visible=0, size=(1400, 900))
display.start()
os.environ['PYOPENGL_PLATFORM'] = 'egl'

print("✓ Environment setup complete!")

# ============= CELL 2: Mount Google Drive =============
from google.colab import drive
drive.mount('/content/drive')

BASE_PATH = "/content/drive/MyDrive/ecolang"
print(f"Base path: {BASE_PATH}")
print(f"Path exists: {os.path.exists(BASE_PATH)}")

# ============= CELL 3: Load SMPL-X Model =============
import torch
import smplx
import shutil

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

SMPLX_DIR = os.path.join(BASE_PATH, "models", "smplx")
os.makedirs(SMPLX_DIR, exist_ok=True)

source_model = os.path.join(BASE_PATH, "models", "SMPLX_NEUTRAL.npz")
target_model = os.path.join(SMPLX_DIR, "SMPLX_NEUTRAL.npz")

if os.path.exists(source_model) and not os.path.exists(target_model):
    shutil.copy(source_model, target_model)
    print(f"✓ Copied model to: {SMPLX_DIR}")

smplx_model = smplx.create(
    model_path=os.path.join(BASE_PATH, "models"),
    model_type='smplx',
    gender='neutral',
    use_face_contour=False,
    use_pca=False,
    flat_hand_mean=False
).to(device)

print("✓ SMPL-X model loaded!")

# ============= CELL 4: Video Configuration =============
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

# ============= CELL 5: OPTIMIZED Rendering Functions =============
import numpy as np
import trimesh
import pyrender
import cv2
from concurrent.futures import ThreadPoolExecutor
import threading

# Initialize renderer once (reuse it)
renderer_lock = threading.Lock()
global_renderer = None

def get_renderer(img_size=720):
    """Get or create renderer (thread-safe singleton)"""
    global global_renderer
    if global_renderer is None:
        with renderer_lock:
            if global_renderer is None:
                global_renderer = pyrender.OffscreenRenderer(img_size, img_size)
    return global_renderer

def load_frame_params(video_id, frame_num):
    """Load SMPL-X parameters from NPZ file"""
    folder = VIDEO_FOLDER_MAPPING[video_id]
    npz_path = os.path.join(BASE_PATH, "Extracted_parameters", folder, f"frame_{frame_num:04d}_params.npz")

    if not os.path.exists(npz_path):
        return None, f"NPZ not found"

    try:
        data = np.load(npz_path, allow_pickle=True)
        person_ids = data.get('person_ids', np.array([]))

        if len(person_ids) == 0:
            return None, "No person"

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
    """Render 3D mesh to image (optimized)"""
    try:
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

        # Center and scale
        bounds = mesh.bounds
        center = bounds.mean(axis=0)
        mesh.vertices -= center
        scale = 2.0 / (bounds[1] - bounds[0]).max()
        mesh.vertices *= scale

        # Material and mesh
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
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
        scene.add(light, pose=camera_pose)

        # Render using global renderer
        with renderer_lock:
            renderer = get_renderer(img_size)
            color, _ = renderer.render(scene)

        return color, None

    except Exception as e:
        return None, str(e)

def render_frame(video_id, frame_num):
    """Render a single frame"""
    result, error = load_frame_params(video_id, frame_num)
    if error:
        return None, error

    params, cam_trans = result

    with torch.no_grad():
        output = smplx_model(**params)
        vertices = output.vertices[0].cpu().numpy()
        faces = smplx_model.faces

    if cam_trans is not None:
        vertices += cam_trans

    img, error = render_mesh(vertices, faces)
    return img, error

print("✓ Rendering functions loaded!")

# ============= CELL 6: FastAPI with Background Tasks =============
from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import asyncio
import time

app = FastAPI(title="ECOLANG Rendering API")

# Store rendering jobs
render_jobs = {}

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

def render_video_background(video_id: str):
    """Background task for rendering video"""
    try:
        info = VIDEO_INFO[video_id]
        total_frames = info['frames']
        fps = info['fps']

        # Output path
        output_dir = os.path.join(BASE_PATH, "rendered_videos")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{video_id}_rendered.mp4")

        # Video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (720, 720))

        render_jobs[video_id] = {
            "status": "rendering",
            "current": 0,
            "total": total_frames,
            "output_path": output_path,
            "start_time": time.time()
        }

        last_valid_frame = None

        # Render frames
        for frame_num in range(1, total_frames + 1):
            img, error = render_frame(video_id, frame_num)

            if img is not None:
                frame_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                video_writer.write(frame_bgr)
                last_valid_frame = frame_bgr
            elif last_valid_frame is not None:
                # Use last valid frame as fallback
                video_writer.write(last_valid_frame)
            else:
                # Black frame as last resort
                black_frame = np.zeros((720, 720, 3), dtype=np.uint8)
                video_writer.write(black_frame)

            # Update progress
            render_jobs[video_id]["current"] = frame_num

            if frame_num % 50 == 0:
                elapsed = time.time() - render_jobs[video_id]["start_time"]
                fps_rate = frame_num / elapsed
                eta = (total_frames - frame_num) / fps_rate if fps_rate > 0 else 0
                print(f"[{video_id}] {frame_num}/{total_frames} frames | {fps_rate:.1f} fps | ETA: {eta:.0f}s")

        video_writer.release()

        # Mark complete
        render_jobs[video_id]["status"] = "complete"
        render_jobs[video_id]["video_url"] = output_path
        total_time = time.time() - render_jobs[video_id]["start_time"]
        print(f"✓ Rendering complete: {output_path} in {total_time:.1f}s")

    except Exception as e:
        print(f"Error rendering {video_id}: {e}")
        render_jobs[video_id]["status"] = "error"
        render_jobs[video_id]["error"] = str(e)

@app.post("/render_video")
async def start_render(request: RenderRequest, background_tasks: BackgroundTasks):
    """Start video rendering in background"""
    video_id = request.video_id

    if video_id not in VIDEO_INFO:
        return JSONResponse(
            status_code=404,
            content={"success": False, "error": f"Video {video_id} not found"}
        )

    # Check if already rendering
    if video_id in render_jobs and render_jobs[video_id]["status"] == "rendering":
        return {
            "success": True,
            "message": "Already rendering",
            "job_id": video_id
        }

    # Start background rendering
    background_tasks.add_task(render_video_background, video_id)

    return {
        "success": True,
        "message": "Rendering started",
        "job_id": video_id,
        "total_frames": VIDEO_INFO[video_id]["frames"]
    }

@app.get("/render_progress/{video_id}")
async def get_progress(video_id: str):
    """Get rendering progress"""
    if video_id not in render_jobs:
        return {"status": "not_started", "current": 0, "total": 0}

    job = render_jobs[video_id]
    response = {
        "status": job["status"],
        "current": job["current"],
        "total": job["total"]
    }

    if job["status"] == "complete":
        response["video_url"] = job["output_path"]

    if job["status"] == "error":
        response["error"] = job.get("error", "Unknown error")

    return response

print("✓ FastAPI server configured!")

# ============= CELL 7: Start Server =============
import nest_asyncio
nest_asyncio.apply()

from pyngrok import ngrok

# Set your ngrok token (get it from https://dashboard.ngrok.com/get-started/your-authtoken)
ngrok.set_auth_token("YOUR_NGROK_TOKEN")  # REPLACE THIS

# Create tunnel
public_url = ngrok.connect(8000)
print(f"\n🌐 Public URL: {public_url}")
print(f"📋 Add this to Streamlit secrets: COLAB_API_URL = \"{public_url}\"\n")

# Start server
import uvicorn
config = uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="info")
server = uvicorn.Server(config)
await server.serve()
