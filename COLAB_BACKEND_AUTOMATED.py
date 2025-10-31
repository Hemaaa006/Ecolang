"""
ECOLANG Colab Backend - FULLY AUTOMATED
No manual steps required - automatic Drive upload with public sharing
"""

# ============= CELL 1: Setup Environment =============
!apt-get install -y xvfb ffmpeg
!pip install -q pyvirtualdisplay smplx trimesh pyrender nest_asyncio fastapi uvicorn pyngrok opencv-python pillow google-api-python-client google-auth

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

# ============= CELL 5: Rendering Functions =============
import numpy as np
import trimesh
import pyrender
import cv2
import subprocess

def load_frame_params(video_id, frame_num):
    """Load SMPL-X parameters from NPZ file"""
    folder = VIDEO_FOLDER_MAPPING[video_id]
    npz_path = os.path.join(BASE_PATH, "Extracted_parameters", folder, f"frame_{frame_num:04d}.npz")

    if not os.path.exists(npz_path):
        return None, "NPZ not found"

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
    """Render 3D mesh to image"""
    try:
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        bounds = mesh.bounds
        center = bounds.mean(axis=0)
        mesh.vertices -= center
        scale = 2.0 / (bounds[1] - bounds[0]).max()
        mesh.vertices *= scale

        material = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=[0.8, 0.8, 0.8, 1.0],
            metallicFactor=0.0,
            roughnessFactor=0.7
        )
        py_mesh = pyrender.Mesh.from_trimesh(mesh, material=material)

        scene = pyrender.Scene(bg_color=[0.96, 0.96, 0.96, 1.0])
        scene.add(py_mesh)

        camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
        camera_pose = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 2.5],
            [0.0, 0.0, 0.0, 1.0]
        ])
        scene.add(camera, pose=camera_pose)

        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
        scene.add(light, pose=camera_pose)

        renderer = pyrender.OffscreenRenderer(img_size, img_size)
        color, _ = renderer.render(scene)
        renderer.delete()

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

def convert_to_web_compatible(input_path, output_path):
    """Convert video to web-compatible H.264 MP4"""
    cmd = [
        'ffmpeg', '-y', '-i', input_path,
        '-vcodec', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-profile:v', 'baseline',
        '-level', '3.0',
        '-movflags', '+faststart',
        output_path
    ]
    subprocess.run(cmd, check=True, capture_output=True)

print("✓ Rendering functions loaded!")

# ============= CELL 5.5: Google Drive API Auto-Upload =============
from google.colab import auth
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
import time

# Authenticate once at startup
print("Authenticating with Google Drive...")
auth.authenticate_user()
print("✓ Authentication complete!")

def upload_to_drive_auto(file_path, file_name):
    """
    Upload file to Google Drive with automatic public sharing

    Args:
        file_path: Local path to file
        file_name: Name for uploaded file

    Returns:
        dict with file_id, preview_url, embed_url, download_url
    """
    try:
        service = build('drive', 'v3')

        # File metadata
        file_metadata = {
            'name': file_name,
            'mimeType': 'video/mp4'
        }

        # Upload file
        media = MediaFileUpload(
            file_path,
            mimetype='video/mp4',
            resumable=True,
            chunksize=10 * 1024 * 1024  # 10MB chunks
        )

        print(f"📤 Uploading {file_name} to Google Drive...")
        file = service.files().create(
            body=file_metadata,
            media_body=media,
            fields='id, name, size, webViewLink'
        ).execute()

        file_id = file.get('id')
        file_size_mb = int(file.get('size', 0)) / (1024 * 1024)
        print(f"✓ Uploaded! File ID: {file_id} ({file_size_mb:.1f} MB)")

        # Set public permissions
        print("🔓 Setting public permissions...")
        permission = {
            'type': 'anyone',
            'role': 'reader',
            'allowFileDiscovery': False
        }

        service.permissions().create(
            fileId=file_id,
            body=permission,
            fields='id'
        ).execute()

        print("✓ File is now publicly accessible!")

        # Generate URLs
        preview_url = f"https://drive.google.com/file/d/{file_id}/preview"
        view_url = f"https://drive.google.com/file/d/{file_id}/view"
        download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
        embed_url = f"https://drive.google.com/file/d/{file_id}/preview"

        return {
            'success': True,
            'file_id': file_id,
            'file_name': file_name,
            'size_mb': file_size_mb,
            'preview_url': preview_url,
            'view_url': view_url,
            'download_url': download_url,
            'embed_url': embed_url,
            'web_view_link': file.get('webViewLink')
        }

    except Exception as e:
        print(f"❌ Upload failed: {str(e)}")
        return {
            'success': False,
            'error': str(e)
        }

print("✓ Google Drive auto-upload ready!")

# ============= CELL 6: FastAPI with Auto-Upload =============
from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel

app = FastAPI(title="ECOLANG Rendering API")
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
        "model_loaded": True,
        "drive_auth": True
    }

def render_video_background(video_id: str):
    """Background task for rendering video with auto-upload"""
    try:
        info = VIDEO_INFO[video_id]
        total_frames = info['frames']
        fps = info['fps']

        output_dir = os.path.join(BASE_PATH, "rendered_videos")
        os.makedirs(output_dir, exist_ok=True)
        temp_path = os.path.join(output_dir, f"{video_id}_temp.mp4")
        final_path = os.path.join(output_dir, f"{video_id}_rendered.mp4")

        # Video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(temp_path, fourcc, fps, (720, 720))

        render_jobs[video_id] = {
            "status": "rendering",
            "current": 0,
            "total": total_frames,
            "start_time": time.time()
        }

        last_valid_frame = None

        # Render all frames
        for frame_num in range(1, total_frames + 1):
            img, error = render_frame(video_id, frame_num)

            if img is not None:
                frame_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                video_writer.write(frame_bgr)
                last_valid_frame = frame_bgr
            elif last_valid_frame is not None:
                video_writer.write(last_valid_frame)
            else:
                black_frame = np.zeros((720, 720, 3), dtype=np.uint8)
                video_writer.write(black_frame)

            render_jobs[video_id]["current"] = frame_num

            if frame_num % 100 == 0:
                elapsed = time.time() - render_jobs[video_id]["start_time"]
                fps_rate = frame_num / elapsed
                eta = (total_frames - frame_num) / fps_rate if fps_rate > 0 else 0
                print(f"[{video_id}] {frame_num}/{total_frames} | {fps_rate:.1f} fps | ETA: {eta:.0f}s")

        video_writer.release()

        # Convert to web-compatible format
        print(f"\n🎬 Converting to web-compatible format...")
        convert_to_web_compatible(temp_path, final_path)
        os.remove(temp_path)

        # AUTO-UPLOAD TO DRIVE WITH PUBLIC SHARING
        print(f"\n☁️  Uploading to Google Drive with auto-sharing...")
        upload_result = upload_to_drive_auto(final_path, f"{video_id}_rendered.mp4")

        if upload_result['success']:
            render_jobs[video_id]["status"] = "complete"
            render_jobs[video_id]["video_url"] = upload_result['embed_url']
            render_jobs[video_id]["file_id"] = upload_result['file_id']
            render_jobs[video_id]["preview_url"] = upload_result['preview_url']
            render_jobs[video_id]["download_url"] = upload_result['download_url']
            render_jobs[video_id]["size_mb"] = upload_result['size_mb']

            total_time = time.time() - render_jobs[video_id]["start_time"]

            print(f"\n{'='*60}")
            print(f"✅ RENDERING COMPLETE!")
            print(f"{'='*60}")
            print(f"⏱️  Total time: {total_time:.1f}s")
            print(f"📺 Preview URL: {upload_result['preview_url']}")
            print(f"⬇️  Download: {upload_result['download_url']}")
            print(f"📁 File ID: {upload_result['file_id']}")
            print(f"💾 Size: {upload_result['size_mb']:.1f} MB")
            print(f"{'='*60}\n")
        else:
            render_jobs[video_id]["status"] = "upload_failed"
            render_jobs[video_id]["error"] = upload_result['error']
            render_jobs[video_id]["local_path"] = final_path
            print(f"⚠️  Upload failed: {upload_result['error']}")
            print(f"📍 Video saved locally: {final_path}")

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        render_jobs[video_id]["status"] = "error"
        render_jobs[video_id]["error"] = str(e)

@app.post("/render_video")
async def start_render(request: RenderRequest, background_tasks: BackgroundTasks):
    """Start video rendering with auto-upload"""
    video_id = request.video_id

    if video_id not in VIDEO_INFO:
        return JSONResponse(
            status_code=404,
            content={"success": False, "error": f"Video {video_id} not found"}
        )

    if video_id in render_jobs and render_jobs[video_id]["status"] == "rendering":
        return {"success": True, "message": "Already rendering", "job_id": video_id}

    background_tasks.add_task(render_video_background, video_id)

    return {
        "success": True,
        "message": "Rendering started (auto-upload enabled)",
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
        response["video_url"] = job.get("video_url")
        response["preview_url"] = job.get("preview_url")
        response["download_url"] = job.get("download_url")
        response["file_id"] = job.get("file_id")
        response["size_mb"] = job.get("size_mb")

    if job["status"] in ["error", "upload_failed"]:
        response["error"] = job.get("error", "Unknown error")
        response["local_path"] = job.get("local_path")

    return response

print("✓ FastAPI server configured with auto-upload!")

# ============= CELL 7: Start Server =============
import nest_asyncio
nest_asyncio.apply()

from pyngrok import ngrok

ngrok.set_auth_token("YOUR_NGROK_TOKEN")  # REPLACE THIS

public_url = ngrok.connect(8000)
print(f"\n{'='*70}")
print(f"🌐 Public URL: {public_url}")
print(f"📋 Add to Streamlit secrets: COLAB_API_URL = \"{public_url}\"")
print(f"{'='*70}\n")

import uvicorn
config = uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="info")
server = uvicorn.Server(config)
await server.serve()
