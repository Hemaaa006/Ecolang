"""
ECOLANG Colab Backend - AUTOMATED VERSION
Renders video with proper camera intrinsics and uploads to Drive automatically
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
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google.auth import default

def load_frame_params(video_id, frame_num):
    """Load SMPL-X parameters and camera intrinsics from NPZ file"""
    folder = VIDEO_FOLDER_MAPPING[video_id]
    npz_path = os.path.join(BASE_PATH, "Extracted_parameters", folder, f"frame_{frame_num:04d}_params.npz")

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

        # Extract camera intrinsics
        fx, fy = map(float, data.get('person_0_focal', [1200.0, 1200.0]))
        cx, cy = map(float, data.get('person_0_princpt', [360.0, 360.0]))

        return (params, cam_trans, fx, fy, cx, cy), None

    except Exception as e:
        return None, str(e)

def render_mesh_with_camera(vertices, faces, fx, fy, cx, cy, cam_trans, img_size=720):
    """Render 3D mesh using proper camera parameters with upper body crop"""
    H, W = img_size, img_size
    MARGIN = 0.05
    UPPER_FRAC = 0.55  # Keep top 55% of body
    PAD_X = 0.60       # Horizontal padding
    PAD_Y = 0.14       # Vertical padding

    # Helper function for 2D projection
    def project_uv(verts_cam, fx, fy, cx, cy):
        X, Y, Z = verts_cam[:,0], verts_cam[:,1], verts_cam[:,2]
        u = fx * (X / Z) + cx
        v = fy * (Y / Z) + cy
        return np.stack([u, v], axis=1)

    try:
        # 1. Put mesh in camera coordinates
        verts_cam = vertices + cam_trans[None, :]

        # 2. Auto-fit to viewport (scale Z if needed)
        uv = project_uv(verts_cam, fx, fy, cx, cy)
        umin, vmin = uv.min(axis=0)
        umax, vmax = uv.max(axis=0)
        bbox_w, bbox_h = umax - umin, vmax - vmin

        W_eff = W * (1.0 - 2*MARGIN)
        H_eff = H * (1.0 - 2*MARGIN)
        k = max(bbox_w / max(W_eff, 1e-6), bbox_h / max(H_eff, 1e-6), 1.0)
        verts_cam[:,2] *= k

        # 3. Recenter in viewport
        uv = project_uv(verts_cam, fx, fy, cx, cy)
        uc, vc = uv.mean(axis=0)
        du = (W/2.0) - uc
        dv = (H/2.0) - vc
        z_mean = np.median(verts_cam[:,2])
        verts_cam[:,0] += (du * z_mean) / fx
        verts_cam[:,1] += (dv * z_mean) / fy

        # 4. Convert to OpenGL coordinates (flip Y and Z)
        verts_gl = verts_cam.copy()
        verts_gl[:,1] *= -1.0
        verts_gl[:,2] *= -1.0

        # 5. Create scene with proper camera
        scene = pyrender.Scene(bg_color=[0.96, 0.96, 0.96, 1.0])
        material = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=[0.86, 0.86, 0.86, 1.0],
            metallicFactor=0.0,
            roughnessFactor=0.7
        )
        tri = trimesh.Trimesh(verts_gl, faces, process=False)
        scene.add(pyrender.Mesh.from_trimesh(tri, material=material, smooth=True))

        # Use IntrinsicsCamera with actual focal length and principal point
        camera = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy)
        scene.add(camera, pose=np.eye(4, dtype=np.float32))

        # 6. Render with lighting
        scene.add(pyrender.DirectionalLight(intensity=3.0), pose=np.eye(4, dtype=np.float32))
        L2 = np.eye(4, dtype=np.float32)
        L2[:3,3] = np.array([-1.0, -0.3, 2.0], dtype=np.float32)
        scene.add(pyrender.DirectionalLight(intensity=1.8), pose=L2)

        renderer = pyrender.OffscreenRenderer(viewport_width=W, viewport_height=H)
        rgba, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        renderer.delete()

        # 7. Body-aware crop (upper torso only)
        uv = project_uv(verts_cam, fx, fy, cx, cy)
        umin, vmin = uv.min(axis=0)
        umax, vmax = uv.max(axis=0)
        bbox_w, bbox_h = umax - umin, vmax - vmin

        left = umin - PAD_X * bbox_w
        right = umax + PAD_X * bbox_w
        top = vmin - PAD_Y * bbox_h
        bottom = vmin + UPPER_FRAC * bbox_h

        # Clamp to canvas bounds
        left = max(0.0, left)
        right = min(float(W), right)
        top = max(0.0, top)
        bottom = min(float(H), bottom)

        x0, y0 = int(round(left)), int(round(top))
        x1, y1 = int(round(right)), int(round(bottom))

        if x1 <= x0 or y1 <= y0:
            x0, y0, x1, y1 = 0, 0, W, H

        cropped = rgba[y0:y1, x0:x1, :3]  # RGB only

        return cropped, None

    except Exception as e:
        return None, str(e)

def render_frame(video_id, frame_num):
    """Render a single frame"""
    result, error = load_frame_params(video_id, frame_num)
    if error:
        return None, error

    params, cam_trans, fx, fy, cx, cy = result

    with torch.no_grad():
        output = smplx_model(**params)
        vertices = output.vertices[0].cpu().numpy()
        faces = smplx_model.faces

    if cam_trans is not None:
        vertices += cam_trans

    img, error = render_mesh_with_camera(vertices, faces, fx, fy, cx, cy, cam_trans)
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

def upload_to_drive_auto(file_path, file_name):
    """Upload file to Google Drive with automatic public sharing"""
    try:
        creds, _ = default()
        service = build('drive', 'v3', credentials=creds)

        # Upload file
        file_metadata = {'name': file_name, 'mimeType': 'video/mp4'}
        media = MediaFileUpload(file_path, mimetype='video/mp4', resumable=True)

        file = service.files().create(
            body=file_metadata,
            media_body=media,
            fields='id, name, size, webViewLink'
        ).execute()

        file_id = file.get('id')

        # Set public permissions automatically
        service.permissions().create(
            fileId=file_id,
            body={'type': 'anyone', 'role': 'reader'}
        ).execute()

        size_mb = int(file.get('size', 0)) / (1024 * 1024)

        return {
            'file_id': file_id,
            'preview_url': f"https://drive.google.com/file/d/{file_id}/preview",
            'embed_url': f"https://drive.google.com/file/d/{file_id}/preview",
            'size_mb': size_mb
        }
    except Exception as e:
        print(f"Upload error: {e}")
        return None

print("✓ Rendering functions loaded!")

# ============= CELL 6: FastAPI with Automatic Drive Upload =============
from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import time

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
        "model_loaded": True
    }

def render_video_background(video_id: str):
    """Background task for rendering video"""
    try:
        info = VIDEO_INFO[video_id]
        total_frames = info['frames']
        fps = info['fps']

        output_dir = os.path.join(BASE_PATH, "rendered_videos")
        os.makedirs(output_dir, exist_ok=True)
        temp_path = os.path.join(output_dir, f"{video_id}_temp.mp4")
        final_path = os.path.join(output_dir, f"{video_id}_rendered.mp4")

        # Get first frame to determine output size
        first_img, _ = render_frame(video_id, 1)
        if first_img is not None:
            output_h, output_w = first_img.shape[:2]
        else:
            output_h, output_w = 720, 720

        # Video writer with standard codec
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(temp_path, fourcc, fps, (output_w, output_h))

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
                # Resize if needed
                if img.shape[:2] != (output_h, output_w):
                    img = cv2.resize(img, (output_w, output_h))
                frame_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                video_writer.write(frame_bgr)
                last_valid_frame = frame_bgr
            elif last_valid_frame is not None:
                video_writer.write(last_valid_frame)
            else:
                black_frame = np.zeros((output_h, output_w, 3), dtype=np.uint8)
                video_writer.write(black_frame)

            render_jobs[video_id]["current"] = frame_num

            if frame_num % 50 == 0:
                elapsed = time.time() - render_jobs[video_id]["start_time"]
                fps_rate = frame_num / elapsed
                eta = (total_frames - frame_num) / fps_rate if fps_rate > 0 else 0
                print(f"[{video_id}] {frame_num}/{total_frames} | {fps_rate:.1f} fps | ETA: {eta:.0f}s")

        video_writer.release()

        # Convert to web-compatible format
        print(f"Converting to web-compatible format...")
        convert_to_web_compatible(temp_path, final_path)
        os.remove(temp_path)

        # Upload to Drive automatically
        print(f"Uploading to Google Drive...")
        upload_result = upload_to_drive_auto(final_path, f"{video_id}_rendered.mp4")

        if upload_result:
            render_jobs[video_id]["status"] = "complete"
            render_jobs[video_id]["video_url"] = upload_result['preview_url']
            render_jobs[video_id]["file_id"] = upload_result['file_id']
            render_jobs[video_id]["size_mb"] = upload_result['size_mb']
            render_jobs[video_id]["drive_path"] = final_path

            total_time = time.time() - render_jobs[video_id]["start_time"]
            print(f"✓ Complete: {final_path} in {total_time:.1f}s")
            print(f"✓ Uploaded: {upload_result['file_id']} ({upload_result['size_mb']:.1f} MB)")
            print(f"✓ Preview URL: {upload_result['preview_url']}")
        else:
            # Rendering succeeded but upload failed
            render_jobs[video_id]["status"] = "complete"
            render_jobs[video_id]["video_url"] = None
            render_jobs[video_id]["local_path"] = final_path
            print(f"⚠️ Upload failed - video saved locally at {final_path}")

    except Exception as e:
        print(f"Error: {e}")
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

    if video_id in render_jobs and render_jobs[video_id]["status"] == "rendering":
        return {"success": True, "message": "Already rendering", "job_id": video_id}

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
        response["video_url"] = job.get("video_url")
        response["file_id"] = job.get("file_id")
        response["size_mb"] = job.get("size_mb", 0)
        response["drive_path"] = job.get("drive_path")
        response["local_path"] = job.get("local_path")

    if job["status"] == "error":
        response["error"] = job.get("error", "Unknown error")

    return response

print("✓ FastAPI server configured!")

# ============= CELL 7: Start Server =============
import nest_asyncio
nest_asyncio.apply()

from pyngrok import ngrok

ngrok.set_auth_token("YOUR_NGROK_TOKEN")  # REPLACE THIS

public_url = ngrok.connect(8000)
print(f"\n🌐 Public URL: {public_url}")
print(f"📋 Add to Streamlit secrets: COLAB_API_URL = \"{public_url}\"\n")

import uvicorn
config = uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="info")
server = uvicorn.Server(config)
await server.serve()
