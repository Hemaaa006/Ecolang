"""
ECOLANG Colab Backend - AUTOMATED VERSION
Renders video with proper camera intrinsics and uploads to Drive automatically
Uses the exact working rendering script provided by user
"""

# ============= CELL 1: Setup Environment =============
!apt-get install -y xvfb ffmpeg
!pip install -q pyvirtualdisplay smplx trimesh pyrender opencv-python pillow google-api-python-client google-auth PyOpenGL imageio nest_asyncio fastapi uvicorn pyngrok

from pyvirtualdisplay import Display
import os
display = Display(visible=0, size=(1400, 900))
display.start()
os.environ['PYOPENGL_PLATFORM'] = 'egl'

print("Environment setup complete!")

# ============= CELL 2: Mount Google Drive =============
from google.colab import drive
drive.mount('/content/drive')

BASE_PATH = "/content/drive/MyDrive/ecolang"
print(f"Base path: {BASE_PATH}")
print(f"Path exists: {os.path.exists(BASE_PATH)}")

# ============= CELL 3: Load SMPL-X Model =============
import torch
import smplx

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

MODEL_PATH = os.path.join(BASE_PATH, "models", "SMPLX_NEUTRAL.npz")
print(f"Model path: {MODEL_PATH}")
print(f"Model exists: {os.path.exists(MODEL_PATH)}")

smplx_model = smplx.create(
    model_path=MODEL_PATH,
    model_type='smplx',
    gender='neutral',
    num_betas=10,
    num_expression_coeffs=10,
    use_pca=False,
    use_face_contour=True,
    ext='npz'
).to(device).eval()

print("SMPL-X model loaded!")

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

print("Video mapping configured!")

# ============= CELL 5: Rendering Functions (Exact Working Script) =============
import numpy as np
import trimesh
import pyrender
import cv2
import subprocess
from PIL import Image
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google.auth import default

# Rendering parameters (same as working script)
H, W       = 720, 1280           # output canvas (16:9)
MARGIN     = 0.05                # stage the full-frame auto-fit
UPPER_FRAC = 0.55                # keep top 55% of body bbox
PAD_X      = 0.60                # extra horizontal room
PAD_Y      = 0.14                # extra vertical headroom
BG         = (245, 245, 245, 255)

def project_uv(verts_cam, fx, fy, cx, cy):
    """Project 3D vertices to 2D image coordinates"""
    X, Y, Z = verts_cam[:,0], verts_cam[:,1], verts_cam[:,2]
    u = fx * (X / Z) + cx
    v = fy * (Y / Z) + cy
    return np.stack([u, v], axis=1)

def render_frame(video_id, frame_num):
    """Render a single frame using the exact working script logic"""
    try:
        folder = VIDEO_FOLDER_MAPPING[video_id]
        npz_path = os.path.join(BASE_PATH, "Extracted_parameters", folder, f"frame_{frame_num:04d}_params.npz")

        if not os.path.exists(npz_path):
            return None, "NPZ not found"

        # 1) Load NPZ and gather parameters
        Z = np.load(npz_path, allow_pickle=False)
        PERSON_ID = 0
        pfx = f"person_{PERSON_ID}_smplx_"

        params_np = {
            "global_orient":   Z[pfx+"root_pose"].reshape(1,3).astype(np.float32),
            "body_pose":       Z[pfx+"body_pose"].reshape(1,-1).astype(np.float32),
            "left_hand_pose":  Z[pfx+"lhand_pose"].reshape(1,-1).astype(np.float32),
            "right_hand_pose": Z[pfx+"rhand_pose"].reshape(1,-1).astype(np.float32),
            "jaw_pose":        Z[pfx+"jaw_pose"].reshape(1,3).astype(np.float32),
            "betas":           Z[pfx+"shape"].reshape(1,-1).astype(np.float32),
            "expression":      Z[pfx+"expr"].reshape(1,-1).astype(np.float32),
            "leye_pose":       np.zeros((1,3), dtype=np.float32),
            "reye_pose":       np.zeros((1,3), dtype=np.float32),
        }
        cam_trans = Z[f"person_{PERSON_ID}_cam_trans"].astype(np.float32)
        fx, fy    = map(float, Z[f"person_{PERSON_ID}_focal"])
        cx, cy    = map(float, Z[f"person_{PERSON_ID}_princpt"])

        # 2) Build SMPL-X mesh
        with torch.no_grad():
            tens  = {k: torch.from_numpy(v).float().to(device) for k, v in params_np.items()}
            out   = smplx_model(**tens)
            verts = out.vertices[0].cpu().numpy().astype(np.float32)
        faces = smplx_model.faces

        # 3) Put mesh in camera coordinates
        verts_cam = verts + cam_trans[None, :]

        # 4) Auto-fit to viewport (push Z if needed; recenter)
        uv = project_uv(verts_cam, fx, fy, cx, cy)
        umin, vmin = uv.min(axis=0)
        umax, vmax = uv.max(axis=0)
        bbox_w, bbox_h = umax - umin, vmax - vmin

        W_eff = W * (1.0 - 2*MARGIN)
        H_eff = H * (1.0 - 2*MARGIN)
        k = max(bbox_w / max(W_eff,1e-6), bbox_h / max(H_eff,1e-6), 1.0)
        verts_cam[:,2] *= k
        uv = project_uv(verts_cam, fx, fy, cx, cy)

        uc, vc = uv.mean(axis=0)
        du = (W/2.0) - uc
        dv = (H/2.0) - vc
        z_mean = np.median(verts_cam[:,2])
        verts_cam[:,0] += (du * z_mean) / fx
        verts_cam[:,1] += (dv * z_mean) / fy

        # 5) Convert to OpenGL coords for pyrender
        verts_gl       = verts_cam.copy()
        verts_gl[:,1] *= -1.0
        verts_gl[:,2] *= -1.0

        # 6) Render full frame
        scene = pyrender.Scene(bg_color=[c/255.0 for c in BG])
        material = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=[0.86, 0.86, 0.86, 1.0],
            metallicFactor=0.0,
            roughnessFactor=0.7
        )
        tri = trimesh.Trimesh(verts_gl, faces, process=False)
        scene.add(pyrender.Mesh.from_trimesh(tri, material=material, smooth=True))

        camera = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy)
        scene.add(camera, pose=np.eye(4, dtype=np.float32))

        scene.add(pyrender.DirectionalLight(intensity=3.0), pose=np.eye(4, dtype=np.float32))
        L2 = np.eye(4, dtype=np.float32)
        L2[:3,3] = np.array([-1.0, -0.3, 2.0], dtype=np.float32)
        scene.add(pyrender.DirectionalLight(intensity=1.8), pose=L2)

        renderer = pyrender.OffscreenRenderer(viewport_width=W, viewport_height=H)
        rgba, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        renderer.delete()

        # 7) Body-aware crop (upper torso, with extra left/right space)
        uv = project_uv(verts_cam, fx, fy, cx, cy)
        umin, vmin = uv.min(axis=0)
        umax, vmax = uv.max(axis=0)
        bbox_w, bbox_h = umax - umin, vmax - vmin

        left   = umin - PAD_X * bbox_w
        right  = umax + PAD_X * bbox_w
        top    = vmin - PAD_Y * bbox_h
        bottom = vmin + UPPER_FRAC * bbox_h

        # Clamp to canvas
        left   = max(0.0, left)
        right  = min(float(W), right)
        top    = max(0.0, top)
        bottom = min(float(H), bottom)

        # Make sure we have a valid crop
        x0, y0 = int(round(left)),  int(round(top))
        x1, y1 = int(round(right)), int(round(bottom))
        if x1 <= x0 or y1 <= y0:
            x0, y0, x1, y1 = 0, 0, W, H

        cropped = rgba[y0:y1, x0:x1, :3]  # RGB only

        return cropped, None

    except Exception as e:
        return None, str(e)

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

        file_metadata = {'name': file_name, 'mimeType': 'video/mp4'}
        media = MediaFileUpload(file_path, mimetype='video/mp4', resumable=True)

        file = service.files().create(
            body=file_metadata,
            media_body=media,
            fields='id, name, size, webViewLink'
        ).execute()

        file_id = file.get('id')

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

def check_existing_rendered_video(video_id):
    """Check if a rendered video already exists in Google Drive"""
    try:
        output_dir = os.path.join(BASE_PATH, "rendered_videos")
        file_path = os.path.join(output_dir, f"{video_id}_rendered.mp4")

        # Check if file exists locally
        if not os.path.exists(file_path):
            return None

        # File exists - now search for it in Google Drive or upload it
        creds, _ = default()
        service = build('drive', 'v3', credentials=creds)

        # Search for the file in Google Drive
        query = f"name='{video_id}_rendered.mp4' and trashed=false"
        results = service.files().list(
            q=query,
            fields='files(id, name, size, webViewLink)',
            pageSize=10
        ).execute()

        files = results.get('files', [])

        if files:
            # File found in Drive - use existing
            file = files[0]
            file_id = file.get('id')

            # Ensure it's publicly accessible
            try:
                service.permissions().create(
                    fileId=file_id,
                    body={'type': 'anyone', 'role': 'reader'}
                ).execute()
            except:
                pass  # Permission might already exist

            size_mb = int(file.get('size', 0)) / (1024 * 1024)

            return {
                'file_id': file_id,
                'preview_url': f"https://drive.google.com/file/d/{file_id}/preview",
                'embed_url': f"https://drive.google.com/file/d/{file_id}/preview",
                'size_mb': size_mb,
                'drive_path': file_path,
                'already_exists': True
            }
        else:
            # File exists locally but not in Drive - upload it
            upload_result = upload_to_drive_auto(file_path, f"{video_id}_rendered.mp4")
            if upload_result:
                upload_result['drive_path'] = file_path
                upload_result['already_exists'] = True
                return upload_result

        return None

    except Exception as e:
        print(f"Error checking existing video: {e}")
        return None

print("Rendering functions loaded!")

# ============= CELL 6: FastAPI =============
from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
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
    return {"status": "ok", "message": "Colab API is running", "device": str(device), "model_loaded": True}

def render_video_background(video_id: str):
    try:
        info = VIDEO_INFO[video_id]
        total_frames = info['frames']
        fps = info['fps']

        output_dir = os.path.join(BASE_PATH, "rendered_videos")
        os.makedirs(output_dir, exist_ok=True)
        temp_path = os.path.join(output_dir, f"{video_id}_temp.mp4")
        final_path = os.path.join(output_dir, f"{video_id}_rendered.mp4")

        first_img, _ = render_frame(video_id, 1)
        if first_img is not None:
            output_h, output_w = first_img.shape[:2]
        else:
            output_h, output_w = 720, 1280

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(temp_path, fourcc, fps, (output_w, output_h))

        render_jobs[video_id] = {"status": "rendering", "current": 0, "total": total_frames, "start_time": time.time()}

        last_valid_frame = None

        for frame_num in range(1, total_frames + 1):
            img, error = render_frame(video_id, frame_num)

            if img is not None:
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

        print(f"Converting to web-compatible format...")
        convert_to_web_compatible(temp_path, final_path)
        os.remove(temp_path)

        print(f"Uploading to Google Drive...")
        upload_result = upload_to_drive_auto(final_path, f"{video_id}_rendered.mp4")

        if upload_result:
            render_jobs[video_id]["status"] = "complete"
            render_jobs[video_id]["video_url"] = upload_result['preview_url']
            render_jobs[video_id]["file_id"] = upload_result['file_id']
            render_jobs[video_id]["size_mb"] = upload_result['size_mb']
            render_jobs[video_id]["drive_path"] = final_path

            total_time = time.time() - render_jobs[video_id]["start_time"]
            print(f"Complete: {final_path} in {total_time:.1f}s")
            print(f"Uploaded: {upload_result['file_id']} ({upload_result['size_mb']:.1f} MB)")
            print(f"Preview URL: {upload_result['preview_url']}")
        else:
            render_jobs[video_id]["status"] = "complete"
            render_jobs[video_id]["video_url"] = None
            render_jobs[video_id]["local_path"] = final_path
            print(f"Upload failed - video saved locally at {final_path}")

    except Exception as e:
        print(f"Error: {e}")
        render_jobs[video_id]["status"] = "error"
        render_jobs[video_id]["error"] = str(e)

@app.post("/render_video")
async def start_render(request: RenderRequest, background_tasks: BackgroundTasks):
    video_id = request.video_id
    if video_id not in VIDEO_INFO:
        return JSONResponse(status_code=404, content={"success": False, "error": f"Video {video_id} not found"})

    # Check if video already exists before rendering
    existing_video = check_existing_rendered_video(video_id)
    if existing_video:
        # Video already exists - return existing info
        render_jobs[video_id] = {
            "status": "complete",
            "current": VIDEO_INFO[video_id]["frames"],
            "total": VIDEO_INFO[video_id]["frames"],
            "video_url": existing_video['preview_url'],
            "file_id": existing_video['file_id'],
            "size_mb": existing_video['size_mb'],
            "drive_path": existing_video.get('drive_path'),
            "already_exists": True
        }
        return {
            "success": True,
            "message": "Rendered video already exists",
            "job_id": video_id,
            "already_exists": True,
            "video_url": existing_video['preview_url'],
            "file_id": existing_video['file_id'],
            "size_mb": existing_video['size_mb']
        }

    # Check if currently rendering
    if video_id in render_jobs and render_jobs[video_id]["status"] == "rendering":
        return {"success": True, "message": "Already rendering", "job_id": video_id}

    # Start new rendering
    background_tasks.add_task(render_video_background, video_id)
    return {"success": True, "message": "Rendering started", "job_id": video_id, "total_frames": VIDEO_INFO[video_id]["frames"]}

@app.get("/render_progress/{video_id}")
async def get_progress(video_id: str):
    if video_id not in render_jobs:
        return {"status": "not_started", "current": 0, "total": 0}
    job = render_jobs[video_id]
    response = {"status": job["status"], "current": job["current"], "total": job["total"]}
    if job["status"] == "complete":
        response["video_url"] = job.get("video_url")
        response["file_id"] = job.get("file_id")
        response["size_mb"] = job.get("size_mb", 0)
        response["drive_path"] = job.get("drive_path")
        response["local_path"] = job.get("local_path")
    if job["status"] == "error":
        response["error"] = job.get("error", "Unknown error")
    return response

# Serve rendered videos directly when Drive upload isn't available
@app.get("/rendered_file/{video_id}")
async def get_rendered_file(video_id: str):
    path = os.path.join(BASE_PATH, "rendered_videos", f"{video_id}_rendered.mp4")
    if not os.path.exists(path):
        return JSONResponse(status_code=404, content={"error": "file_not_found"})
    # Return raw MP4; browsers can play this directly
    return FileResponse(path, media_type="video/mp4", filename=f"{video_id}_rendered.mp4")

@app.get("/rendered/{video_id}", response_class=HTMLResponse)
async def get_rendered_embed(video_id: str):
    path = os.path.join(BASE_PATH, "rendered_videos", f"{video_id}_rendered.mp4")
    if not os.path.exists(path):
        return HTMLResponse(status_code=404, content="<html><body>File not found</body></html>")
    # Simple responsive page to embed in an iframe
    return f"""<!doctype html>
<html>
  <head>
    <meta charset=\"utf-8\">
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
    <style>
      html, body {{ height:100%; margin:0; background:#000; }}
      .wrap {{ position:relative; width:100%; height:100%; }}
      video {{ position:absolute; top:0; left:0; width:100%; height:100%; background:#000; }}
    </style>
  </head>
  <body>
    <div class=\"wrap\">
      <video src=\"/rendered_file/{video_id}\" controls autoplay playsinline></video>
    </div>
  </body>
  </html>"""

print("FastAPI server configured!")

# ============= CELL 7: Start Server =============
import nest_asyncio
nest_asyncio.apply()

from pyngrok import ngrok

ngrok.set_auth_token("YOUR_NGROK_TOKEN")  # REPLACE THIS

public_url = ngrok.connect(8000)
print(f"\nPublic URL: {public_url}")
print(f"Add to Streamlit secrets: COLAB_API_URL = \"{public_url}\"\n")

import uvicorn
config = uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="info")
server = uvicorn.Server(config)
await server.serve()
