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
print("Video mapping configured!")

# ============= CELL 5: Rendering Functions (Exact Working Script) =============
import json
from pathlib import Path
import asyncio
import numpy as np
import trimesh
import pyrender
import cv2
import subprocess
import time
import threading
from typing import Optional
from PIL import Image
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload
from google.auth import default
from google.auth.transport.requests import Request
from google.auth.exceptions import DefaultCredentialsError, RefreshError
from google.oauth2 import service_account
from google.oauth2.credentials import Credentials as UserCredentials

# Rendering parameters (same as working script)
H, W       = 720, 1280           # output canvas (16:9)
MARGIN     = 0.05                # stage the full-frame auto-fit
UPPER_FRAC = 0.55                # keep top 55% of body bbox
PAD_X      = 0.60                # extra horizontal room
PAD_Y      = 0.14                # extra vertical headroom
BG         = (245, 245, 245, 255)
DRIVE_SCOPES = ['https://www.googleapis.com/auth/drive']
_drive_service_lock = threading.Lock()
_drive_service = None
_rendered_videos_folder_id: Optional[str] = None
_drive_credentials = None
_manifest_lock = threading.Lock()
_drive_auth_disabled = False
_drive_auth_error: Optional[str] = None
_drive_service_warned = False

VIDEOS_DIR = os.path.join(BASE_PATH, "videos")
PARAMETERS_DIR = os.path.join(BASE_PATH, "Extracted_parameters")
RENDERED_DIR = os.path.join(BASE_PATH, "rendered_videos")

VIDEO_FOLDER_MAPPING: dict[str, str] = {}
VIDEO_INFO: dict[str, dict] = {}
_video_library_lock = threading.Lock()
_video_library_refresh_interval = 60.0
_video_library_last_scan = 0.0

CACHE_DIR = Path(BASE_PATH) / ".cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CREDENTIAL_CACHE_PATH = CACHE_DIR / "drive_credentials.json"
FOLDER_CACHE_PATH = CACHE_DIR / "rendered_folder.json"
MANIFEST_PATH = CACHE_DIR / "render_manifest.json"
DEFAULT_API_BASE_URL = os.getenv("COLAB_API_BASE_URL", "").rstrip("/")
_render_manifest: Optional[dict] = None


def _slug_to_title(slug: str) -> str:
    """Convert file stem to a readable title."""
    return slug.replace("_", " ").strip().title()


def _get_video_metrics(path: str) -> tuple[float, int]:
    """Return fps and frame count for a video, falling back to sensible defaults."""
    default_fps = 30.0
    default_frames = int(default_fps * 60)  # Assume 1 minute if unknown
    fps = default_fps
    frames = default_frames
    cap = None
    try:
        cap = cv2.VideoCapture(path)
        if cap.isOpened():
            fps_read = cap.get(cv2.CAP_PROP_FPS)
            if fps_read and fps_read > 0:
                fps = float(fps_read)
            frames_read = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            if frames_read and frames_read > 0:
                frames = int(frames_read)
    except Exception as err:
        print(f"Video metric read failed for {path}: {err}")
    finally:
        if cap is not None:
            cap.release()
    if frames <= 0:
        frames = default_frames
    if fps <= 0:
        fps = default_fps
    return fps, max(frames, 1)


def _resolve_npz_folder(video_id: str) -> Optional[str]:
    """Locate NPZ parameter folder for a given video."""
    candidate = Path(PARAMETERS_DIR) / f"{video_id}_parameters"
    if candidate.exists():
        return candidate.name
    if not Path(PARAMETERS_DIR).exists():
        return None
    for folder in Path(PARAMETERS_DIR).iterdir():
        if folder.is_dir() and folder.name.startswith(video_id):
            return folder.name
    return None


def refresh_video_library(force: bool = False) -> dict[str, dict]:
    """Discover available source videos and cache metadata."""
    global _video_library_last_scan
    now = time.time()
    with _video_library_lock:
        if (
            not force
            and VIDEO_INFO
            and (now - _video_library_last_scan) < _video_library_refresh_interval
        ):
            return VIDEO_INFO

        VIDEO_FOLDER_MAPPING.clear()
        VIDEO_INFO.clear()

        video_dir = Path(VIDEOS_DIR)
        if not video_dir.exists():
            _video_library_last_scan = now
            return VIDEO_INFO

        supported_ext = {".mp4", ".mov", ".m4v", ".avi", ".mkv"}

        for file_path in sorted(video_dir.iterdir()):
            if not file_path.is_file():
                continue
            if file_path.suffix.lower() not in supported_ext:
                continue

            video_id = file_path.stem
            fps, frames = _get_video_metrics(str(file_path))
            npz_folder = _resolve_npz_folder(video_id)
            npz_exists = npz_folder is not None
            if npz_exists:
                VIDEO_FOLDER_MAPPING[video_id] = npz_folder

            size_mb = None
            try:
                size_mb = round(file_path.stat().st_size / (1024 * 1024), 2)
            except Exception:
                pass

            VIDEO_INFO[video_id] = {
                "video_id": video_id,
                "title": _slug_to_title(video_id),
                "filename": file_path.name,
                "original_path": str(file_path),
                "original_size_mb": size_mb,
                "fps": fps,
                "frames": frames,
                "npz_folder": npz_folder,
                "npz_exists": npz_exists,
                "rendered_path": str(Path(RENDERED_DIR) / f"{video_id}_rendered.mp4"),
            }

        _video_library_last_scan = now
        return VIDEO_INFO


def get_video_info(video_id: str, force_refresh: bool = False) -> Optional[dict]:
    """Return cached metadata for a specific video."""
    info = refresh_video_library(force=force_refresh).get(video_id)
    if not info and not force_refresh:
        info = refresh_video_library(force=True).get(video_id)
    return info


def _get_active_render_job():
    """Return the currently running render job if any."""
    for vid, data in render_jobs.items():
        if data.get("status") in {"initializing", "rendering", "cancelling"}:
            return vid, data
    return None, None


def _save_user_credentials(creds):
    """Persist user OAuth credentials for reuse."""
    if not creds:
        return
    # Service account credentials do not need caching
    if isinstance(creds, service_account.Credentials):
        return
    if not hasattr(creds, "to_json"):
        return
    try:
        data = json.loads(creds.to_json())
        if not data.get("refresh_token"):
            # Nothing reusable
            return
        CREDENTIAL_CACHE_PATH.write_text(json.dumps(data))
    except Exception as err:
        print(f"Credential cache write failed: {err}")


def _load_cached_user_credentials():
    """Load cached OAuth credentials if available."""
    if not CREDENTIAL_CACHE_PATH.exists():
        return None
    try:
        data = json.loads(CREDENTIAL_CACHE_PATH.read_text())
        creds = UserCredentials.from_authorized_user_info(data, scopes=DRIVE_SCOPES)
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        return creds
    except Exception as err:
        print(f"Cached credential load failed: {err}")
        return None


def get_drive_credentials(force_refresh: bool = False):
    """Resolve Drive credentials using service accounts, cached tokens, or Colab auth."""
    global _drive_credentials, _drive_auth_disabled, _drive_auth_error
    if _drive_credentials is not None and not force_refresh:
        return _drive_credentials
    if _drive_auth_disabled and not force_refresh:
        return None

    # Service account from file path
    sa_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if sa_path and os.path.exists(sa_path):
        try:
            _drive_credentials = service_account.Credentials.from_service_account_file(
                sa_path, scopes=DRIVE_SCOPES
            )
            return _drive_credentials
        except Exception as err:
            print(f"Service account file credentials failed: {err}")

    # Service account from JSON env var
    sa_json = os.getenv("SERVICE_ACCOUNT_JSON")
    if sa_json:
        try:
            info = json.loads(sa_json)
            _drive_credentials = service_account.Credentials.from_service_account_info(
                info, scopes=DRIVE_SCOPES
            )
            return _drive_credentials
        except Exception as err:
            print(f"Service account JSON credentials failed: {err}")

    # Cached user credentials
    if not force_refresh:
        cached = _load_cached_user_credentials()
        if cached:
            _drive_credentials = cached
            return _drive_credentials

    # Default application credentials (may work if user already authenticated)
    try:
        creds, _ = default(scopes=DRIVE_SCOPES)
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        if creds:
            _drive_credentials = creds
            _save_user_credentials(creds)
            return _drive_credentials
    except (DefaultCredentialsError, RefreshError) as err:
        msg = f"Default credential lookup failed: {err}"
        if _drive_auth_error != msg:
            print(msg)
        _drive_auth_error = msg
    except Exception as err:
        msg = f"Default credential lookup failed: {err}"
        if _drive_auth_error != msg:
            print(msg)
        _drive_auth_error = msg

    # Interactive Colab authentication fallback
    try:
        from google.colab import auth as colab_auth

        colab_auth.authenticate_user()
        creds, _ = default(scopes=DRIVE_SCOPES)
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        if creds:
            _drive_credentials = creds
            _save_user_credentials(creds)
            return _drive_credentials
    except ImportError:
        pass
    except Exception as err:
        print(f"Colab authentication failed: {err}")

    _drive_auth_disabled = True
    if _drive_auth_error:
        print(f"Drive authentication disabled after failure: {_drive_auth_error}")
    else:
        print("Drive authentication disabled: no credential sources available")
    return None


def get_drive_service(force_refresh: bool = False):
    """Return a singleton Drive service instance, creating it lazily."""
    global _drive_service
    with _drive_service_lock:
        if _drive_service is not None and not force_refresh:
            return _drive_service
        try:
            creds = get_drive_credentials(force_refresh=force_refresh)
            if not creds:
                if not _drive_auth_disabled:
                    print("Drive auth error: No credentials available")
                return None
            service = build('drive', 'v3', credentials=creds, cache_discovery=False)
            _drive_service = service
            return service
        except Exception as build_err:
            print(f"Drive service init failed: {build_err}")
        return None


def _load_cached_folder_id():
    """Load cached rendered_videos folder id if present."""
    global _rendered_videos_folder_id
    if _rendered_videos_folder_id:
        return _rendered_videos_folder_id
    if FOLDER_CACHE_PATH.exists():
        try:
            data = json.loads(FOLDER_CACHE_PATH.read_text())
            folder_id = data.get("folder_id")
            if folder_id:
                _rendered_videos_folder_id = folder_id
                return folder_id
        except Exception as err:
            print(f"Folder cache read failed: {err}")
    return None


def _persist_folder_id(folder_id: str):
    """Persist rendered_videos folder id to cache."""
    if not folder_id:
        return
    try:
        FOLDER_CACHE_PATH.write_text(json.dumps({"folder_id": folder_id}))
    except Exception as err:
        print(f"Folder cache write failed: {err}")


def _load_manifest():
    """Return manifest dictionary, loading it once from disk."""
    global _render_manifest
    if _render_manifest is not None:
        return _render_manifest
    if MANIFEST_PATH.exists():
        try:
            _render_manifest = json.loads(MANIFEST_PATH.read_text())
        except Exception as err:
            print(f"Manifest load failed: {err}")
            _render_manifest = {}
    else:
        _render_manifest = {}
    return _render_manifest


def _save_manifest():
    """Persist manifest to disk."""
    if _render_manifest is None:
        return
    try:
        MANIFEST_PATH.write_text(json.dumps(_render_manifest, indent=2))
    except Exception as err:
        print(f"Manifest write failed: {err}")


def _update_manifest(video_id: str, info: Optional[dict]):
    """Update in-memory + on-disk manifest entry."""
    payload = {
        "video_id": video_id,
        "status": (info.get("status") if info and info.get("status") else "ready") if info else "missing",
        "source": info.get("source") if info else None,
        "file_id": info.get("file_id") if info else None,
        "size_mb": info.get("size_mb") if info else None,
        "local_path": info.get("local_path") if info else None,
        "drive_path": info.get("drive_path") if info else None,
        "video_path": info.get("video_path") if info else None,
        "embed_path": info.get("embed_path") if info else (info.get("video_path") if info else None),
        "file_path": info.get("file_path") if info else None,
        "drive_preview_url": info.get("drive_preview_url") if info else None,
        "drive_download_url": info.get("drive_download_url") if info else None,
        "already_exists": info.get("already_exists") if info else False,
        "original_path": info.get("original_path") if info else None,
        "original_size_mb": info.get("original_size_mb") if info else None,
        "npz_exists": info.get("npz_exists") if info else False,
        "npz_folder": info.get("npz_folder") if info else None,
        "updated_at": time.time()
    }
    with _manifest_lock:
        manifest = _load_manifest()
        manifest[video_id] = payload
        _save_manifest()


def _make_absolute(path: Optional[str], request: Optional['Request'] = None):
    """Convert relative API path to absolute URL when request/base is available."""
    if not path:
        return None
    if path.startswith("http://") or path.startswith("https://"):
        return path
    base_url = None
    if request is not None:
        base_url = str(request.base_url).rstrip("/")
    if not base_url and DEFAULT_API_BASE_URL:
        base_url = DEFAULT_API_BASE_URL
    if base_url:
        return f"{base_url}{path}"
    return path


def _build_video_paths(video_id: str):
    """Return relative API paths for rendered video access."""
    return {
        "video_path": f"/rendered/{video_id}",
        "embed_path": f"/rendered/{video_id}",
        "file_path": f"/rendered_file/{video_id}",
    }


def project_uv(verts_cam, fx, fy, cx, cy):
    """Project 3D vertices to 2D image coordinates"""
    X, Y, Z = verts_cam[:,0], verts_cam[:,1], verts_cam[:,2]
    u = fx * (X / Z) + cx
    v = fy * (Y / Z) + cy
    return np.stack([u, v], axis=1)

def render_frame(video_id, frame_num):
    """Render a single frame using the exact working script logic"""
    try:
        video_info = get_video_info(video_id)
        if not video_info:
            return None, "Video not registered"

        folder = video_info.get("npz_folder")
        if not folder:
            folder = _resolve_npz_folder(video_id)
            if folder:
                VIDEO_FOLDER_MAPPING[video_id] = folder
                video_info["npz_folder"] = folder
                video_info["npz_exists"] = True
        if not folder:
            return None, "NPZ folder not found"

        npz_path = os.path.join(PARAMETERS_DIR, folder, f"frame_{frame_num:04d}_params.npz")

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

def get_or_create_rendered_videos_folder(service: Optional[object] = None):
    """Find or create the rendered_videos folder in Google Drive under ecolang."""
    try:
        global _rendered_videos_folder_id

        cached_id = _load_cached_folder_id()
        drive_service = service or get_drive_service()

        if cached_id and drive_service:
            try:
                drive_service.files().get(fileId=cached_id, fields='id').execute()
                _rendered_videos_folder_id = cached_id
                return cached_id
            except Exception:
                print("Cached folder id invalid - refreshing")
                _rendered_videos_folder_id = None
                try:
                    FOLDER_CACHE_PATH.unlink()
                except Exception:
                    pass

        if not drive_service:
            global _drive_service_warned
            if not _drive_service_warned:
                print("Drive service unavailable - cannot resolve rendered_videos folder")
                _drive_service_warned = True
            return None

        # First, find the ecolang folder
        ecolang_query = "name='ecolang' and mimeType='application/vnd.google-apps.folder' and trashed=false"
        ecolang_results = drive_service.files().list(
            q=ecolang_query,
            fields='files(id, name)',
            pageSize=10
        ).execute()

        ecolang_folders = ecolang_results.get('files', [])

        if not ecolang_folders:
            print("Warning: 'ecolang' folder not found in Google Drive")
            # Search for rendered_videos folder anywhere
            query = "name='rendered_videos' and mimeType='application/vnd.google-apps.folder' and trashed=false"
            results = drive_service.files().list(
                q=query,
                fields='files(id, name)',
                pageSize=10
            ).execute()

            folders = results.get('files', [])
            if folders:
                folder_id = folders[0]['id']
                print(f"Found 'rendered_videos' folder (ID: {folder_id})")
                _rendered_videos_folder_id = folder_id
                _persist_folder_id(folder_id)
                return _rendered_videos_folder_id

            # Create rendered_videos in root if ecolang doesn't exist
            file_metadata = {
                'name': 'rendered_videos',
                'mimeType': 'application/vnd.google-apps.folder'
            }
            folder = drive_service.files().create(body=file_metadata, fields='id').execute()
            folder_id = folder.get('id')
            print(f"Created 'rendered_videos' folder in root (ID: {folder_id})")
            _rendered_videos_folder_id = folder_id
            _persist_folder_id(folder_id)
            return _rendered_videos_folder_id

        ecolang_id = ecolang_folders[0]['id']
        print(f"Found 'ecolang' folder (ID: {ecolang_id})")

        # Now search for rendered_videos folder inside ecolang
        query = f"name='rendered_videos' and '{ecolang_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
        results = drive_service.files().list(
            q=query,
            fields='files(id, name)',
            pageSize=10
        ).execute()

        folders = results.get('files', [])

        if folders:
            folder_id = folders[0]['id']
            print(f"Found 'rendered_videos' folder (ID: {folder_id})")
            _rendered_videos_folder_id = folder_id
            _persist_folder_id(folder_id)
            return _rendered_videos_folder_id

        # Create the folder if it doesn't exist
        file_metadata = {
            'name': 'rendered_videos',
            'mimeType': 'application/vnd.google-apps.folder',
            'parents': [ecolang_id]
        }
        folder = drive_service.files().create(body=file_metadata, fields='id').execute()
        folder_id = folder.get('id')
        print(f"Created 'rendered_videos' folder (ID: {folder_id})")
        _rendered_videos_folder_id = folder_id
        _persist_folder_id(folder_id)
        return _rendered_videos_folder_id

    except HttpError as http_err:
        print(f"Drive API error while locating folder: {http_err}")
    except Exception as e:
        print(f"Error getting/creating folder: {e}")
        return None

def upload_to_drive_auto(file_path, file_name, parent_folder_id=None, service: Optional[object] = None):
    """Upload file to Google Drive with automatic public sharing."""
    try:
        drive_service = service or get_drive_service()
        if not drive_service:
            print("Drive service unavailable - skipping upload")
            return None

        file_metadata = {'name': file_name, 'mimeType': 'video/mp4'}

        # Add parent folder if specified
        if parent_folder_id:
            file_metadata['parents'] = [parent_folder_id]
            print(f"Uploading to folder ID: {parent_folder_id}")

        media = MediaFileUpload(os.fspath(file_path), mimetype='video/mp4', resumable=True)

        file = drive_service.files().create(
            body=file_metadata,
            media_body=media,
            fields='id, name, size, webViewLink'
        ).execute()

        file_id = file.get('id')

        drive_service.permissions().create(
            fileId=file_id,
            body={'type': 'anyone', 'role': 'reader'}
        ).execute()

        size_mb = int(file.get('size', 0)) / (1024 * 1024)

        return {
            'file_id': file_id,
            'preview_url': f"https://drive.google.com/file/d/{file_id}/preview",
            'embed_url': f"https://drive.google.com/file/d/{file_id}/preview",
            'drive_preview_url': f"https://drive.google.com/file/d/{file_id}/preview",
            'drive_download_url': f"https://drive.google.com/uc?id={file_id}&export=download",
            'size_mb': size_mb
        }
    except HttpError as http_err:
        print(f"Upload error (HTTP): {http_err}")
    except Exception as e:
        print(f"Upload error: {e}")
        return None

def check_existing_rendered_video(video_id, video_info: Optional[dict] = None):
    """Check if a rendered video already exists in Google Drive."""
    try:
        video_info = video_info or get_video_info(video_id)
        if not video_info:
            _update_manifest(video_id, None)
            return None

        output_dir = Path(RENDERED_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)
        file_path = output_dir / f"{video_id}_rendered.mp4"
        file_path_fs = os.fspath(file_path)

        drive_service = get_drive_service()
        folder_id = None

        if drive_service:
            folder_id = get_or_create_rendered_videos_folder(service=drive_service)
            if not folder_id:
                if not _drive_service_warned:
                    print("Drive folder lookup failed - continuing with local fallback")
                    _drive_service_warned = True
        else:
            if not _drive_service_warned:
                print("Drive service unavailable - skipping remote Drive lookup")
                _drive_service_warned = True

        if drive_service and folder_id:
            query = f"name='{video_id}_rendered.mp4' and '{folder_id}' in parents and trashed=false"
            print(f"Searching Drive with query: {query}")

            results = drive_service.files().list(
                q=query,
                fields='files(id, name, size, webViewLink)',
                pageSize=10
            ).execute()

            files = results.get('files', [])
            print(f"Found {len(files)} file(s) matching '{video_id}_rendered.mp4' in rendered_videos folder")

            if files:
                file = files[0]
                file_id = file.get('id')

                try:
                    drive_service.permissions().create(
                        fileId=file_id,
                        body={'type': 'anyone', 'role': 'reader'}
                    ).execute()
                    print(f"Set public permissions for file {file_id}")
                except Exception as perm_error:
                    print(f"Permission already exists or error: {perm_error}")

                size_mb = int(file.get('size', 0)) / (1024 * 1024)

                print(f"[drive] Found existing rendered video in Drive: {video_id}_rendered.mp4 (File ID: {file_id}, Size: {size_mb:.1f} MB)")

                info = {
                    'file_id': file_id,
                    'drive_preview_url': f"https://drive.google.com/file/d/{file_id}/preview",
                    'drive_download_url': f"https://drive.google.com/uc?id={file_id}&export=download",
                    'size_mb': size_mb,
                    'drive_path': file_path_fs,
                    'local_path': file_path_fs,
                    'already_exists': True,
                    'source': 'drive',
                    'video_id': video_id,
                    'original_path': video_info.get('original_path'),
                    'original_size_mb': video_info.get('original_size_mb'),
                    'npz_exists': video_info.get('npz_exists'),
                    'npz_folder': video_info.get('npz_folder'),
                    **_build_video_paths(video_id)
                }
                _update_manifest(video_id, info)
                return info

        if os.path.exists(file_path_fs):
            size_mb = os.path.getsize(file_path_fs) / (1024 * 1024)

            if drive_service and folder_id:
                print(f"Found video locally but not in Drive - uploading: {video_id}_rendered.mp4")
                upload_result = upload_to_drive_auto(
                    file_path_fs,
                    f"{video_id}_rendered.mp4",
                    parent_folder_id=folder_id,
                    service=drive_service
                )
                if upload_result:
                    upload_result['drive_path'] = file_path_fs
                    upload_result['local_path'] = file_path_fs
                    upload_result['already_exists'] = True
                    upload_result['source'] = 'upload'
                    if 'drive_preview_url' not in upload_result:
                        upload_result['drive_preview_url'] = upload_result.get('preview_url')
                    upload_result.update(_build_video_paths(video_id))
                    upload_result['drive_download_url'] = f"https://drive.google.com/uc?id={upload_result['file_id']}&export=download" if upload_result.get('file_id') else None
                    print("Successfully uploaded existing local file to Drive")
                    upload_result['video_id'] = video_id
                    upload_result['original_path'] = video_info.get('original_path')
                    upload_result['original_size_mb'] = video_info.get('original_size_mb')
                    upload_result['npz_exists'] = video_info.get('npz_exists')
                    upload_result['npz_folder'] = video_info.get('npz_folder')
                    _update_manifest(video_id, upload_result)
                    return upload_result

            print(f"[local] Found existing rendered video locally: {file_path_fs} ({size_mb:.1f} MB)")
            info = {
                'file_id': None,
                'drive_preview_url': None,
                'drive_download_url': None,
                'size_mb': size_mb,
                'drive_path': file_path_fs,
                'already_exists': True,
                'local_only': True,
                'local_path': file_path_fs,
                'source': 'local',
                'video_id': video_id,
                'original_path': video_info.get('original_path'),
                'original_size_mb': video_info.get('original_size_mb'),
                'npz_exists': video_info.get('npz_exists'),
                'npz_folder': video_info.get('npz_folder'),
                **_build_video_paths(video_id)
            }
            _update_manifest(video_id, info)
            return info

        print(f"[drive] No existing rendered video found for: {video_id}")
        _update_manifest(video_id, None)
        return None

    except HttpError as http_err:
        print(f"Drive API error while checking video: {http_err}")
    except Exception as e:
        print(f"Error checking existing video: {e}")
        import traceback
        traceback.print_exc()
    _update_manifest(video_id, None)
    return None

print("Rendering functions loaded!")

# ============= CELL 6: FastAPI =============
from fastapi import FastAPI, BackgroundTasks, Request
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from pydantic import BaseModel
app = FastAPI(title="ECOLANG Rendering API")
render_jobs = {}

class RenderRequest(BaseModel):
    video_id: str
    force: bool = False

@app.get("/")
async def root():
    return {"message": "ECOLANG Rendering API", "status": "running"}

@app.get("/health")
async def health_check():
    return {"status": "ok", "message": "Colab API is running", "device": str(device), "model_loaded": True}


@app.get("/render_manifest")
async def render_manifest(request: Request, refresh: bool = False):
    """Provide readiness metadata for all known videos."""
    video_map = refresh_video_library(force=refresh)
    if refresh:
        for vid in list(video_map.keys()):
            try:
                check_existing_rendered_video(vid, video_map.get(vid))
            except Exception as err:
                print(f"Manifest refresh for {vid} failed: {err}")

    with _manifest_lock:
        manifest_data = json.loads(json.dumps(_load_manifest()))

    response = {}
    for vid, meta in video_map.items():
        entry = manifest_data.get(vid, {}).copy() if manifest_data.get(vid) else {}
        if not entry.get("video_path"):
            entry.update(_build_video_paths(vid))
        entry.setdefault("status", "missing")
        entry.setdefault("video_id", vid)
        entry.setdefault("title", meta.get("title"))
        entry.setdefault("fps", meta.get("fps"))
        entry.setdefault("frames", meta.get("frames"))
        entry.setdefault("original_path", meta.get("original_path"))
        entry.setdefault("original_size_mb", meta.get("original_size_mb"))
        entry.setdefault("npz_exists", meta.get("npz_exists"))
        entry.setdefault("npz_folder", meta.get("npz_folder"))
        entry["video_url"] = _make_absolute(entry.get("file_path"), request)
        entry["file_url"] = _make_absolute(entry.get("file_path"), request)
        entry["embed_url"] = _make_absolute(entry.get("embed_path") or entry.get("video_path"), request)
        entry["original_url"] = _make_absolute(f"/original_file/{vid}", request)
        entry["rendered_url"] = entry["video_url"]
        entry["original_endpoint"] = f"/original_file/{vid}"
        entry["rendered_endpoint"] = entry.get("file_path") or f"/rendered_file/{vid}"
        entry["rendered_exists"] = entry.get("status") == "ready" or bool(entry.get("file_path"))
        response[vid] = entry

    active_job_id, active_job = _get_active_render_job()
    if active_job:
        response.setdefault(active_job_id, {}).setdefault("render_job_status", active_job.get("status"))

    return {
        "videos": response,
        "base_url": str(request.base_url).rstrip("/"),
        "active_job": active_job_id
    }


@app.get("/video_library")
async def video_library(request: Request, refresh: bool = False):
    """Return an array of videos with original and rendered status metadata."""
    manifest = await render_manifest(request, refresh=refresh)
    videos = manifest.get("videos", {})
    items = []
    for vid, data in videos.items():
        entry = data.copy()
        entry.setdefault("video_id", vid)
        items.append(entry)
    return {
        "videos": items,
        "base_url": manifest.get("base_url"),
        "active_job": manifest.get("active_job")
    }

def render_video_background(video_id: str, cancel_event: threading.Event):
    job = render_jobs.get(video_id, {})
    job.update({
        "status": "initializing",
        "current": 0,
        "total": 0,
        "start_time": time.time(),
        "cancel_event": cancel_event,
        "video_id": video_id
    })
    render_jobs[video_id] = job

    try:
        video_info = get_video_info(video_id)
        if not video_info:
            raise RuntimeError(f"Video {video_id} not registered")

        total_frames = max(int(video_info.get('frames', 0)), 1)
        fps = float(video_info.get('fps', 30))

        output_dir = Path(RENDERED_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)
        temp_path = output_dir / f"{video_id}_temp.mp4"
        final_path = output_dir / f"{video_id}_rendered.mp4"
        job["output_path"] = str(final_path)

        first_img, _ = render_frame(video_id, 1)
        if first_img is not None:
            output_h, output_w = first_img.shape[:2]
        else:
            output_h, output_w = 720, 1280

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(str(temp_path), fourcc, fps, (output_w, output_h))

        job["status"] = "rendering"
        job["total"] = total_frames
        last_valid_frame = None

        for frame_num in range(1, total_frames + 1):
            if cancel_event.is_set():
                print(f"[{video_id}] Cancellation requested at frame {frame_num}")
                job["status"] = "cancelled"
                job["current"] = frame_num - 1
                break

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

            job["current"] = frame_num

            if frame_num % 50 == 0:
                elapsed = time.time() - job["start_time"]
                fps_rate = frame_num / elapsed if elapsed > 0 else 0
                eta = (total_frames - frame_num) / fps_rate if fps_rate > 0 else 0
                print(f"[{video_id}] {frame_num}/{total_frames} | {fps_rate:.1f} fps | ETA: {eta:.0f}s")

        video_writer.release()

        if cancel_event.is_set():
            print(f"[{video_id}] Render cancelled, cleaning up temporary files")
            try:
                if temp_path.exists():
                    temp_path.unlink()
            except Exception as cleanup_err:
                print(f"Temp cleanup failed: {cleanup_err}")
            job["status"] = "cancelled"
            job["error"] = "cancelled"
            return

        print(f"[{video_id}] Converting to web-compatible format...")
        convert_to_web_compatible(str(temp_path), str(final_path))
        try:
            temp_path.unlink()
        except FileNotFoundError:
            pass

        if cancel_event.is_set():
            print(f"[{video_id}] Cancellation detected post-conversion, removing output")
            try:
                if final_path.exists():
                    final_path.unlink()
            except Exception as cleanup_err:
                print(f"Final cleanup failed: {cleanup_err}")
            job["status"] = "cancelled"
            job["error"] = "cancelled"
            return

        print(f"[{video_id}] Uploading to Google Drive...")
        drive_service = get_drive_service()
        folder_id = None
        if drive_service:
            folder_id = get_or_create_rendered_videos_folder(service=drive_service)
        else:
            print("Drive service unavailable during upload")

        upload_result = None
        if folder_id:
            upload_result = upload_to_drive_auto(
                str(final_path),
                f"{video_id}_rendered.mp4",
                parent_folder_id=folder_id,
                service=drive_service
            )
        else:
            print("Skipping Drive upload - missing folder id")

        paths = _build_video_paths(video_id)
        local_size = final_path.stat().st_size / (1024 * 1024) if final_path.exists() else None

        if upload_result:
            upload_result.setdefault('video_path', paths['video_path'])
            upload_result.setdefault('embed_path', paths['embed_path'])
            upload_result.setdefault('file_path', paths['file_path'])
            upload_result['local_path'] = str(final_path)
            upload_result.setdefault('drive_path', str(final_path))
            upload_result.setdefault('source', 'drive')
            upload_result['video_id'] = video_id
            upload_result['original_path'] = video_info.get('original_path')
            upload_result['original_size_mb'] = video_info.get('original_size_mb')
            upload_result['npz_exists'] = video_info.get('npz_exists')
            upload_result['npz_folder'] = video_info.get('npz_folder')
            _update_manifest(video_id, upload_result)

            job.update({
                "status": "complete",
                "video_url": upload_result['file_path'],
                "file_url": upload_result['file_path'],
                "embed_url": upload_result.get('embed_path', upload_result.get('video_path')),
                "file_path": upload_result['file_path'],
                "video_path": upload_result.get('embed_path', upload_result.get('video_path')),
                "file_id": upload_result['file_id'],
                "size_mb": upload_result['size_mb'],
                "drive_path": upload_result.get('drive_path', str(final_path)),
                "drive_preview_url": upload_result.get('drive_preview_url'),
                "drive_download_url": upload_result.get('drive_download_url'),
                "local_path": str(final_path),
                "local_only": False,
                "source": upload_result.get("source", "drive"),
            })

            total_time = time.time() - job["start_time"]
            print(f"[{video_id}] Complete: {final_path} in {total_time:.1f}s")
            print(f"[{video_id}] Uploaded: {upload_result['file_id']} ({upload_result['size_mb']:.1f} MB)")
            print(f"[{video_id}] Preview URL: {upload_result.get('drive_preview_url') or upload_result.get('preview_url')}")
        else:
            info = {
                'file_id': None,
                'drive_preview_url': None,
                'drive_download_url': None,
                'size_mb': local_size,
                'drive_path': str(final_path),
                'already_exists': True,
                'local_only': True,
                'local_path': str(final_path),
                'source': 'local',
                'video_id': video_id,
                'original_path': video_info.get('original_path'),
                'original_size_mb': video_info.get('original_size_mb'),
                'npz_exists': video_info.get('npz_exists'),
                'npz_folder': video_info.get('npz_folder'),
                **paths
            }
            job.update({
                "status": "complete",
                "video_url": paths['file_path'],
                "file_url": paths['file_path'],
                "embed_url": paths['video_path'],
                "video_path": paths['video_path'],
                "file_path": paths['file_path'],
                "local_path": str(final_path),
                "drive_preview_url": None,
                "drive_download_url": None,
                "local_only": True,
                "source": "local",
                "size_mb": local_size,
            })
            _update_manifest(video_id, info)
            print(f"[{video_id}] Upload failed - video saved locally at {final_path}")

    except Exception as e:
        print(f"Error rendering {video_id}: {e}")
        import traceback
        traceback.print_exc()

        job = render_jobs.get(video_id)
        if not job:
            render_jobs[video_id] = {"status": "error", "current": 0, "total": 0, "error": str(e)}
        else:
            job["status"] = "error"
            job["error"] = str(e)

@app.post("/render_video")
async def start_render(payload: RenderRequest, background_tasks: BackgroundTasks, request: Request):
    video_id = payload.video_id
    video_info = get_video_info(video_id)
    if not video_info:
        return JSONResponse(status_code=404, content={"success": False, "error": f"Video {video_id} not found"})

    # Check if video already exists before rendering
    existing_video = check_existing_rendered_video(video_id, video_info)
    if existing_video:
        stream_path = existing_video.get('file_path')
        embed_path = existing_video.get('embed_path') or existing_video.get('video_path')
        preview_url = existing_video.get('drive_preview_url') or existing_video.get('preview_url')
        download_url = existing_video.get('drive_download_url')
        file_id = existing_video.get('file_id')
        size_mb = existing_video.get('size_mb')
        is_local_only = existing_video.get('local_only', False)
        local_path = existing_video.get('local_path')
        source = existing_video.get('source')

        stream_url_abs = _make_absolute(stream_path, request)
        embed_url_abs = _make_absolute(embed_path, request)

        render_jobs[video_id] = {
            "status": "complete",
            "current": video_info.get("frames", 0),
            "total": video_info.get("frames", 0),
            "video_url": stream_path,
            "file_url": stream_path,
            "embed_url": embed_path,
            "video_path": embed_path,
            "file_path": stream_path,
            "drive_preview_url": preview_url,
            "drive_download_url": download_url,
            "file_id": file_id,
            "size_mb": size_mb,
            "drive_path": existing_video.get('drive_path'),
            "already_exists": True,
            "local_only": is_local_only,
            "local_path": local_path,
            "source": source,
            "npz_exists": video_info.get("npz_exists"),
            "npz_folder": video_info.get("npz_folder"),
            "original_path": video_info.get("original_path"),
            "original_size_mb": video_info.get("original_size_mb"),
        }
        _update_manifest(video_id, existing_video)
        return {
            "success": True,
            "message": "Rendered video already exists",
            "job_id": video_id,
            "already_exists": True,
            "video_url": stream_url_abs or preview_url,
            "file_url": stream_url_abs,
            "embed_url": embed_url_abs,
            "video_path": embed_path,
            "file_path": stream_path,
            "preview_url": preview_url,
            "download_url": download_url,
            "file_id": file_id,
            "size_mb": size_mb,
            "local_only": is_local_only,
            "local_path": local_path,
            "source": source
        }

    if not video_info.get("npz_exists"):
        return JSONResponse(
            status_code=400,
            content={"success": False, "error": "npz_missing", "video_id": video_id}
        )

    active_video_id, active_job = _get_active_render_job()
    if active_video_id and active_video_id != video_id:
        if not payload.force:
            return JSONResponse(
                status_code=409,
                content={
                    "success": False,
                    "error": "render_in_progress",
                    "active_job": active_video_id
                }
            )
        cancel_event = active_job.get("cancel_event")
        if cancel_event and not cancel_event.is_set():
            cancel_event.set()
            active_job["status"] = "cancelling"
        wait_started = time.time()
        while True:
            status = render_jobs.get(active_video_id, {}).get("status")
            if status not in {"initializing", "rendering", "cancelling"}:
                break
            if time.time() - wait_started > 60:
                break
            await asyncio.sleep(0.25)

    active_video_id, active_job = _get_active_render_job()
    if active_video_id and active_video_id != video_id:
        return JSONResponse(
            status_code=423,
            content={
                "success": False,
                "error": "unable_to_cancel",
                "active_job": active_video_id
            }
        )

    if video_id in render_jobs and render_jobs[video_id].get("status") in {"rendering", "initializing"}:
        return {"success": True, "message": "Already rendering", "job_id": video_id}

    cancel_event = threading.Event()
    render_jobs[video_id] = {
        "status": "initializing",
        "current": 0,
        "total": video_info.get("frames", 0),
        "start_time": time.time(),
        "cancel_event": cancel_event,
        "video_id": video_id,
        "npz_exists": video_info.get("npz_exists"),
        "npz_folder": video_info.get("npz_folder"),
        "original_path": video_info.get("original_path"),
        "original_size_mb": video_info.get("original_size_mb"),
    }

    background_tasks.add_task(render_video_background, video_id, cancel_event)

    return {
        "success": True,
        "message": "Rendering started",
        "job_id": video_id,
        "total_frames": video_info.get("frames", 0),
        "forced": payload.force
    }

@app.get("/render_progress/{video_id}")
async def get_progress(video_id: str, request: Request):
    if video_id not in render_jobs:
        return {"status": "not_started", "current": 0, "total": 0}
    job = render_jobs[video_id]
    response = {"status": job["status"], "current": job["current"], "total": job["total"]}
    response["npz_exists"] = job.get("npz_exists")
    response["npz_folder"] = job.get("npz_folder")
    response["original_path"] = job.get("original_path")
    response["original_size_mb"] = job.get("original_size_mb")
    if job["status"] == "complete":
        stream_path = job.get("file_path") or job.get("file_url") or job.get("video_url")
        embed_path = job.get("embed_url") or job.get("video_path")
        response["video_path"] = embed_path
        response["file_path"] = stream_path
        response["video_url"] = _make_absolute(stream_path, request)
        response["file_url"] = _make_absolute(stream_path, request)
        response["embed_url"] = _make_absolute(embed_path, request)
        response["file_id"] = job.get("file_id")
        response["size_mb"] = job.get("size_mb", 0)
        response["drive_path"] = job.get("drive_path")
        response["local_path"] = job.get("local_path")
        response["local_only"] = job.get("local_only", False)
        response["drive_preview_url"] = job.get("drive_preview_url")
        response["drive_download_url"] = job.get("drive_download_url")
        response["source"] = job.get("source")
    if job["status"] == "error":
        response["error"] = job.get("error", "Unknown error")
    if job["status"] == "cancelled":
        response["error"] = job.get("error", "cancelled")
    return response

# Serve original videos directly
@app.get("/original_file/{video_id}")
async def get_original_file(video_id: str):
    video_info = get_video_info(video_id)
    if not video_info:
        return JSONResponse(status_code=404, content={"error": "video_not_found"})
    path = video_info.get("original_path")
    if not path or not os.path.exists(path):
        return JSONResponse(status_code=404, content={"error": "file_not_found"})
    filename = video_info.get("filename") or f"{video_id}.mp4"
    return FileResponse(path, media_type="video/mp4", filename=filename)

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
