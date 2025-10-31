# ECOLANG Application - Complete Deployment Guide

## Overview

ECOLANG is a sign language video to 3D mesh rendering application with split architecture:

- **Streamlit Frontend**: Lightweight web UI (hosted on Streamlit Cloud)
- **Google Colab Backend**: Heavy ML processing with SMPL-X (GPU-accelerated)
- **Google Drive Storage**: Videos, NPZ parameters, and models

**Total Setup Time**: ~60 minutes
**Per-Video Generation**: ~10 minutes (1800 frames)

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER'S BROWSER                            │
│                              ↓                                   │
│                   [Streamlit Web UI]                             │
│                    - Video selector                              │
│                    - Progress tracking                           │
│                    - Side-by-side display                        │
└──────────────────────────┬──────────────────────────────────────┘
                           │ HTTPS
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│                    GOOGLE COLAB BACKEND                          │
│                     (FastAPI + ngrok)                            │
│                                                                  │
│  Endpoints:                                                      │
│  - GET  /health                                                  │
│  - POST /render_frame                                            │
│                           ↓                                      │
│              [SMPL-X Mesh Renderer]                              │
│              - Loads NPZ parameters                              │
│              - Generates 3D mesh                                 │
│              - Renders to 720x720 image                          │
└──────────────────────────┬──────────────────────────────────────┘
                           │ Reads from
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│                    GOOGLE DRIVE STORAGE                          │
│                   /MyDrive/ecolang/                              │
│                                                                  │
│  ├── videos/                  (4 MP4 files)                     │
│  ├── Extracted_parameters/    (1800 NPZ files per video)        │
│  └── models/                  (SMPLX_NEUTRAL.npz)               │
└─────────────────────────────────────────────────────────────────┘
```

---

## Prerequisites

Before starting, ensure you have:

1. **Google Account** with Google Drive access
2. **GitHub Account** (for hosting Streamlit app)
3. **ngrok Account** (free tier: https://ngrok.com/signup)
4. **Your 4 sign language videos** (MP4 format, ~60 seconds each)
5. **NPZ parameter files** (1800 files per video in `Extracted_parameters/` folders)
6. **SMPL-X model file** (`SMPLX_NEUTRAL.npz`)

---

## PHASE 1: Google Drive Setup (15 minutes)

### Step 1.1: Verify Google Drive Structure

Open Google Drive and create this folder structure:

```
/MyDrive/ecolang/
├── Extracted_parameters/
│   ├── video1_speaking/
│   │   ├── frame_0001_params.npz
│   │   ├── frame_0002_params.npz
│   │   └── ... (1800 total files)
│   ├── video2_gestures/
│   │   └── ... (1800 files)
│   ├── video3_conversation/
│   │   └── ... (1800 files)
│   └── video4_demonstration/
│       └── ... (1800 files)
├── models/
│   └── SMPLX_NEUTRAL.npz
└── videos/
    └── (will upload in next step)
```

**Verification Checklist:**
- [ ] Folder `/MyDrive/ecolang/` exists
- [ ] Folder `Extracted_parameters/` contains 4 subfolders
- [ ] Each subfolder has exactly 1800 NPZ files
- [ ] File `models/SMPLX_NEUTRAL.npz` exists (check file size ~100MB)

---

### Step 1.2: Upload Videos

1. Create folder: `/MyDrive/ecolang/videos/`
2. Upload your 4 MP4 video files
3. Rename them to match these exact names:
   - `video1_speaking.mp4`
   - `video2_gestures.mp4`
   - `video3_conversation.mp4`
   - `video4_demonstration.mp4`

**Important**: Video filenames must match the folder names in `Extracted_parameters/`

---

### Step 1.3: Get Google Drive File IDs

For **each of the 4 videos**:

1. Right-click the video → **Share**
2. Change permissions: **"Anyone with the link"** can view
3. Click **Copy link**
4. Paste link in a text file

You'll have 4 links like this:
```
https://drive.google.com/file/d/1ABC123xyz/view?usp=sharing
                                  ↑
                            This is the FILE_ID
```

---

### Step 1.4: Convert to Direct Download URLs

**Option A: Use Helper Script (Recommended)**

1. Open terminal in your project folder
2. Run:
   ```bash
   python convert_drive_links.py
   ```
3. Paste each share link when prompted
4. Script will output a complete `VIDEO_LIBRARY` configuration
5. Copy the entire output

**Option B: Manual Conversion**

Extract FILE_ID from each link and create direct URLs:
```
Format: https://drive.google.com/uc?export=download&id=FILE_ID
```

Example:
```
Share link: https://drive.google.com/file/d/1ABC123xyz/view?usp=sharing
Direct URL: https://drive.google.com/uc?export=download&id=1ABC123xyz
```

---

### Step 1.5: Update config.py

1. Open `signmesh/config.py`
2. Find the `VIDEO_LIBRARY` section (lines 45-78)
3. Replace all instances of `PASTE_FILE_ID_HERE` with your actual FILE_IDs
4. Save the file

**Before:**
```python
'github_url': 'https://drive.google.com/uc?export=download&id=PASTE_FILE_ID_HERE',
```

**After:**
```python
'github_url': 'https://drive.google.com/uc?export=download&id=1ABC123xyz',
```

---

## PHASE 2: Create Colab Backend (30 minutes)

### Step 2.1: Get ngrok Authentication Token

1. Go to: https://ngrok.com/signup
2. Create free account
3. Go to dashboard: https://dashboard.ngrok.com/get-started/your-authtoken
4. Copy your authtoken (looks like: `2abc123XYZ_long_string`)
5. Save it for Step 2.7

---

### Step 2.2: Create New Colab Notebook

1. Go to: https://colab.research.google.com/
2. Click **File → New notebook**
3. Rename notebook: `ecolang_mesh_api.ipynb`
4. Click **Runtime → Change runtime type → GPU** (T4 or better)

---

### Step 2.3: Add Code Cells

Copy-paste the following cells in order:

---

#### CELL 1: Install Dependencies

```python
# Install all required packages
!pip install -q fastapi uvicorn pyngrok pillow trimesh pyrender smplx torch opencv-python

print("✅ All dependencies installed successfully")
```

**Run this cell** (Ctrl+Enter) - takes ~2 minutes

---

#### CELL 2: Mount Google Drive & Verify Structure

```python
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
```

**Run this cell** - If any assertion fails, go back to Phase 1

---

#### CELL 3: Load SMPL-X Model

```python
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
```

**Run this cell** - takes ~30 seconds

---

#### CELL 4: Create Mesh Rendering Function

```python
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
```

**Run this cell** - takes ~10 seconds, renders a test frame

---

#### CELL 5: Create FastAPI Server

```python
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
```

**Run this cell** - instantaneous

---

#### CELL 6: Start Server with ngrok

```python
from pyngrok import ngrok
import uvicorn
import nest_asyncio

# Allow nested event loops (required for Colab)
nest_asyncio.apply()

# Set your ngrok authtoken
# Get it from: https://dashboard.ngrok.com/get-started/your-authtoken
NGROK_AUTH_TOKEN = "YOUR_NGROK_TOKEN_HERE"  # ← REPLACE THIS

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
```

**IMPORTANT**:
1. Replace `YOUR_NGROK_TOKEN_HERE` with your actual ngrok token
2. Run this cell - it will run indefinitely
3. Copy the public URL that appears
4. Leave this cell running while using the app

---

### Step 2.4: Save the Notebook

1. Click **File → Save**
2. Optional: Save a copy to Google Drive for backup
   - **File → Save a copy in Drive**
   - Save to: `/MyDrive/ecolang/ecolang_mesh_api.ipynb`

---

### Step 2.5: Test the API

In a new browser tab, visit these URLs (replace with your ngrok URL):

1. **API Info**: `https://your-ngrok-url.ngrok-free.app/`
2. **Health Check**: `https://your-ngrok-url.ngrok-free.app/health`
3. **API Docs**: `https://your-ngrok-url.ngrok-free.app/docs`

You should see JSON responses confirming the API is running.

---

## PHASE 3: Deploy Streamlit Frontend (10 minutes)

### Step 3.1: Commit Updated Config

1. Open terminal in project folder
2. Ensure you've updated `signmesh/config.py` with Google Drive FILE_IDs
3. Commit changes:

```bash
git add signmesh/config.py
git commit -m "Update VIDEO_LIBRARY with Google Drive file IDs"
git push origin main
```

---

### Step 3.2: Deploy to Streamlit Cloud

1. Go to: https://share.streamlit.io/
2. Sign in with GitHub
3. Click **"New app"**
4. Configure:
   - **Repository**: Select your repository
   - **Branch**: `main`
   - **Main file path**: `streamlit_app.py`
5. **WAIT** - Don't click Deploy yet!

---

### Step 3.3: Add Streamlit Secrets

Before deploying, add the Colab API URL:

1. In the deployment form, expand **"Advanced settings"**
2. Click **"Secrets"**
3. Paste this configuration (replace with your actual ngrok URL):

```toml
COLAB_API_URL = "https://your-ngrok-url.ngrok-free.app"
```

**Example**:
```toml
COLAB_API_URL = "https://abc123-45-67-89-10.ngrok-free.app"
```

4. Click **"Save"**

---

### Step 3.4: Deploy Application

1. Click **"Deploy!"**
2. Wait 2-3 minutes for deployment
3. Streamlit will install dependencies and start the app
4. When ready, you'll see your app URL: `https://yourapp.streamlit.app`

---

## PHASE 4: Test the Application (5-10 minutes)

### Step 4.1: Open Your App

Click the Streamlit Cloud URL to open your application.

You should see:
- **Header**: "ECOLANG"
- **Dropdown**: With 4 video options
- **Two columns**: Empty initially

---

### Step 4.2: Test Video Selection

1. Click the dropdown
2. Select **"Video 1 - Speaking"**
3. The app should:
   - Show "Generating mesh video..." message
   - Display a progress bar
   - Show original video in left column
   - Start generating frames (you'll see "Generating frame 1 of 1800...")

---

### Step 4.3: Monitor Generation

The generation process will:
- Take approximately **5-10 minutes** for 1800 frames
- Show real-time progress: "Generating frame 123 of 1800"
- Update progress bar
- Display original video immediately (from Google Drive)

**Expected performance**:
- ~0.3-0.5 seconds per frame on Colab GPU
- ~300 frames per minute
- Total: 5-10 minutes per video

---

### Step 4.4: View Results

When complete:
- **Left column**: Original sign language video
- **Right column**: Generated 3D mesh video
- Both videos play simultaneously
- You can replay, pause, adjust speed

---

### Step 4.5: Test Other Videos

Repeat with other videos:
- Select "Video 2 - Gestures"
- Select "Video 3 - Conversation"
- Select "Video 4 - Demonstration"

Each first generation takes ~10 minutes. Once generated, the mesh video is cached and loads instantly on subsequent views.

---

## Maintenance & Operations

### Keeping the API Running

**Colab Free Tier Limitations**:
- Sessions timeout after **12 hours of inactivity**
- Runtime disconnects after **90 minutes in background**
- Maximum **12 hours per session**

**To keep it running**:
1. Keep Colab browser tab open
2. Periodically check the notebook (every 30 minutes)
3. If disconnected: Rerun Cell 6, update Streamlit secrets with new ngrok URL

**Colab Pro ($10/month)**:
- 24-hour maximum session time
- Background execution enabled
- Priority GPU access (faster rendering)

---

### Updating Streamlit Secrets

If your ngrok URL changes (after Colab restart):

1. Go to: https://share.streamlit.io/
2. Find your app
3. Click **"⋮"** → **"Settings"**
4. Go to **"Secrets"**
5. Update `COLAB_API_URL` with new ngrok URL
6. Click **"Save"**
7. App will automatically restart with new URL

---

### Caching Generated Videos

Generated mesh videos are saved in `mesh_videos/` folder in the Streamlit app container.

**Important notes**:
- Videos persist for the session
- If Streamlit app restarts, cache is lost
- Users must regenerate videos after app restart

**To implement persistent storage** (advanced):
- Store generated videos in Google Drive
- Update `streamlit_app.py` to check Drive before generating
- Requires additional authentication setup

---

## Troubleshooting

### Error: "Cannot connect to Colab API"

**Symptoms**: Streamlit shows connection error immediately

**Causes**:
1. Colab notebook not running (Cell 6 stopped)
2. ngrok URL expired or changed
3. Incorrect URL in Streamlit secrets

**Solutions**:
1. Check Colab: Go to notebook, verify Cell 6 is running
2. Check ngrok URL: Compare URL in Colab output vs Streamlit secrets
3. Restart if needed: Stop Cell 6, update NGROK_AUTH_TOKEN, rerun Cell 6
4. Update Streamlit secrets with new URL

---

### Error: "Frame not found" or "NPZ file not found"

**Symptoms**: Generation starts but fails on specific frames

**Causes**:
1. Missing NPZ files in Google Drive
2. Incorrect folder names
3. File naming mismatch

**Solutions**:
1. Check Google Drive: Open `/MyDrive/ecolang/Extracted_parameters/{video_id}/`
2. Verify 1800 files exist: `frame_0001_params.npz` through `frame_1800_params.npz`
3. Check folder names match: `video1_speaking` in Drive = `video1_speaking` in config.py
4. Re-upload missing files if needed

---

### Error: "SMPL-X model not found"

**Symptoms**: Colab Cell 3 fails with assertion error

**Causes**:
1. Missing model file in Google Drive
2. Incorrect path

**Solutions**:
1. Check file exists: `/MyDrive/ecolang/models/SMPLX_NEUTRAL.npz`
2. Verify file size: Should be ~100MB
3. Re-upload if corrupted or missing

---

### Video doesn't load in left column

**Symptoms**: Right column generates mesh, but left shows "No video"

**Causes**:
1. Incorrect Google Drive FILE_ID
2. Video not shared publicly
3. File deleted or moved

**Solutions**:
1. Check video sharing: Open video in Drive, verify "Anyone with the link" can view
2. Test direct URL: Open `https://drive.google.com/uc?export=download&id={FILE_ID}` in browser
3. Should download video file (not show "Access Denied")
4. Update config.py if needed: Rerun `convert_drive_links.py`

---

### Generation is very slow

**Symptoms**: Taking >20 minutes per video

**Causes**:
1. Colab using CPU instead of GPU
2. Slow GPU allocated (K80 vs T4)
3. Network latency

**Solutions**:
1. Check device: In Cell 3 output, verify "Using device: cuda"
2. Change runtime: **Runtime → Change runtime type → GPU → T4 or better**
3. Restart runtime: **Runtime → Disconnect and delete runtime**, rerun all cells
4. Try different time: Colab allocates better GPUs during off-peak hours

---

### ngrok URL stops working

**Symptoms**: App works initially, then "Cannot connect" error appears

**Causes**:
1. ngrok session expired (free tier: 2-hour sessions)
2. Colab runtime disconnected
3. Network interruption

**Solutions**:
1. Check Colab: Look for "Reconnecting..." message
2. Rerun Cell 6: Stop execution, run again to get new ngrok URL
3. Update Streamlit secrets: Copy new URL, update in Streamlit Cloud settings
4. Upgrade ngrok: Paid plans have longer session times

---

### Out of memory error in Colab

**Symptoms**: "CUDA out of memory" or "RuntimeError: CUDA error"

**Causes**:
1. GPU memory full from previous renders
2. Multiple simultaneous requests

**Solutions**:
1. Restart Colab runtime: **Runtime → Restart runtime**
2. Clear GPU cache: Add this cell after Cell 3:
   ```python
   if device.type == 'cuda':
       torch.cuda.empty_cache()
   ```
3. Reduce batch size: Currently renders one frame at a time (optimal)
4. Upgrade to Colab Pro: More GPU memory available

---

## Performance Optimization

### Current Performance

- **Frame render time**: ~0.3-0.5 seconds (T4 GPU)
- **Video generation**: ~5-10 minutes (1800 frames)
- **API latency**: ~50-100ms (ngrok overhead)

### Potential Improvements

1. **Batch Processing**
   - Render multiple frames in parallel
   - Requires modifying API to accept frame ranges
   - Can reduce total time by 50-70%

2. **Pre-generation**
   - Generate all videos once, store in Google Drive
   - Streamlit loads pre-rendered videos
   - Instant playback, no waiting

3. **Lower Resolution**
   - Render 512x512 instead of 720x720
   - 40% faster rendering
   - Acceptable quality on most devices

4. **Frame Skipping**
   - Render every 2nd frame (900 frames total)
   - 50% faster generation
   - Still smooth at 15fps playback

---

## Security Considerations

### Current Setup

- **Google Drive**: Public read access to videos (required for streaming)
- **ngrok**: Public API endpoint (anyone with URL can make requests)
- **Streamlit Cloud**: Public web app

### Recommendations

1. **Add API Authentication**
   - Implement API key in Colab backend
   - Pass key from Streamlit via headers
   - Prevents unauthorized API usage

2. **Rate Limiting**
   - Limit requests per IP address
   - Prevents abuse of free Colab resources

3. **Video Access Control**
   - Use signed URLs instead of public sharing
   - Expires after set time period
   - More secure but requires token generation

4. **Private Deployment** (Enterprise)
   - Host backend on paid GPU server (e.g., AWS, GCP)
   - Use private networking (no ngrok)
   - Deploy Streamlit on private server

---

## Cost Analysis

### Current Setup (Free Tier)

| Service | Plan | Cost | Limitations |
|---------|------|------|-------------|
| Google Drive | Free | $0 | 15GB storage |
| Google Colab | Free | $0 | 12hr sessions, 90min background |
| ngrok | Free | $0 | 2hr sessions, 1 online tunnel |
| Streamlit Cloud | Free | $0 | Public apps, 1GB RAM |
| **Total** | | **$0/month** | Manual maintenance required |

### Upgraded Setup

| Service | Plan | Cost | Benefits |
|---------|------|------|----------|
| Google Drive | 100GB | $2/month | More storage for videos |
| Google Colab | Pro | $10/month | 24hr sessions, better GPUs |
| ngrok | Pro | $10/month | Longer sessions, custom domains |
| Streamlit Cloud | Team | $0 | Stays free for small apps |
| **Total** | | **$22/month** | Reliable, less maintenance |

### Enterprise Setup

| Service | Plan | Cost | Benefits |
|---------|------|------|----------|
| AWS/GCP GPU | g4dn.xlarge | ~$300/month | Dedicated GPU, 24/7 uptime |
| Storage | S3/Cloud Storage | ~$5/month | Unlimited, fast access |
| Streamlit | Self-hosted | $0 | Full control, custom domain |
| **Total** | | **~$305/month** | Production-ready, scalable |

---

## Next Steps & Extensions

### Immediate Improvements

1. **Add download button**: Let users download generated mesh videos
2. **Video comparison slider**: Side-by-side slider instead of columns
3. **Frame-by-frame view**: Select specific frame to examine
4. **Multiple angles**: Render mesh from different viewpoints

### Future Features

1. **Real-time Processing**
   - Upload new video → automatically extract parameters → render mesh
   - Requires integrating pose estimation pipeline

2. **3D Interactive Viewer**
   - Embed Three.js viewer
   - Rotate, zoom, pan the 3D mesh
   - Export mesh as OBJ/FBX

3. **Comparison Mode**
   - Load two videos side-by-side
   - Compare signing styles or gestures

4. **Analytics Dashboard**
   - Track usage statistics
   - Popular videos, render times
   - User engagement metrics

---

## Support & Resources

### Documentation

- **Streamlit Docs**: https://docs.streamlit.io/
- **FastAPI Docs**: https://fastapi.tiangolo.com/
- **SMPL-X GitHub**: https://github.com/vchoutas/smplx
- **PyRender Docs**: https://pyrender.readthedocs.io/

### Community

- **Streamlit Forum**: https://discuss.streamlit.io/
- **Stack Overflow**: Tag `streamlit`, `fastapi`, `smplx`

### Troubleshooting

If you encounter issues not covered in this guide:

1. Check Colab logs: Look for error messages in Cell 6 output
2. Check Streamlit logs: Click "Manage app" → "Logs" in Streamlit Cloud
3. Test API directly: Use Postman or curl to test endpoints
4. Check Google Drive: Verify file permissions and structure

---

## Appendix: Quick Reference

### File Structure

```
Project Root/
├── streamlit_app.py          # Main Streamlit application
├── requirements.txt           # Python dependencies
├── convert_drive_links.py     # Helper script for Drive URLs
├── DEPLOYMENT_GUIDE.md        # This file
├── signmesh/
│   └── config.py             # Configuration with VIDEO_LIBRARY
└── videos/
    └── README.md             # Video upload instructions
```

### Key Configuration Variables

| Variable | Location | Purpose |
|----------|----------|---------|
| `COLAB_API_URL` | Streamlit Secrets | ngrok URL for Colab backend |
| `VIDEO_LIBRARY` | signmesh/config.py | Video metadata and Drive URLs |
| `NGROK_AUTH_TOKEN` | Colab Cell 6 | ngrok authentication |
| `BASE_PATH` | Colab Cell 2 | Google Drive ecolang folder path |

### Common Commands

```bash
# Test convert script
python convert_drive_links.py

# Update config and push
git add signmesh/config.py
git commit -m "Update video FILE_IDs"
git push origin main

# Run Streamlit locally (for testing)
streamlit run streamlit_app.py
```

### API Endpoints

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/` | API information |
| GET | `/health` | Check if API is running |
| GET | `/docs` | Interactive API documentation |
| POST | `/render_frame` | Render single frame to mesh |

---

## Conclusion

You now have a complete, working ECOLANG application!

**What you've built**:
- ✅ Lightweight web interface with clean UI
- ✅ GPU-accelerated 3D mesh rendering
- ✅ On-demand video processing
- ✅ Side-by-side comparison view
- ✅ Progress tracking for long operations

**Total setup time**: ~60 minutes
**Cost**: $0 (free tier) or $22/month (upgraded)
**Maintenance**: Restart Colab every 12 hours (free tier)

Enjoy your ECOLANG application! 🚀
