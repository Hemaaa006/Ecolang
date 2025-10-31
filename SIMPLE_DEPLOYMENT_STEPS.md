# ECOLANG - Simple Deployment Steps

You've already completed Phase 1 (Google Drive setup). Here are the remaining steps in plain English.

---

## ✅ PHASE 1: GOOGLE DRIVE SETUP - DONE!

You've already:
- Uploaded videos to Google Drive
- Updated config.py with FILE_IDs

---

## 📝 PHASE 2: CREATE GOOGLE COLAB BACKEND (20 minutes)

This creates the "brain" that will generate the 3D meshes.

### Step 2.1: Get ngrok Account (5 min)

1. Open browser, go to: **https://ngrok.com/signup**
2. Sign up (it's free - use email or Google account)
3. After signing up, go to: **https://dashboard.ngrok.com/get-started/your-authtoken**
4. You'll see a token like: `2abc123XYZ_long_random_string`
5. **COPY THIS TOKEN** and save it in a text file - you'll need it soon

### Step 2.2: Create Google Colab Notebook (15 min)

1. **Open Google Colab**
   - Go to: **https://colab.research.google.com/**
   - Sign in with your Google account (same one that has your Drive files)

2. **Create New Notebook**
   - Click: **File → New notebook**
   - A blank notebook opens

3. **Change to GPU**
   - Click: **Runtime → Change runtime type**
   - Under "Hardware accelerator", select: **GPU**
   - Click: **Save**

4. **Add the Code Cells**

   I'll give you 6 code blocks below. For each block:
   - Click in the notebook to create a new cell
   - Copy the entire code block
   - Paste it into the cell
   - Then create a new cell for the next block (click "+ Code" button)

---

### CELL 1: Install Dependencies

```python
# Install all required packages
!pip install -q fastapi uvicorn pyngrok pillow trimesh pyrender smplx torch opencv-python

print("✓ All dependencies installed successfully")
```

**After pasting, click the PLAY button (▶) on the left of the cell**
- This takes ~2 minutes
- Wait for the checkmark to appear

---

### CELL 2: Mount Google Drive

```python
from google.colab import drive
import os

# Mount Google Drive
drive.mount('/content/drive')

# Define paths
BASE_PATH = "/content/drive/MyDrive/ecolang"
NPZ_BASE_PATH = os.path.join(BASE_PATH, "Extracted_parameters")
MODEL_PATH = os.path.join(BASE_PATH, "models", "SMPLX_NEUTRAL.npz")

# Verify structure
print("Verifying Google Drive structure...\n")

# Check folders exist
assert os.path.exists(BASE_PATH), f"❌ Base folder not found: {BASE_PATH}"
print(f"✓ Base folder exists: {BASE_PATH}")

assert os.path.exists(NPZ_BASE_PATH), f"❌ NPZ folder not found: {NPZ_BASE_PATH}"
video_folders = [f for f in os.listdir(NPZ_BASE_PATH) if os.path.isdir(os.path.join(NPZ_BASE_PATH, f))]
print(f"✓ NPZ folder exists with {len(video_folders)} video folders:")
for folder in video_folders:
    folder_path = os.path.join(NPZ_BASE_PATH, folder)
    file_count = len([f for f in os.listdir(folder_path) if f.endswith('.npz')])
    print(f"   - {folder}: {file_count} NPZ files")

assert os.path.exists(MODEL_PATH), f"❌ SMPL-X model not found: {MODEL_PATH}"
print(f"✓ SMPL-X model exists")

print("\n" + "="*60)
print("✓ ALL CHECKS PASSED - Ready to start!")
print("="*60)
```

**After pasting, click PLAY**
- A popup appears asking to connect to Google Drive
- Click: **Connect to Google Drive**
- Select your Google account
- Click: **Allow**
- Wait for checkmarks

---

### CELL 3: Load SMPL-X Model

```python
import torch
import smplx
import numpy as np

# Select device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Load SMPL-X model
print("\nLoading SMPL-X model...")
smplx_model = smplx.create(
    model_path=os.path.dirname(MODEL_PATH),
    model_type='smplx',
    gender='neutral',
    use_face_contour=False,
    use_pca=True,
    num_pca_comps=12,
    flat_hand_mean=True
).to(device)

print("✓ SMPL-X model loaded successfully!")
```

**After pasting, click PLAY**
- Takes ~30 seconds
- Should see: "Using device: cuda"
- If it says "cpu" instead, go back to Runtime → Change runtime type → GPU

---

### CELL 4: Create Rendering Function

```python
import pyrender
import trimesh
from PIL import Image
import io
import base64

def render_mesh_from_npz(npz_path, person_id=0):
    """Load NPZ and render 3D mesh to image"""

    # Load parameters
    data = np.load(npz_path, allow_pickle=True)
    prefix = f'person_{person_id}_smplx_'

    # Check person exists
    person_ids = data.get('person_ids', np.array([]))
    if len(person_ids) == 0 or person_id not in person_ids:
        raise ValueError(f"Person {person_id} not found")

    # Extract parameters
    global_orient = torch.tensor(data[prefix + 'root_pose'].reshape(1, 3), dtype=torch.float32).to(device)
    body_pose = torch.tensor(data[prefix + 'body_pose'].reshape(1, 21, 3), dtype=torch.float32).to(device)
    betas = torch.tensor(data[prefix + 'shape'].reshape(1, 10), dtype=torch.float32).to(device)
    expression = torch.tensor(data.get(prefix + 'expr', np.zeros((1, 10))).reshape(1, 10), dtype=torch.float32).to(device)
    jaw_pose = torch.tensor(data.get(prefix + 'jaw_pose', np.zeros((1, 3))).reshape(1, 3), dtype=torch.float32).to(device)
    left_hand_pose = torch.tensor(data[prefix + 'lhand_pose'].reshape(1, 12), dtype=torch.float32).to(device)
    right_hand_pose = torch.tensor(data[prefix + 'rhand_pose'].reshape(1, 12), dtype=torch.float32).to(device)

    # Generate mesh
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

    # Create mesh
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    # Render
    scene = pyrender.Scene(ambient_light=[0.3, 0.3, 0.3])
    mesh_material = pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.2,
        roughnessFactor=0.8,
        baseColorFactor=[0.3, 0.3, 0.8, 1.0]
    )
    mesh_node = pyrender.Mesh.from_trimesh(mesh, material=mesh_material)
    scene.add(mesh_node)

    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
    camera_pose = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 2.5],
        [0.0, 0.0, 0.0, 1.0]
    ])
    scene.add(camera, pose=camera_pose)

    light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
    scene.add(light, pose=camera_pose)

    renderer = pyrender.OffscreenRenderer(720, 720)
    color, _ = renderer.render(scene)
    renderer.delete()

    img = Image.fromarray(color)
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

    return img_base64

# Test it
print("Testing render function...")
test_npz = os.path.join(NPZ_BASE_PATH, video_folders[0], "frame_0001_params.npz")
if os.path.exists(test_npz):
    try:
        test_img = render_mesh_from_npz(test_npz)
        print(f"✓ Render function works!")
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
```

**After pasting, click PLAY**
- Takes ~10 seconds
- Should see: "✓ Render function works!"

---

### CELL 5: Create API Server

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

app = FastAPI(title="ECOLANG Mesh API")

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
    return {"message": "ECOLANG API", "status": "online"}

@app.get("/health")
def health_check():
    return {"status": "ok", "message": "Running", "device": str(device)}

@app.post("/render_frame", response_model=RenderResponse)
async def render_frame(request: RenderRequest):
    try:
        if request.frame_number < 1 or request.frame_number > 1800:
            raise HTTPException(status_code=400, detail=f"Frame must be 1-1800")

        npz_filename = f"frame_{request.frame_number:04d}_params.npz"
        npz_path = os.path.join(NPZ_BASE_PATH, request.video_id, npz_filename)

        if not os.path.exists(npz_path):
            raise HTTPException(status_code=404, detail=f"Frame not found")

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
        return RenderResponse(success=False, image=None, frame_number=request.frame_number, error=str(e))

print("✓ API server configured")
```

**After pasting, click PLAY**
- Instant, just shows: "✓ API server configured"

---

### CELL 6: Start Server with ngrok

**⚠️ IMPORTANT: Before pasting, replace `YOUR_NGROK_TOKEN_HERE` with your actual token from Step 2.1**

```python
from pyngrok import ngrok
import uvicorn
import nest_asyncio

nest_asyncio.apply()

# PASTE YOUR NGROK TOKEN HERE (between the quotes)
NGROK_AUTH_TOKEN = "YOUR_NGROK_TOKEN_HERE"

ngrok.set_auth_token(NGROK_AUTH_TOKEN)
ngrok.kill()

public_url = ngrok.connect(8000)

print("\n" + "="*70)
print("🚀 ECOLANG API IS RUNNING!")
print("="*70)
print(f"\n📡 Public URL: {public_url}")
print("\n" + "="*70)
print("📋 COPY THIS URL - YOU'LL NEED IT FOR STREAMLIT:")
print("="*70)
print(f'\n{public_url}\n')
print("="*70)
print("⚠️  KEEP THIS CELL RUNNING - DON'T STOP IT!")
print("="*70 + "\n")

uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
```

**After pasting your token, click PLAY**
- A URL appears like: `https://abc123-xx-xx-xx-xx.ngrok-free.app`
- **COPY THIS URL** - you'll need it in Phase 3
- **LEAVE THIS RUNNING** - don't stop this cell!

---

### Step 2.3: Save the Notebook (Optional)

- Click: **File → Save**
- Rename it to: `ecolang_mesh_api`

---

## 🚀 PHASE 3: DEPLOY STREAMLIT (15 minutes)

Now we'll put your app on the internet.

### Step 3.1: Push Code to GitHub (if you haven't already)

**Skip this if your code is already on GitHub.**

1. Open Command Prompt:
   ```bash
   cd "c:\Users\dell\Desktop\Streamlit_render app"
   git init
   git add .
   git commit -m "Initial commit - ECOLANG app"
   ```

2. Create GitHub repository:
   - Go to: **https://github.com/new**
   - Repository name: `ecolang-app`
   - Make it Public
   - Click: **Create repository**

3. Push code:
   ```bash
   git remote add origin https://github.com/YOUR_USERNAME/ecolang-app.git
   git branch -M main
   git push -u origin main
   ```

### Step 3.2: Deploy to Streamlit Cloud

1. **Go to Streamlit Cloud**
   - Open: **https://share.streamlit.io/**
   - Sign in with GitHub

2. **Create New App**
   - Click: **"New app"** button
   - Repository: Select `ecolang-app` (or your repo name)
   - Branch: `main`
   - Main file path: `streamlit_app.py`

3. **⚠️ BEFORE CLICKING DEPLOY:**
   - Click: **"Advanced settings"**
   - Click: **"Secrets"**
   - Paste this (replace with YOUR ngrok URL from Phase 2):
     ```
     COLAB_API_URL = "https://your-ngrok-url.ngrok-free.app"
     ```
   - Click: **"Save"**

4. **Deploy**
   - Click: **"Deploy!"**
   - Wait 2-3 minutes
   - App URL appears: `https://yourapp.streamlit.app`

---

## 🧪 PHASE 4: TEST THE APPLICATION (10 minutes)

### Step 4.1: Open Your App

1. Click the Streamlit URL: `https://yourapp.streamlit.app`
2. You should see:
   - Header: "ECOLANG"
   - A dropdown menu with 4 videos

### Step 4.2: Generate First Mesh

1. Click the dropdown
2. Select: **"Video 1 - Speaking"**
3. The app will:
   - Show: "Generating mesh video..."
   - Display progress bar
   - Show original video on left immediately
   - Generate 1800 frames (~5-10 minutes)

### Step 4.3: Watch the Magic! ✨

When complete, you'll see:
- **Left side**: Original sign language video
- **Right side**: 3D mesh animation
- Both play at the same time!

### Step 4.4: Test Other Videos

Try selecting other videos from the dropdown. Each generates on first selection, then cached for instant playback.

---

## ✅ SUCCESS CHECKLIST

You're done when you can:
- [ ] Open your Streamlit app URL
- [ ] See the ECOLANG header
- [ ] Select a video from dropdown
- [ ] See progress bar counting frames
- [ ] Original video plays on left
- [ ] 3D mesh video plays on right after generation
- [ ] Both videos sync together

---

## 🆘 TROUBLESHOOTING

### "Cannot connect to Colab API"

**Fix:**
1. Go back to Google Colab
2. Check Cell 6 is still running (there's a spinning icon or it says "Running")
3. If stopped, click the PLAY button again
4. Copy the NEW ngrok URL
5. Update Streamlit Cloud secrets:
   - Go to: https://share.streamlit.io/
   - Find your app
   - Click: **"⋮" → "Settings" → "Secrets"**
   - Update `COLAB_API_URL` with new URL
   - Click: **"Save"**

### Video doesn't show on left

**Fix:**
1. Check your Google Drive videos are shared publicly
2. Test the direct URL in browser - it should download the video

### Generation is very slow

**Fix:**
1. In Colab, check Cell 3 output - should say "cuda" not "cpu"
2. If "cpu": **Runtime → Change runtime type → GPU → Save**
3. Rerun all cells

---

## 📊 WHAT TO EXPECT

- **First video generation**: 5-10 minutes (1800 frames)
- **Subsequent views**: Instant (cached)
- **Frame render speed**: ~0.3-0.5 seconds per frame
- **Total frames**: 1800 per video (60 seconds at 30fps)

---

## 💡 IMPORTANT NOTES

1. **Keep Colab running**: If Colab Cell 6 stops, your app breaks
2. **Free Colab limits**: Sessions timeout after 12 hours
3. **ngrok URL changes**: If you restart Colab, update Streamlit secrets with new URL
4. **First time only**: Each video generates once, then cached

---

## 🎉 CONGRATULATIONS!

You now have a working ECOLANG application that:
- Displays sign language videos
- Generates 3D meshes in real-time
- Shows them side-by-side
- All running for FREE on cloud services!

---

**Need help?** Check the detailed guides:
- [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) - Full documentation
- [README.md](README.md) - Project overview

**Questions?** Review the troubleshooting sections in those guides.
