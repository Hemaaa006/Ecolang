# ✅ Setup Google Colab Backend - CUSTOM FOR YOUR STRUCTURE

Your config.py is already updated with the correct FILE_IDs!

**Location:** `c:\Users\dell\Desktop\Steamlit_render app\ecolang\config.py`

---

## 📁 Your Google Drive Structure (Confirmed):

```
/MyDrive/ecolang/
├── Extracted_parameters/
│   ├── ch07_speakerview_012_parameters/
│   ├── ch08_speakerview_025_parameters/
│   ├── ch09_speakerview_027_parameters/
│   └── ch11_speakerview_002_parameters/
├── models/
│   └── SMPLX_NEUTRAL.npz
└── videos/
    ├── ch07_speakerview_012.mp4
    ├── ch08_speakerview_025.mp4
    ├── ch09_speakerview_027.mp4
    └── ch11_speakerview_002.mp4
```

---

## 🚀 Run Colab Backend - Copy These 6 Cells

### CELL 1: Install Dependencies

```python
!pip install -q fastapi uvicorn pyngrok pillow trimesh pyrender smplx torch opencv-python
print("✓ Dependencies installed")
```

---

### CELL 2: Mount Drive & Verify

```python
from google.colab import drive
import os

drive.mount('/content/drive')

BASE_PATH = "/content/drive/MyDrive/ecolang"
NPZ_BASE_PATH = os.path.join(BASE_PATH, "Extracted_parameters")
MODEL_PATH = os.path.join(BASE_PATH, "models", "SMPLX_NEUTRAL.npz")

VIDEO_FOLDER_MAPPING = {
    "ch07_speakerview_012": "ch07_speakerview_012_parameters",
    "ch08_speakerview_025": "ch08_speakerview_025_parameters",
    "ch09_speakerview_027": "ch09_speakerview_027_parameters",
    "ch11_speakerview_002": "ch11_speakerview_002_parameters"
}

print("Verifying structure...")
assert os.path.exists(BASE_PATH), f"Base folder not found: {BASE_PATH}"
print(f"✓ Base: {BASE_PATH}")

assert os.path.exists(NPZ_BASE_PATH), f"NPZ folder not found"
print(f"✓ NPZ folder found")

for video_id, folder_name in VIDEO_FOLDER_MAPPING.items():
    folder_path = os.path.join(NPZ_BASE_PATH, folder_name)
    if os.path.exists(folder_path):
        count = len([f for f in os.listdir(folder_path) if f.endswith('.npz')])
        print(f"✓ {folder_name}: {count} files")

assert os.path.exists(MODEL_PATH), f"Model not found: {MODEL_PATH}"
print(f"✓ SMPL-X model found")
print("\n✓ ALL CHECKS PASSED!")
```

---

### CELL 3: Load SMPL-X Model

```python
import torch
import smplx
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using: {device}")

# Create smplx subfolder if needed
SMPLX_DIR = os.path.join(BASE_PATH, "models", "smplx")
os.makedirs(SMPLX_DIR, exist_ok=True)

# Check if model file is in the right location
source_model = os.path.join(BASE_PATH, "models", "SMPLX_NEUTRAL.npz")
target_model = os.path.join(SMPLX_DIR, "SMPLX_NEUTRAL.npz")

if os.path.exists(source_model) and not os.path.exists(target_model):
    # Copy model to correct location
    import shutil
    shutil.copy(source_model, target_model)
    print(f"✓ Copied model to: {SMPLX_DIR}")

smplx_model = smplx.create(
    model_path=os.path.join(BASE_PATH, "models"),
    model_type='smplx',
    gender='neutral',
    use_face_contour=False,
    use_pca=False,  # Changed to False - your data has full hand poses
    flat_hand_mean=False
).to(device)

print("✓ SMPL-X loaded with full hand articulation!")
```

---

### CELL 4: Create Render Function

```python
import pyrender
import trimesh
from PIL import Image
import io
import base64

def render_mesh_from_npz(npz_path, person_id=0):
    data = np.load(npz_path, allow_pickle=True)
    prefix = f'person_{person_id}_smplx_'

    person_ids = data.get('person_ids', np.array([]))
    if len(person_ids) == 0 or person_id not in person_ids:
        raise ValueError(f"Person {person_id} not found")

    global_orient = torch.tensor(data[prefix + 'root_pose'].reshape(1, 3), dtype=torch.float32).to(device)
    body_pose = torch.tensor(data[prefix + 'body_pose'].reshape(1, 21, 3), dtype=torch.float32).to(device)
    betas = torch.tensor(data[prefix + 'shape'].reshape(1, 10), dtype=torch.float32).to(device)
    expression = torch.tensor(data.get(prefix + 'expr', np.zeros((1, 10))).reshape(1, 10), dtype=torch.float32).to(device)
    jaw_pose = torch.tensor(data.get(prefix + 'jaw_pose', np.zeros((1, 3))).reshape(1, 3), dtype=torch.float32).to(device)
    left_hand_pose = torch.tensor(data[prefix + 'lhand_pose'].reshape(1, 45), dtype=torch.float32).to(device)
    right_hand_pose = torch.tensor(data[prefix + 'rhand_pose'].reshape(1, 45), dtype=torch.float32).to(device)

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

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    scene = pyrender.Scene(ambient_light=[0.3, 0.3, 0.3])
    mesh_material = pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.2, roughnessFactor=0.8,
        baseColorFactor=[0.3, 0.3, 0.8, 1.0]
    )
    mesh_node = pyrender.Mesh.from_trimesh(mesh, material=mesh_material)
    scene.add(mesh_node)

    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
    camera_pose = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,2.5],[0,0,0,1]])
    scene.add(camera, pose=camera_pose)

    light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
    scene.add(light, pose=camera_pose)

    renderer = pyrender.OffscreenRenderer(720, 720)
    color, _ = renderer.render(scene)
    renderer.delete()

    img = Image.fromarray(color)
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

# Test with debugging
print("Testing render function...")
test_folder = list(VIDEO_FOLDER_MAPPING.values())[0]
test_dir = os.path.join(NPZ_BASE_PATH, test_folder)

print(f"\nLooking in: {test_dir}")
if os.path.exists(test_dir):
    files = sorted([f for f in os.listdir(test_dir) if f.endswith('.npz')])
    print(f"Found {len(files)} NPZ files")
    if files:
        print(f"First file: {files[0]}")
        test_npz = os.path.join(test_dir, files[0])
        try:
            test_img = render_mesh_from_npz(test_npz)
            print(f"✓ Render function works!")
        except Exception as e:
            print(f"❌ Render failed: {str(e)}")
    else:
        print("❌ No NPZ files found in directory")
else:
    print(f"❌ Directory does not exist: {test_dir}")
```

---

### CELL 5: Create API

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

app = FastAPI()

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
    return {"status": "online", "videos": list(VIDEO_FOLDER_MAPPING.keys())}

@app.get("/health")
def health():
    return {"status": "ok", "device": str(device)}

@app.post("/render_frame", response_model=RenderResponse)
async def render_frame(req: RenderRequest):
    try:
        if req.video_id not in VIDEO_FOLDER_MAPPING:
            raise HTTPException(404, f"Video {req.video_id} not found")

        folder = VIDEO_FOLDER_MAPPING[req.video_id]
        npz_file = f"frame_{req.frame_number:04d}_params.npz"
        npz_path = os.path.join(NPZ_BASE_PATH, folder, npz_file)

        if not os.path.exists(npz_path):
            raise HTTPException(404, f"Frame {req.frame_number} not found")

        img = render_mesh_from_npz(npz_path, req.person_id)
        return RenderResponse(success=True, image=img, frame_number=req.frame_number)
    except HTTPException:
        raise
    except Exception as e:
        return RenderResponse(success=False, image=None, frame_number=req.frame_number, error=str(e))

print("✓ API configured")
```

---

### CELL 6: Start Server (ADD YOUR NGROK TOKEN)

```python
from pyngrok import ngrok
import nest_asyncio
from threading import Thread

nest_asyncio.apply()

# GET YOUR TOKEN FROM: https://dashboard.ngrok.com/get-started/your-authtoken
NGROK_AUTH_TOKEN = "YOUR_NGROK_TOKEN_HERE"  # ← REPLACE THIS

ngrok.set_auth_token(NGROK_AUTH_TOKEN)
ngrok.kill()

public_url = ngrok.connect(8000)

# Extract clean URL
url_str = str(public_url).split('"')[1] if '"' in str(public_url) else str(public_url)

print("\n" + "="*70)
print("🚀 API RUNNING!")
print("="*70)
print(f"\nPublic URL: {url_str}")
print("\nCopy this URL for Streamlit secrets:")
print(f'\nCOLAB_API_URL = "{url_str}"')
print("\n" + "="*70)
print("⚠️  KEEP THIS RUNNING!")
print("="*70 + "\n")

# Run server in thread to avoid event loop issues
import uvicorn
config = uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="info")
server = uvicorn.Server(config)

# Keep this cell running
import asyncio
await server.serve()
```

---

## ✅ Steps:

1. Open: https://colab.research.google.com/
2. New notebook → GPU
3. Copy each cell above into Colab
4. Get ngrok token from https://ngrok.com/signup
5. Add token to CELL 6
6. Run all cells
7. Copy the URL from CELL 6 output

---

**Next:** Use that URL in your Streamlit deployment!
