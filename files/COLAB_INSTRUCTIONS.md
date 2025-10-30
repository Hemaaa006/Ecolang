# SMPL-X Mesh Renderer - Google Colab Quick Start

## Option 1: Copy-Paste Cells (Recommended)

Open the file `colab_notebook.py` and copy each cell into Google Colab.

### Steps:

1. **Open Google Colab**: https://colab.research.google.com/
2. **Create a new notebook**
3. **Copy each cell** from `colab_notebook.py` into separate Colab cells
4. **Update the paths** in Cell 5 to point to your files
5. **Run the cells in order**

## Option 2: Use Single Script

If you prefer a single Python script, use `render_single_frame.py`:

```python
# In Colab, first install dependencies:
!pip install smplx torch trimesh pyrender pillow PyOpenGL

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Download the script
!wget https://your-link-to/render_single_frame.py

# Run it
!python render_single_frame.py \
    --npz_file "/content/drive/MyDrive/OSXBackup/out_npz/frame_0001_params.npz" \
    --output "mesh_output.png" \
    --model_path "/content/drive/MyDrive/models"
```

## Quick Copy-Paste Version

Here's the minimal code to render one frame. Copy this into a single Colab cell:

```python
# Install dependencies
!pip install -q smplx torch trimesh pyrender pillow PyOpenGL

# Imports
import os, numpy as np, torch, trimesh, pyrender, smplx
from PIL import Image
from IPython.display import Image as IPImage, display

# Mount Drive (if needed)
from google.colab import drive
drive.mount('/content/drive')

# === UPDATE THESE PATHS ===
NPZ_FILE = "/content/drive/MyDrive/OSXBackup/out_npz/frame_0001_params.npz"
MODEL_PATH = "/content/drive/MyDrive/models"
OUTPUT = "/content/mesh.png"

# Load NPZ
data = np.load(NPZ_FILE, allow_pickle=False)
print("NPZ keys:", list(data.keys())[:10])

def get(keys, default=None):
    for k in keys:
        if k in data: return data[k]
        if f'person_0_{k}' in data: return data[f'person_0_{k}']
    return default if default is not None else np.zeros(3)

# Extract parameters
params = {
    'global_orient': get(['smplx_root_pose', 'root_pose', 'global_orient']).reshape(1,3),
    'body_pose': get(['smplx_body_pose', 'body_pose'], np.zeros(63)).reshape(1,-1),
    'left_hand_pose': get(['smplx_lhand_pose', 'lhand_pose'], np.zeros(45)).reshape(1,-1),
    'right_hand_pose': get(['smplx_rhand_pose', 'rhand_pose'], np.zeros(45)).reshape(1,-1),
    'jaw_pose': get(['smplx_jaw_pose', 'jaw_pose']).reshape(1,3),
    'betas': get(['smplx_shape', 'shape', 'betas'], np.zeros(10)).reshape(1,-1),
    'expression': get(['smplx_expr', 'expr'], np.zeros(10)).reshape(1,-1),
    'leye_pose': np.zeros((1,3)),
    'reye_pose': np.zeros((1,3))
}

# Initialize SMPL-X
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using: {device}")

model = smplx.create(MODEL_PATH, 'smplx', 'neutral', 
                     num_betas=params['betas'].shape[1],
                     num_expression_coeffs=params['expression'].shape[1],
                     use_pca=False, ext='npz').to(device)

# Reconstruct mesh
with torch.no_grad():
    tensors = {k: torch.from_numpy(v).float().to(device) for k,v in params.items()}
    output = model(**tensors)
    vertices = output.vertices[0].cpu().numpy()
    faces = model.faces

print(f"Mesh: {len(vertices)} verts, {len(faces)} faces")

# Render
os.environ['PYOPENGL_PLATFORM'] = 'egl'
mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
mesh.vertices -= mesh.bounds.mean(0)
mesh.vertices *= 1.8 / mesh.extents.max()

material = pyrender.MetallicRoughnessMaterial(
    baseColorFactor=[0.8,0.8,0.8,1.0], metallicFactor=0.0, roughnessFactor=0.7)
py_mesh = pyrender.Mesh.from_trimesh(mesh, material=material)

scene = pyrender.Scene(bg_color=[0.96,0.96,0.96,1.0])
scene.add(py_mesh)
scene.add(pyrender.PerspectiveCamera(yfov=np.pi/3.0), 
          pose=np.diag([1,1,1,1]).astype(float))
scene.add(pyrender.DirectionalLight(color=[1,1,1], intensity=3.0),
          pose=np.array([[1,0,0,0],[0,1,0,1],[0,0,1,2],[0,0,0,1]]).astype(float))

renderer = pyrender.OffscreenRenderer(720, 720)
color, _ = renderer.render(scene)
renderer.delete()

# Save and display
Image.fromarray(color).save(OUTPUT)
print(f"✓ Saved: {OUTPUT}")
display(IPImage(OUTPUT))
```

## Troubleshooting

### "SMPL-X models not found"
1. Download from https://smpl-x.is.tue.mpg.de/
2. Upload `SMPLX_NEUTRAL.npz` to your Google Drive
3. Update `MODEL_PATH` to point to the folder containing it

### "NPZ file not found"
- Check the path is correct
- Make sure Google Drive is mounted
- Use `!ls /content/drive/MyDrive/` to verify

### "Key not found in NPZ"
- Run Cell 7 (inspect NPZ) to see available keys
- Your NPZ might use different key names
- The code tries multiple common key names automatically

### Rendering is slow
- Colab gives you a free GPU - make sure it's enabled
- Runtime → Change runtime type → GPU

### Out of memory
- Reduce `img_size` to 512 or 480
- Use CPU instead: `device='cpu'`

## Expected Output

You should see:
1. Installation messages
2. "Using: cuda" (or cpu)
3. "Mesh: XXXXX verts, XXXXX faces"
4. "✓ Saved: /content/mesh.png"
5. The rendered mesh image displayed below

## Next Steps

Once you have one frame working:
- Use Cell 8 to render multiple frames
- Adjust lighting/camera in the render function
- Try different background colors
- Export mesh to GLB for 3D viewing

## Full Documentation

For the complete system with video rendering and web viewers, see the main README.md in the full package.
