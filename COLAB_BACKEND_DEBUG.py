"""
ECOLANG Colab Backend - DEBUG VERSION
Adds debugging to identify why video is rendering black
"""

# ============= CELL 1: Setup Environment =============
!apt-get install -y xvfb ffmpeg
!pip install -q pyvirtualdisplay smplx trimesh pyrender nest_asyncio fastapi uvicorn pyngrok opencv-python pillow google-api-python-client google-auth matplotlib

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
print(f"Model faces: {len(smplx_model.faces)}")
print(f"Model device: {next(smplx_model.parameters()).device}")

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

# ============= CELL 4.5: Test Single Frame Rendering =============
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def test_single_frame(video_id='ch07_speakerview_012', frame_num=1):
    """Test rendering a single frame with debugging"""
    print(f"\n{'='*60}")
    print(f"🔍 DEBUGGING FRAME {frame_num} OF {video_id}")
    print(f"{'='*60}\n")

    folder = VIDEO_FOLDER_MAPPING[video_id]
    npz_path = os.path.join(BASE_PATH, "Extracted_parameters", folder, f"frame_{frame_num:04d}_params.npz")

    print(f"1. Loading NPZ: {npz_path}")
    print(f"   Exists: {os.path.exists(npz_path)}")

    if not os.path.exists(npz_path):
        print("❌ NPZ file not found!")
        return

    # Load data
    data = np.load(npz_path, allow_pickle=True)
    print(f"✓ NPZ loaded")
    print(f"   Keys: {list(data.keys())[:10]}...")

    person_ids = data.get('person_ids', np.array([]))
    print(f"   Person IDs: {person_ids}")

    if len(person_ids) == 0:
        print("❌ No person detected in frame!")
        return

    prefix = 'person_0_smplx_'

    # Check parameters
    print(f"\n2. Checking SMPL-X parameters:")
    for key in ['root_pose', 'body_pose', 'shape', 'lhand_pose', 'rhand_pose']:
        param_key = prefix + key
        if param_key in data:
            param = data[param_key]
            print(f"   ✓ {key}: shape={param.shape}, range=[{param.min():.3f}, {param.max():.3f}]")
        else:
            print(f"   ❌ {key}: MISSING")

    # Check camera translation
    cam_trans = data.get('person_0_cam_trans')
    if cam_trans is not None:
        print(f"   ✓ cam_trans: {cam_trans}")
    else:
        print(f"   ⚠️  cam_trans: None (will use default)")

    # Load parameters as tensors
    print(f"\n3. Creating SMPL-X parameters:")
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

    for k, v in params.items():
        print(f"   {k}: {v.shape}")

    # Generate mesh
    print(f"\n4. Generating mesh:")
    with torch.no_grad():
        output = smplx_model(**params)
        vertices = output.vertices[0].cpu().numpy()
        faces = smplx_model.faces

    print(f"   ✓ Vertices: {vertices.shape}")
    print(f"   ✓ Faces: {faces.shape}")
    print(f"   Vertex range: [{vertices.min():.3f}, {vertices.max():.3f}]")
    print(f"   Vertex mean: {vertices.mean(axis=0)}")

    # Apply camera translation
    if cam_trans is not None:
        print(f"\n5. Applying camera translation: {cam_trans}")
        vertices += cam_trans
        print(f"   After translation: mean={vertices.mean(axis=0)}")

    # Render
    print(f"\n6. Rendering mesh:")
    import trimesh
    import pyrender

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    print(f"   Mesh bounds: {mesh.bounds}")
    print(f"   Mesh center: {mesh.bounds.mean(axis=0)}")

    # Center and scale
    bounds = mesh.bounds
    center = bounds.mean(axis=0)
    mesh.vertices -= center
    scale = 2.0 / (bounds[1] - bounds[0]).max()
    mesh.vertices *= scale

    print(f"   After centering: bounds={mesh.bounds}")
    print(f"   Scale factor: {scale}")

    # Create material
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

    # Render
    renderer = pyrender.OffscreenRenderer(720, 720)
    color, depth = renderer.render(scene)
    renderer.delete()

    print(f"   ✓ Rendered image: {color.shape}")
    print(f"   Image range: [{color.min()}, {color.max()}]")
    print(f"   Image mean: {color.mean()}")
    print(f"   Non-background pixels: {(color != 245).any(axis=2).sum()}")

    # Display
    print(f"\n7. Displaying result:")
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(color)
    axes[0].set_title(f"Rendered Frame {frame_num}")
    axes[0].axis('off')

    axes[1].imshow(depth, cmap='gray')
    axes[1].set_title("Depth Map")
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig(f'/content/debug_frame_{frame_num}.png', dpi=150, bbox_inches='tight')
    print(f"   ✓ Saved to: /content/debug_frame_{frame_num}.png")
    plt.show()

    # Save render
    img = Image.fromarray(color)
    img.save(f'/content/rendered_frame_{frame_num}.png')
    print(f"   ✓ Saved render to: /content/rendered_frame_{frame_num}.png")

    print(f"\n{'='*60}")
    print(f"✅ DEBUG COMPLETE")
    print(f"{'='*60}\n")

    return color, depth

# Run test
print("\n🧪 RUNNING SINGLE FRAME TEST...")
result = test_single_frame('ch07_speakerview_012', frame_num=100)

print("\n✓ Test complete! Check the images above.")
print("If the render is blank, the issue is with mesh generation or camera positioning.")
