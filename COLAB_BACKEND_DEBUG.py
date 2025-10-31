"""
ECOLANG Colab Backend - DEBUG VERSION
Tests single frame rendering with proper camera intrinsics
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
import trimesh
import pyrender

def test_single_frame(video_id='ch07_speakerview_012', frame_num=100):
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
    for key in ['root_pose', 'body_pose', 'shape', 'lhand_pose', 'rhand_pose', 'jaw_pose', 'expr']:
        param_key = prefix + key
        if param_key in data:
            param = data[param_key]
            print(f"   ✓ {key}: shape={param.shape}, range=[{param.min():.3f}, {param.max():.3f}]")
        else:
            print(f"   ❌ {key}: MISSING")

    # Check camera parameters
    cam_trans = data.get('person_0_cam_trans')
    fx, fy = map(float, data.get('person_0_focal', [1200.0, 1200.0]))
    cx, cy = map(float, data.get('person_0_princpt', [360.0, 360.0]))

    print(f"\n3. Camera parameters:")
    print(f"   cam_trans: {cam_trans}")
    print(f"   focal (fx, fy): ({fx}, {fy})")
    print(f"   principal point (cx, cy): ({cx}, {cy})")

    # Load parameters as tensors
    print(f"\n4. Creating SMPL-X parameters:")
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
    print(f"\n5. Generating mesh:")
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
        print(f"\n6. Applying camera translation: {cam_trans}")
        vertices += cam_trans
        print(f"   After translation: mean={vertices.mean(axis=0)}")

    # Render using proper camera intrinsics
    print(f"\n7. Rendering mesh with camera intrinsics:")

    H, W = 720, 720
    MARGIN = 0.05
    UPPER_FRAC = 0.55
    PAD_X = 0.60
    PAD_Y = 0.14

    def project_uv(verts_cam, fx, fy, cx, cy):
        X, Y, Z = verts_cam[:,0], verts_cam[:,1], verts_cam[:,2]
        u = fx * (X / Z) + cx
        v = fy * (Y / Z) + cy
        return np.stack([u, v], axis=1)

    # Put mesh in camera coordinates
    verts_cam = vertices.copy()

    # Auto-fit to viewport
    uv = project_uv(verts_cam, fx, fy, cx, cy)
    umin, vmin = uv.min(axis=0)
    umax, vmax = uv.max(axis=0)
    bbox_w, bbox_h = umax - umin, vmax - vmin

    W_eff = W * (1.0 - 2*MARGIN)
    H_eff = H * (1.0 - 2*MARGIN)
    k = max(bbox_w / max(W_eff, 1e-6), bbox_h / max(H_eff, 1e-6), 1.0)
    verts_cam[:,2] *= k

    print(f"   Scaling factor k: {k}")

    # Recenter in viewport
    uv = project_uv(verts_cam, fx, fy, cx, cy)
    uc, vc = uv.mean(axis=0)
    du = (W/2.0) - uc
    dv = (H/2.0) - vc
    z_mean = np.median(verts_cam[:,2])
    verts_cam[:,0] += (du * z_mean) / fx
    verts_cam[:,1] += (dv * z_mean) / fy

    print(f"   Recentering: du={du:.2f}, dv={dv:.2f}")

    # Convert to OpenGL coordinates
    verts_gl = verts_cam.copy()
    verts_gl[:,1] *= -1.0
    verts_gl[:,2] *= -1.0

    print(f"   OpenGL vertices: mean={verts_gl.mean(axis=0)}")

    # Create scene
    scene = pyrender.Scene(bg_color=[0.96, 0.96, 0.96, 1.0])
    material = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=[0.86, 0.86, 0.86, 1.0],
        metallicFactor=0.0,
        roughnessFactor=0.7
    )
    tri = trimesh.Trimesh(verts_gl, faces, process=False)
    scene.add(pyrender.Mesh.from_trimesh(tri, material=material, smooth=True))

    # Use IntrinsicsCamera
    camera = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy)
    scene.add(camera, pose=np.eye(4, dtype=np.float32))

    # Add lights
    scene.add(pyrender.DirectionalLight(intensity=3.0), pose=np.eye(4, dtype=np.float32))
    L2 = np.eye(4, dtype=np.float32)
    L2[:3,3] = np.array([-1.0, -0.3, 2.0], dtype=np.float32)
    scene.add(pyrender.DirectionalLight(intensity=1.8), pose=L2)

    # Render
    renderer = pyrender.OffscreenRenderer(viewport_width=W, viewport_height=H)
    rgba, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    renderer.delete()

    print(f"   ✓ Rendered full image: {rgba.shape}")
    print(f"   Image range: [{rgba.min()}, {rgba.max()}]")
    print(f"   Non-background pixels: {(rgba[:,:,:3] != 245).any(axis=2).sum()}")

    # Body-aware crop
    uv = project_uv(verts_cam, fx, fy, cx, cy)
    umin, vmin = uv.min(axis=0)
    umax, vmax = uv.max(axis=0)
    bbox_w, bbox_h = umax - umin, vmax - vmin

    left = umin - PAD_X * bbox_w
    right = umax + PAD_X * bbox_w
    top = vmin - PAD_Y * bbox_h
    bottom = vmin + UPPER_FRAC * bbox_h

    left = max(0.0, left)
    right = min(float(W), right)
    top = max(0.0, top)
    bottom = min(float(H), bottom)

    x0, y0 = int(round(left)), int(round(top))
    x1, y1 = int(round(right)), int(round(bottom))

    if x1 <= x0 or y1 <= y0:
        x0, y0, x1, y1 = 0, 0, W, H

    cropped = rgba[y0:y1, x0:x1, :3]

    print(f"   Crop region: ({x0}, {y0}) to ({x1}, {y1})")
    print(f"   Cropped size: {cropped.shape}")

    # Display
    print(f"\n8. Displaying result:")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(rgba[:,:,:3])
    axes[0].set_title(f"Full Render - Frame {frame_num}")
    axes[0].axis('off')

    # Draw crop rectangle
    from matplotlib.patches import Rectangle
    axes[0].add_patch(Rectangle((x0, y0), x1-x0, y1-y0,
                                  linewidth=2, edgecolor='red', facecolor='none'))

    axes[1].imshow(cropped)
    axes[1].set_title(f"Upper Body Crop")
    axes[1].axis('off')

    axes[2].imshow(depth, cmap='gray')
    axes[2].set_title("Depth Map")
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(f'/content/debug_frame_{frame_num}.png', dpi=150, bbox_inches='tight')
    print(f"   ✓ Saved to: /content/debug_frame_{frame_num}.png")
    plt.show()

    # Save cropped render
    img = Image.fromarray(cropped)
    img.save(f'/content/rendered_frame_{frame_num}.png')
    print(f"   ✓ Saved render to: /content/rendered_frame_{frame_num}.png")

    print(f"\n{'='*60}")
    print(f"✅ DEBUG COMPLETE")
    print(f"{'='*60}\n")

    return cropped, depth

# Run test
print("\n🧪 RUNNING SINGLE FRAME TEST...")
result = test_single_frame('ch07_speakerview_012', frame_num=100)

print("\n✓ Test complete! Check the images above.")
print("The render should show:")
print("  - Front view (not upside-down)")
print("  - Upper body only (55% of full body)")
print("  - Proper lighting and material")
