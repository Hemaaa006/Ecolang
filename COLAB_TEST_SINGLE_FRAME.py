"""
ECOLANG - Test Single Frame Rendering
Tests the exact rendering logic before deploying full video rendering
Run this in Google Colab to verify the mesh renders correctly (front-facing, upper body)
"""

# ============= CELL 1: Setup Environment =============
!apt-get install -y xvfb ffmpeg
!pip install -q pyvirtualdisplay smplx trimesh pyrender PyOpenGL pillow matplotlib

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

# ============= CELL 4: Rendering Function (Exact Working Script) =============
import numpy as np
import trimesh
import pyrender
from PIL import Image
import matplotlib.pyplot as plt

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

def test_render_frame(video_folder, frame_num):
    """Test rendering a single frame - returns RGB image or None"""
    npz_path = os.path.join(BASE_PATH, "Extracted_parameters", video_folder, f"frame_{frame_num:04d}_params.npz")

    print(f"\nTesting frame {frame_num} from {video_folder}")
    print(f"NPZ path: {npz_path}")
    print(f"File exists: {os.path.exists(npz_path)}")

    if not os.path.exists(npz_path):
        print("ERROR: NPZ file not found!")
        return None

    try:
        # 1) Load NPZ and gather parameters
        Z = np.load(npz_path, allow_pickle=False)
        PERSON_ID = 0
        pfx = f"person_{PERSON_ID}_smplx_"

        print(f"\nLoading SMPL-X parameters...")
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

        print(f"Camera parameters:")
        print(f"  cam_trans: {cam_trans}")
        print(f"  focal (fx, fy): ({fx}, {fy})")
        print(f"  principal point (cx, cy): ({cx}, {cy})")

        # 2) Build SMPL-X mesh
        print(f"\nGenerating SMPL-X mesh...")
        with torch.no_grad():
            tens  = {k: torch.from_numpy(v).float().to(device) for k, v in params_np.items()}
            out   = smplx_model(**tens)
            verts = out.vertices[0].cpu().numpy().astype(np.float32)
        faces = smplx_model.faces

        print(f"  Vertices: {verts.shape}")
        print(f"  Faces: {faces.shape}")

        # 3) Put mesh in camera coordinates
        verts_cam = verts + cam_trans[None, :]
        print(f"  Vertices in camera coords: mean={verts_cam.mean(axis=0)}")

        # 4) Auto-fit to viewport (push Z if needed; recenter)
        print(f"\nAuto-fitting to viewport...")
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

        print(f"  Scaling factor: {k:.3f}")
        print(f"  Recentering: du={du:.1f}, dv={dv:.1f}")

        # 5) Convert to OpenGL coords for pyrender
        verts_gl       = verts_cam.copy()
        verts_gl[:,1] *= -1.0
        verts_gl[:,2] *= -1.0
        print(f"  OpenGL vertices: mean={verts_gl.mean(axis=0)}")

        # 6) Render full frame
        print(f"\nRendering full frame...")
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
        rgba, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        renderer.delete()

        print(f"  Rendered: {rgba.shape}")
        print(f"  Non-background pixels: {(rgba[:,:,:3] != 245).any(axis=2).sum()}")

        # 7) Body-aware crop (upper torso, with extra left/right space)
        print(f"\nApplying body-aware crop...")
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

        print(f"  Crop region: ({x0}, {y0}) -> ({x1}, {y1})")
        print(f"  Cropped size: {cropped.shape}")

        # Display results
        print(f"\nDisplaying results...")
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Full frame with crop rectangle
        axes[0].imshow(rgba[:,:,:3])
        axes[0].set_title(f"Full Frame - Frame {frame_num}")
        axes[0].axis('off')
        from matplotlib.patches import Rectangle
        axes[0].add_patch(Rectangle((x0, y0), x1-x0, y1-y0,
                                     linewidth=3, edgecolor='red', facecolor='none'))

        # Cropped result
        axes[1].imshow(cropped)
        axes[1].set_title(f"Upper Body Crop (Final Output)")
        axes[1].axis('off')

        # Depth map
        axes[2].imshow(depth, cmap='viridis')
        axes[2].set_title("Depth Map")
        axes[2].axis('off')

        plt.tight_layout()
        plt.savefig(f'/content/test_frame_{frame_num}.png', dpi=150, bbox_inches='tight')
        print(f"  Saved: /content/test_frame_{frame_num}.png")
        plt.show()

        # Save cropped result
        Image.fromarray(cropped).save(f'/content/rendered_frame_{frame_num}.png')
        print(f"  Saved: /content/rendered_frame_{frame_num}.png")

        print(f"\nSUCCESS! Check the output above.")
        print(f"Expected: Front-facing view, upper body visible, NOT upside-down")

        return cropped

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return None

# ============= CELL 5: Run Test =============
print("="*70)
print("TESTING SINGLE FRAME RENDERING")
print("="*70)

# Test with ch07 frame 100
result = test_render_frame("ch07_speakerview_012_parameters", 100)

if result is not None:
    print("\n" + "="*70)
    print("TEST PASSED!")
    print("="*70)
    print("\nThe mesh should be:")
    print("  - Front-facing (NOT upside-down)")
    print("  - Upper body visible (55% of full body)")
    print("  - Proper lighting and material")
    print("\nIf it looks correct, you can proceed with full video rendering!")
else:
    print("\n" + "="*70)
    print("TEST FAILED!")
    print("="*70)
    print("\nCheck the error messages above.")
