"""
Single Frame SMPL-X Mesh Renderer for Google Colab

This script loads one NPZ file, reconstructs the SMPL-X mesh,
and renders it to an image with a plain background.

Usage:
    python render_single_frame.py \
        --npz_file /path/to/frame_0001.npz \
        --output mesh_output.png \
        --model_path /path/to/smplx/models
"""

import os
import sys
import argparse
import numpy as np
import torch
import trimesh
import pyrender
from PIL import Image

# Import SMPL-X
try:
    import smplx
except ImportError:
    print("Installing smplx...")
    os.system("pip install smplx")
    import smplx


def load_npz_params(npz_path, person_id=0):
    """Load SMPL-X parameters from NPZ file"""
    print(f"Loading NPZ file: {npz_path}")
    data = np.load(npz_path, allow_pickle=False)
    
    print(f"Keys found in NPZ: {list(data.keys())}")
    
    # Helper to get parameter with multiple possible key names
    def get_param(key_options, default_shape=None):
        for key in key_options:
            # Try direct key
            if key in data:
                return data[key]
            # Try with person prefix
            person_key = f"person_{person_id}_{key}"
            if person_key in data:
                return data[person_key]
        
        # Return zeros if not found
        if default_shape:
            print(f"Warning: Could not find {key_options}, using zeros")
            return np.zeros(default_shape, dtype=np.float32)
        return None
    
    params = {}
    
    # Root pose (global orientation)
    root_pose = get_param(['smplx_root_pose', 'root_pose', 'global_orient'], (3,))
    params['global_orient'] = root_pose.reshape(1, 3)
    print(f"  Root pose shape: {params['global_orient'].shape}")
    
    # Body pose
    body_pose = get_param(['smplx_body_pose', 'body_pose'], (63,))
    params['body_pose'] = body_pose.reshape(1, -1)
    print(f"  Body pose shape: {params['body_pose'].shape}")
    
    # Left hand
    lhand_pose = get_param(['smplx_lhand_pose', 'lhand_pose', 'left_hand_pose'], (45,))
    params['left_hand_pose'] = lhand_pose.reshape(1, -1)
    print(f"  Left hand shape: {params['left_hand_pose'].shape}")
    
    # Right hand
    rhand_pose = get_param(['smplx_rhand_pose', 'rhand_pose', 'right_hand_pose'], (45,))
    params['right_hand_pose'] = rhand_pose.reshape(1, -1)
    print(f"  Right hand shape: {params['right_hand_pose'].shape}")
    
    # Jaw pose
    jaw_pose = get_param(['smplx_jaw_pose', 'jaw_pose'], (3,))
    params['jaw_pose'] = jaw_pose.reshape(1, 3)
    print(f"  Jaw pose shape: {params['jaw_pose'].shape}")
    
    # Shape (betas)
    shape = get_param(['smplx_shape', 'shape', 'betas'])
    if shape is None:
        shape = np.zeros(10, dtype=np.float32)
    params['betas'] = shape.reshape(1, -1)
    print(f"  Shape (betas) shape: {params['betas'].shape}")
    
    # Expression
    expr = get_param(['smplx_expr', 'expr', 'expression'])
    if expr is None:
        expr = np.zeros(10, dtype=np.float32)
    params['expression'] = expr.reshape(1, -1)
    print(f"  Expression shape: {params['expression'].shape}")
    
    # Eye poses (always zero)
    params['leye_pose'] = np.zeros((1, 3), dtype=np.float32)
    params['reye_pose'] = np.zeros((1, 3), dtype=np.float32)
    
    # Camera translation (optional)
    cam_trans = get_param(['cam_trans', 'transl'])
    if cam_trans is not None:
        params['cam_trans'] = cam_trans
        print(f"  Camera translation: {cam_trans}")
    
    return params


def reconstruct_mesh(params, model_path, device='cuda'):
    """Reconstruct 3D mesh from SMPL-X parameters"""
    print("\nInitializing SMPL-X model...")
    
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create SMPL-X model
    smplx_model = smplx.create(
        model_path=model_path,
        model_type='smplx',
        gender='neutral',
        num_betas=params['betas'].shape[1],
        num_expression_coeffs=params['expression'].shape[1],
        use_pca=False,
        use_face_contour=True,
        ext='npz'
    ).to(device)
    
    print("Running SMPL-X forward pass...")
    
    # Convert to tensors
    with torch.no_grad():
        tensors = {}
        for key, value in params.items():
            if key != 'cam_trans':
                tensors[key] = torch.from_numpy(value).float().to(device)
        
        # Forward pass
        output = smplx_model(
            betas=tensors['betas'],
            body_pose=tensors['body_pose'],
            global_orient=tensors['global_orient'],
            right_hand_pose=tensors['right_hand_pose'],
            left_hand_pose=tensors['left_hand_pose'],
            jaw_pose=tensors['jaw_pose'],
            leye_pose=tensors['leye_pose'],
            reye_pose=tensors['reye_pose'],
            expression=tensors['expression']
        )
        
        vertices = output.vertices[0].cpu().numpy()
        faces = smplx_model.faces
    
    print(f"Mesh generated: {len(vertices)} vertices, {len(faces)} faces")
    return vertices, faces


def render_mesh(vertices, faces, img_size=720, bg_color=(245, 245, 245), cam_trans=None):
    """Render mesh to image with plain background"""
    print("\nRendering mesh...")
    
    # Set EGL for headless rendering
    os.environ['PYOPENGL_PLATFORM'] = 'egl'
    
    # Create trimesh object
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    
    # Center and scale mesh
    mesh_center = mesh.bounds.mean(axis=0)
    mesh.vertices -= mesh_center
    
    max_extent = np.max(mesh.extents)
    scale = 1.8 / max_extent
    mesh.vertices *= scale
    
    # Apply camera translation if available
    if cam_trans is not None:
        mesh.vertices += cam_trans * scale
        print(f"Applied camera translation: {cam_trans}")
    
    # Create material
    material = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=[0.8, 0.8, 0.8, 1.0],
        metallicFactor=0.0,
        roughnessFactor=0.7
    )
    
    py_mesh = pyrender.Mesh.from_trimesh(mesh, material=material)
    
    # Create scene
    scene = pyrender.Scene(bg_color=[c/255.0 for c in bg_color] + [1.0])
    scene.add(py_mesh)
    
    # Add camera
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
    camera_pose = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 2.5],
        [0.0, 0.0, 0.0, 1.0]
    ])
    scene.add(camera, pose=camera_pose)
    
    # Add lights
    light1 = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
    light2 = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.0)
    
    scene.add(light1, pose=np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 2.0],
        [0.0, 0.0, 0.0, 1.0]
    ]))
    
    scene.add(light2, pose=np.array([
        [1.0, 0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0, -1.0],
        [0.0, 0.0, 1.0, 2.0],
        [0.0, 0.0, 0.0, 1.0]
    ]))
    
    # Render
    renderer = pyrender.OffscreenRenderer(img_size, img_size)
    color, depth = renderer.render(scene)
    renderer.delete()
    
    print(f"Rendered image size: {color.shape}")
    return color


def main():
    parser = argparse.ArgumentParser(description='Render single SMPL-X mesh from NPZ')
    parser.add_argument('--npz_file', required=True, help='Path to NPZ file')
    parser.add_argument('--output', default='mesh_output.png', help='Output image path')
    parser.add_argument('--model_path', default='models', help='Path to SMPL-X model directory')
    parser.add_argument('--img_size', type=int, default=720, help='Output image size')
    parser.add_argument('--bg_color', default='#f5f5f5', help='Background color (hex)')
    parser.add_argument('--device', default='cuda', help='Device (cuda or cpu)')
    parser.add_argument('--person_id', type=int, default=0, help='Person ID (for multi-person)')
    
    args = parser.parse_args()
    
    # Convert hex to RGB
    hex_color = args.bg_color.lstrip('#')
    bg_color = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
    print("="*70)
    print("SMPL-X Single Frame Renderer")
    print("="*70)
    
    # Load parameters
    params = load_npz_params(args.npz_file, person_id=args.person_id)
    
    # Reconstruct mesh
    vertices, faces = reconstruct_mesh(params, args.model_path, device=args.device)
    
    # Get camera translation if available
    cam_trans = params.get('cam_trans')
    
    # Render
    img = render_mesh(
        vertices, 
        faces, 
        img_size=args.img_size, 
        bg_color=bg_color,
        cam_trans=cam_trans
    )
    
    # Save
    print(f"\nSaving to: {args.output}")
    Image.fromarray(img).save(args.output)
    
    print("\n" + "="*70)
    print("✓ Done!")
    print("="*70)
    
    # Display in Colab if available
    try:
        from IPython.display import Image as IPImage, display
        display(IPImage(args.output))
    except:
        pass


if __name__ == '__main__':
    main()
