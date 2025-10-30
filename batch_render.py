"""
Production Script: Batch Render All Frames with Error Handling

Handles missing frames by:
1. Detecting which frames have valid data
2. Using last valid frame for missing detections
3. Generating detailed error report
"""

import numpy as np
import torch
import trimesh
import pyrender
from PIL import Image
import smplx
import os
from pathlib import Path
from tqdm import tqdm
import json
import argparse


def check_frame_has_person(npz_path):
    """Quick check if frame has person detection"""
    try:
        data = np.load(npz_path, allow_pickle=True)
        person_ids = data.get('person_ids', np.array([]))
        return len(person_ids) > 0, data.files if len(person_ids) > 0 else []
    except:
        return False, []


def load_params_safe(npz_path, person_id=0, use_pca=False):
    """Load parameters with error handling"""
    try:
        data = np.load(npz_path, allow_pickle=True)
        
        person_ids = data.get('person_ids', np.array([]))
        if len(person_ids) == 0:
            return None, "no_person"
        
        prefix = f'person_{person_id}_smplx_'
        
        # Check required keys
        required = ['root_pose', 'body_pose', 'shape', 'expr', 'jaw_pose']
        missing = [k for k in required if prefix + k not in data.files]
        if missing:
            return None, f"missing_keys:{','.join(missing)}"
        
        params = {
            'global_orient': data[prefix + 'root_pose'].reshape(1, 3).astype(np.float32),
            'body_pose': data[prefix + 'body_pose'].reshape(1, -1).astype(np.float32),
            'jaw_pose': data[prefix + 'jaw_pose'].reshape(1, 3).astype(np.float32),
            'betas': data[prefix + 'shape'].reshape(1, -1).astype(np.float32),
            'expression': data[prefix + 'expr'].reshape(1, -1).astype(np.float32),
            'leye_pose': np.zeros((1, 3), dtype=np.float32),
            'reye_pose': np.zeros((1, 3), dtype=np.float32)
        }
        
        if use_pca:
            params['left_hand_pose'] = np.zeros((1, 12), dtype=np.float32)
            params['right_hand_pose'] = np.zeros((1, 12), dtype=np.float32)
        else:
            params['left_hand_pose'] = data[prefix + 'lhand_pose'].reshape(1, -1).astype(np.float32)
            params['right_hand_pose'] = data[prefix + 'rhand_pose'].reshape(1, -1).astype(np.float32)
        
        cam_trans = data.get(f'person_{person_id}_cam_trans')
        
        return (params, cam_trans), None
        
    except Exception as e:
        return None, f"error:{str(e)}"


def reconstruct_mesh(model, params, cam_trans, device):
    """Reconstruct mesh"""
    try:
        with torch.no_grad():
            tensors = {k: torch.from_numpy(v).float().to(device) for k, v in params.items()}
            output = model(**tensors)
            vertices = output.vertices[0].cpu().numpy()
            faces = model.faces
        
        if cam_trans is not None:
            vertices += cam_trans
        
        return vertices, faces, None
    except Exception as e:
        return None, None, str(e)


def render_mesh(vertices, faces, img_size=720):
    """Render mesh"""
    try:
        os.environ['PYOPENGL_PLATFORM'] = 'egl'
        
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        bounds = mesh.bounds
        center = bounds.mean(axis=0)
        mesh.vertices -= center
        scale = 2.0 / (bounds[1] - bounds[0]).max()
        mesh.vertices *= scale
        
        material = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=[0.8, 0.8, 0.8, 1.0],
            metallicFactor=0.0,
            roughnessFactor=0.7
        )
        py_mesh = pyrender.Mesh.from_trimesh(mesh, material=material)
        
        scene = pyrender.Scene(bg_color=[0.96, 0.96, 0.96, 1.0])
        scene.add(py_mesh)
        
        camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
        camera_pose = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 2.5],
            [0.0, 0.0, 0.0, 1.0]
        ])
        scene.add(camera, pose=camera_pose)
        
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
        
        renderer = pyrender.OffscreenRenderer(img_size, img_size)
        color, _ = renderer.render(scene)
        renderer.delete()
        
        return color, None
    except Exception as e:
        return None, str(e)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--npz_dir', required=True, help='Directory with NPZ files')
    parser.add_argument('--model_path', required=True, help='SMPL-X model directory')
    parser.add_argument('--output_dir', required=True, help='Output directory')
    parser.add_argument('--img_size', type=int, default=720)
    parser.add_argument('--use_fallback', action='store_true', help='Use last valid frame for missing')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    
    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print("Loading SMPL-X model...")
    use_pca = False
    try:
        model = smplx.create(
            model_path=args.model_path,
            model_type='smplx',
            gender='neutral',
            num_betas=10,
            num_expression_coeffs=10,
            use_pca=False,
            use_face_contour=True,
            ext='npz'
        ).to(device)
    except:
        print("Falling back to PCA hands...")
        model = smplx.create(
            model_path=args.model_path,
            model_type='smplx',
            gender='neutral',
            num_betas=10,
            num_expression_coeffs=10,
            use_pca=True,
            num_pca_comps=12,
            use_face_contour=True,
            ext='npz'
        ).to(device)
        use_pca = True
    
    print("✓ Model loaded")
    
    # Get all NPZ files
    npz_files = sorted(Path(args.npz_dir).glob("*.npz"))
    print(f"Found {len(npz_files)} NPZ files")
    
    # Process frames
    stats = {
        'total': len(npz_files),
        'success': 0,
        'fallback_used': 0,
        'failed': 0,
        'errors': []
    }
    
    last_valid_mesh = None
    
    for idx, npz_file in enumerate(tqdm(npz_files)):
        frame_name = npz_file.stem
        output_path = output_dir / f"{frame_name}.png"
        
        # Load parameters
        result, error = load_params_safe(str(npz_file), person_id=0, use_pca=use_pca)
        
        if error:
            # Try fallback
            if args.use_fallback and last_valid_mesh is not None:
                vertices, faces = last_valid_mesh
                img, err = render_mesh(vertices, faces, args.img_size)
                
                if err is None:
                    Image.fromarray(img).save(output_path)
                    stats['fallback_used'] += 1
                    stats['errors'].append({
                        'frame': frame_name,
                        'error': error,
                        'action': 'used_fallback'
                    })
                    continue
            
            # Failed completely
            stats['failed'] += 1
            stats['errors'].append({
                'frame': frame_name,
                'error': error,
                'action': 'skipped'
            })
            continue
        
        params, cam_trans = result
        
        # Reconstruct
        vertices, faces, err = reconstruct_mesh(model, params, cam_trans, device)
        if err:
            stats['failed'] += 1
            stats['errors'].append({
                'frame': frame_name,
                'error': f'reconstruct:{err}',
                'action': 'skipped'
            })
            continue
        
        # Render
        img, err = render_mesh(vertices, faces, args.img_size)
        if err:
            stats['failed'] += 1
            stats['errors'].append({
                'frame': frame_name,
                'error': f'render:{err}',
                'action': 'skipped'
            })
            continue
        
        # Save
        Image.fromarray(img).save(output_path)
        last_valid_mesh = (vertices, faces)
        stats['success'] += 1
    
    # Save report
    report_path = output_dir / 'render_report.json'
    with open(report_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    # Summary
    print("\n" + "="*70)
    print("RENDERING COMPLETE")
    print("="*70)
    print(f"Total frames: {stats['total']}")
    print(f"Success: {stats['success']} ({stats['success']/stats['total']*100:.1f}%)")
    print(f"Fallback used: {stats['fallback_used']}")
    print(f"Failed: {stats['failed']}")
    print(f"\nReport saved to: {report_path}")
    
    # Show error breakdown
    if stats['errors']:
        error_types = {}
        for err in stats['errors']:
            error_type = err['error'].split(':')[0]
            error_types[error_type] = error_types.get(error_type, 0) + 1
        
        print("\nError breakdown:")
        for err_type, count in sorted(error_types.items(), key=lambda x: -x[1]):
            print(f"  {err_type}: {count}")


if __name__ == '__main__':
    main()
