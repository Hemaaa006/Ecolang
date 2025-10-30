# BUILD_APP.md - SignMesh Application Build Guide

## Project Overview

**Name**: SignMesh  
**Purpose**: Web application for rendering SMPL-X 3D mesh reconstructions from sign language videos  
**Tech Stack**: Streamlit, PyTorch, SMPL-X, PyRender, Trimesh  
**Deployment Target**: Google Colab (Development), Cloud Server (Production)

## Application Description

SignMesh is a production-ready web application that:
1. Displays sign language videos in a dropdown selector
2. Shows original video on the left panel
3. Renders synchronized 3D SMPL-X mesh on the right panel
4. Handles missing person detections gracefully using fallback mechanism
5. Provides statistics and error reporting

## Critical Technical Requirements

### 1. Data Structure

**Input Data Location**: Google Drive (configurable)
```
SignMesh/
├── videos/
│   ├── video1_speaking.mp4
│   ├── video2_gestures.mp4
│   ├── video3_conversation.mp4
│   └── video4_demonstration.mp4
├── npz_files/
│   ├── video1_speaking/
│   │   ├── frame_0001_params.npz
│   │   ├── frame_0002_params.npz
│   │   └── ... (1800 frames total)
│   ├── video2_gestures/
│   ├── video3_conversation/
│   └── video4_demonstration/
└── models/
    └── SMPLX_NEUTRAL.npz
```

**NPZ File Structure** (CRITICAL - This is the actual format):
```python
# Keys in each NPZ file:
'image_path'                      # String, original image path
'image_size'                      # (2,) array
'person_ids'                      # (N,) array of person IDs - CHECK LENGTH FIRST!
'cfg_output_hm_shape'            # (3,) config
'cfg_input_img_shape'            # (2,) config

# Person-specific keys (only exist if person detected):
'person_0_smplx_root_pose'       # (3,) global orientation
'person_0_smplx_body_pose'       # (63,) body joints
'person_0_smplx_lhand_pose'      # (45,) left hand
'person_0_smplx_rhand_pose'      # (45,) right hand
'person_0_smplx_jaw_pose'        # (3,) jaw
'person_0_smplx_shape'           # (10,) betas
'person_0_smplx_expr'            # (10,) expression
'person_0_cam_trans'             # (3,) camera translation
'person_0_joint_img'             # (65, 3) 2D joints
'person_0_smplx_joint_proj'      # (137, 2) projected joints
'person_0_smplx_mesh_cam'        # (10475, 3) mesh in camera space
'person_0_lhand_bbox'            # (4,) bounding box
'person_0_rhand_bbox'            # (4,) bounding box
'person_0_face_bbox'             # (4,) bounding box
'person_0_bb2img_trans'          # (2, 3) transformation
'person_0_bbox_xywh'             # (4,) bbox coordinates
'person_0_focal'                 # (2,) focal length
'person_0_princpt'               # (2,) principal point
```

**CRITICAL**: The key format is `person_0_smplx_root_pose` NOT `person_0_smplx_root_pose` (underscore between each part)

### 2. The Missing Frame Problem (MOST IMPORTANT)

**Problem**: Some frames (5-15%) have NO person detection, causing KeyError:
```python
# This will fail for frames without person:
data['person_0_smplx_root_pose']  # KeyError!
```

**Root Cause**: When OSX doesn't detect a person, the `person_ids` array is empty or has length 0, and ALL `person_0_*` keys are missing from the NPZ file.

**Solution Strategy**:
```python
def check_frame_valid(npz_path):
    """ALWAYS check this first!"""
    data = np.load(npz_path, allow_pickle=True)
    person_ids = data.get('person_ids', np.array([]))
    
    if len(person_ids) == 0:
        return False, "no_person_detected"
    
    # Also verify required keys exist
    required = ['person_0_smplx_root_pose', 'person_0_smplx_body_pose', 
                'person_0_smplx_shape']
    for key in required:
        if key not in data.files:
            return False, f"missing_{key}"
    
    return True, None
```

**Fallback Mechanism** (REQUIRED):
```python
class MeshRenderer:
    def __init__(self):
        self.last_valid_mesh = None  # Cache last successful render
    
    def render_frame(self, npz_path, use_fallback=True):
        # Check if person exists
        is_valid, error = check_frame_valid(npz_path)
        
        if not is_valid:
            if use_fallback and self.last_valid_mesh:
                # Use cached mesh from previous frame
                return self.last_valid_mesh, f"fallback:{error}"
            else:
                return None, error
        
        # Normal rendering...
        mesh = self.reconstruct_and_render(npz_path)
        self.last_valid_mesh = mesh  # Cache for future fallbacks
        return mesh, None
```

### 3. SMPL-X Model Loading (Error-Prone)

**Problem**: SMPL-X models may be missing hand component data, causing AttributeError:
```
AttributeError: 'Struct' object has no attribute 'hands_componentsl'
```

**Solution**: Always try without PCA first, fallback to PCA mode:
```python
def init_model(model_path, device):
    try:
        # Try full model (no PCA)
        model = smplx.create(
            model_path=model_path,
            model_type='smplx',
            gender='neutral',
            num_betas=10,
            num_expression_coeffs=10,
            use_pca=False,
            use_face_contour=True,
            ext='npz'
        ).to(device)
        return model, False  # use_pca=False
        
    except AttributeError as e:
        # Fallback to PCA hands
        if 'hands_componentsl' in str(e):
            model = smplx.create(
                model_path=model_path,
                model_type='smplx',
                gender='neutral',
                num_betas=10,
                num_expression_coeffs=10,
                use_pca=True,
                num_pca_comps=12,
                use_face_contour=True,
                ext='npz'
            ).to(device)
            return model, True  # use_pca=True
        else:
            raise
```

**Impact on Parameters**:
```python
if use_pca:
    # Hand poses are 12D (PCA space)
    params['left_hand_pose'] = np.zeros((1, 12), dtype=np.float32)
    params['right_hand_pose'] = np.zeros((1, 12), dtype=np.float32)
else:
    # Hand poses are 45D (full)
    params['left_hand_pose'] = data['person_0_smplx_lhand_pose'].reshape(1, -1)
    params['right_hand_pose'] = data['person_0_smplx_rhand_pose'].reshape(1, -1)
```

### 4. Complete Parameter Extraction Function

```python
def load_params_from_npz(npz_path, person_id=0, use_pca=False):
    """
    Load SMPL-X parameters with complete error handling
    
    Returns: ((params_dict, cam_trans), error_msg)
    """
    try:
        data = np.load(npz_path, allow_pickle=True)
        
        # CRITICAL: Check person exists
        person_ids = data.get('person_ids', np.array([]))
        if len(person_ids) == 0:
            return None, "no_person_detected"
        
        if person_id >= len(person_ids):
            return None, f"person_id_{person_id}_not_found"
        
        # Extract with exact key names
        prefix = f'person_{person_id}_smplx_'
        
        params = {
            'global_orient': data[prefix + 'root_pose'].reshape(1, 3).astype(np.float32),
            'body_pose': data[prefix + 'body_pose'].reshape(1, -1).astype(np.float32),
            'jaw_pose': data[prefix + 'jaw_pose'].reshape(1, 3).astype(np.float32),
            'betas': data[prefix + 'shape'].reshape(1, -1).astype(np.float32),
            'expression': data[prefix + 'expr'].reshape(1, -1).astype(np.float32),
            'leye_pose': np.zeros((1, 3), dtype=np.float32),
            'reye_pose': np.zeros((1, 3), dtype=np.float32)
        }
        
        # Handle hand poses based on PCA mode
        if use_pca:
            params['left_hand_pose'] = np.zeros((1, 12), dtype=np.float32)
            params['right_hand_pose'] = np.zeros((1, 12), dtype=np.float32)
        else:
            params['left_hand_pose'] = data[prefix + 'lhand_pose'].reshape(1, -1).astype(np.float32)
            params['right_hand_pose'] = data[prefix + 'rhand_pose'].reshape(1, -1).astype(np.float32)
        
        # Get camera translation
        cam_trans = data.get(f'person_{person_id}_cam_trans')
        
        return (params, cam_trans), None
        
    except KeyError as e:
        return None, f"missing_key:{str(e)}"
    except Exception as e:
        return None, f"error:{str(e)}"
```

### 5. Mesh Reconstruction Function

```python
def reconstruct_mesh(model, params, cam_trans, device):
    """
    Reconstruct 3D mesh from SMPL-X parameters
    
    Returns: (vertices, faces, error_msg)
    """
    try:
        with torch.no_grad():
            # Convert to tensors
            tensors = {k: torch.from_numpy(v).float().to(device) 
                      for k, v in params.items()}
            
            # SMPL-X forward pass
            output = model(**tensors)
            vertices = output.vertices[0].cpu().numpy()
            faces = model.faces
        
        # Apply camera translation if available
        if cam_trans is not None:
            vertices += cam_trans
        
        return vertices, faces, None
        
    except Exception as e:
        return None, None, f"reconstruction_error:{str(e)}"
```

### 6. Rendering Function

```python
def render_mesh(vertices, faces, img_size=720, bg_color=(245, 245, 245)):
    """
    Render mesh to image
    
    Returns: (image_array, error_msg)
    """
    try:
        # Set headless rendering
        os.environ['PYOPENGL_PLATFORM'] = 'egl'
        
        # Create trimesh
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        
        # Center and scale
        bounds = mesh.bounds
        center = bounds.mean(axis=0)
        mesh.vertices -= center
        scale = 2.0 / (bounds[1] - bounds[0]).max()
        mesh.vertices *= scale
        
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
        
        # Add lights (2 directional lights)
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
        color, _ = renderer.render(scene)
        renderer.delete()
        
        return color, None
        
    except Exception as e:
        return None, f"render_error:{str(e)}"
```

## File Structure

Create this exact structure:

```
signmesh/
├── app.py                      # Main Streamlit application
├── mesh_renderer.py            # MeshRenderer class
├── file_manager.py             # File/path management
├── batch_render.py             # Batch processing script
├── config.py                   # Configuration
├── requirements.txt            # Dependencies
├── README.md                   # Documentation
├── .streamlit/
│   └── config.toml            # Streamlit config
└── assets/
    └── styles.css             # Custom CSS (optional)
```

## Detailed File Specifications

### 1. requirements.txt

```txt
streamlit>=1.28.0
torch>=2.0.0
torchvision>=0.15.0
smplx>=0.1.28
trimesh>=3.20.0
pyrender>=0.1.45
Pillow>=9.0.0
numpy>=1.21.0
opencv-python>=4.7.0
tqdm>=4.65.0
PyOpenGL>=3.1.0
PyOpenGL-accelerate>=3.1.0
```

### 2. config.py

```python
"""
Configuration for SignMesh application
Automatically detects environment (Colab, local, production)
"""
import os
from pathlib import Path

# Detect environment
IS_COLAB = os.path.exists('/content')
IS_PRODUCTION = os.environ.get('ENV') == 'production'

# Base paths
if IS_COLAB:
    # Google Colab with mounted Drive
    BASE_PATH = "/content/drive/MyDrive/SignMesh"
elif IS_PRODUCTION:
    # Production server
    BASE_PATH = "/app/data"
else:
    # Local development
    BASE_PATH = "./data"

# Data directories
VIDEOS_DIR = os.path.join(BASE_PATH, "videos")
NPZ_DIR = os.path.join(BASE_PATH, "npz_files")
MODEL_PATH = os.path.join(BASE_PATH, "models")
CACHE_DIR = os.path.join(BASE_PATH, "cache")

# Rendering settings
DEFAULT_IMG_SIZE = 720
DEFAULT_BG_COLOR = (245, 245, 245)  # Light gray
DEFAULT_DEVICE = 'cuda' if IS_PRODUCTION else 'cpu'

# Video library
VIDEO_LIBRARY = {
    'video1_speaking': {
        'title': 'Video 1 - Speaking',
        'filename': 'video1_speaking.mp4',
        'duration': '1:00',
        'fps': 30,
        'frames': 1800
    },
    'video2_gestures': {
        'title': 'Video 2 - Gestures',
        'filename': 'video2_gestures.mp4',
        'duration': '1:00',
        'fps': 30,
        'frames': 1800
    },
    'video3_conversation': {
        'title': 'Video 3 - Conversation',
        'filename': 'video3_conversation.mp4',
        'duration': '1:00',
        'fps': 30,
        'frames': 1800
    },
    'video4_demonstration': {
        'title': 'Video 4 - Demonstration',
        'filename': 'video4_demonstration.mp4',
        'duration': '1:00',
        'fps': 30,
        'frames': 1800
    }
}

# Performance settings
CACHE_SIZE = 30  # Number of frames to cache
BATCH_SIZE = 10  # For batch processing
```

### 3. file_manager.py

```python
"""
File management for SignMesh
Handles paths and file access across different environments
"""
import os
from pathlib import Path
import config

class FileManager:
    """Manages file paths for videos and NPZ files"""
    
    def __init__(self):
        self.base_path = Path(config.BASE_PATH)
        self.videos_dir = Path(config.VIDEOS_DIR)
        self.npz_dir = Path(config.NPZ_DIR)
        self.model_path = config.MODEL_PATH
        
    def get_video_path(self, video_id):
        """Get path to video file"""
        video_info = config.VIDEO_LIBRARY.get(video_id)
        if not video_info:
            return None
        
        filename = video_info['filename']
        return str(self.videos_dir / filename)
    
    def get_npz_dir(self, video_id):
        """Get directory containing NPZ files for video"""
        return str(self.npz_dir / video_id)
    
    def get_npz_path(self, video_id, frame_num):
        """Get path to specific NPZ file"""
        npz_dir = self.get_npz_dir(video_id)
        return os.path.join(npz_dir, f"frame_{frame_num:04d}_params.npz")
    
    def check_video_exists(self, video_id):
        """Check if video file exists"""
        path = self.get_video_path(video_id)
        return path and os.path.exists(path)
    
    def check_npz_exists(self, video_id, frame_num):
        """Check if NPZ file exists"""
        path = self.get_npz_path(video_id, frame_num)
        return os.path.exists(path)
    
    def get_video_stats(self, video_id):
        """Get statistics about video's NPZ files"""
        npz_dir = self.get_npz_dir(video_id)
        
        if not os.path.exists(npz_dir):
            return {
                'exists': False,
                'total_frames': 0,
                'valid_frames': 0,
                'missing_frames': 0
            }
        
        npz_files = list(Path(npz_dir).glob("*.npz"))
        total_frames = len(npz_files)
        
        # Count valid frames (with person detection)
        valid_count = 0
        for npz_file in npz_files:
            try:
                import numpy as np
                data = np.load(npz_file, allow_pickle=True)
                person_ids = data.get('person_ids', np.array([]))
                if len(person_ids) > 0:
                    valid_count += 1
            except:
                pass
        
        expected = config.VIDEO_LIBRARY[video_id]['frames']
        
        return {
            'exists': True,
            'total_frames': total_frames,
            'valid_frames': valid_count,
            'missing_frames': expected - valid_count,
            'expected_frames': expected
        }
```

### 4. mesh_renderer.py

```python
"""
SMPL-X mesh renderer with error handling and fallback mechanism
"""
import numpy as np
import torch
import trimesh
import pyrender
import smplx
import os
from PIL import Image

class MeshRenderer:
    """Handles SMPL-X mesh reconstruction and rendering"""
    
    def __init__(self, model_path, device='cpu', img_size=720):
        self.model_path = model_path
        self.device = torch.device(device)
        self.img_size = img_size
        self.model = None
        self.use_pca = False
        self.last_valid_mesh = None  # For fallback
        
        # Initialize model
        self._init_model()
    
    def _init_model(self):
        """Initialize SMPL-X model with fallback to PCA"""
        try:
            # Try without PCA first
            self.model = smplx.create(
                model_path=self.model_path,
                model_type='smplx',
                gender='neutral',
                num_betas=10,
                num_expression_coeffs=10,
                use_pca=False,
                use_face_contour=True,
                ext='npz'
            ).to(self.device)
            self.use_pca = False
            
        except AttributeError as e:
            if 'hands_componentsl' in str(e):
                # Fallback to PCA
                self.model = smplx.create(
                    model_path=self.model_path,
                    model_type='smplx',
                    gender='neutral',
                    num_betas=10,
                    num_expression_coeffs=10,
                    use_pca=True,
                    num_pca_comps=12,
                    use_face_contour=True,
                    ext='npz'
                ).to(self.device)
                self.use_pca = True
            else:
                raise
    
    def check_frame_valid(self, npz_path, person_id=0):
        """Check if frame has valid person detection"""
        try:
            data = np.load(npz_path, allow_pickle=True)
            person_ids = data.get('person_ids', np.array([]))
            
            if len(person_ids) == 0:
                return False, "no_person_detected"
            
            # Check required keys exist
            prefix = f'person_{person_id}_smplx_'
            required = ['root_pose', 'body_pose', 'shape']
            
            for key in required:
                if prefix + key not in data.files:
                    return False, f"missing_{key}"
            
            return True, None
            
        except Exception as e:
            return False, str(e)
    
    def load_params(self, npz_path, person_id=0):
        """Load SMPL-X parameters from NPZ"""
        try:
            data = np.load(npz_path, allow_pickle=True)
            
            # Check person exists
            person_ids = data.get('person_ids', np.array([]))
            if len(person_ids) == 0:
                return None, "no_person_detected"
            
            prefix = f'person_{person_id}_smplx_'
            
            # Extract parameters
            params = {
                'global_orient': data[prefix + 'root_pose'].reshape(1, 3).astype(np.float32),
                'body_pose': data[prefix + 'body_pose'].reshape(1, -1).astype(np.float32),
                'jaw_pose': data[prefix + 'jaw_pose'].reshape(1, 3).astype(np.float32),
                'betas': data[prefix + 'shape'].reshape(1, -1).astype(np.float32),
                'expression': data[prefix + 'expr'].reshape(1, -1).astype(np.float32),
                'leye_pose': np.zeros((1, 3), dtype=np.float32),
                'reye_pose': np.zeros((1, 3), dtype=np.float32)
            }
            
            # Hand poses (depends on PCA mode)
            if self.use_pca:
                params['left_hand_pose'] = np.zeros((1, 12), dtype=np.float32)
                params['right_hand_pose'] = np.zeros((1, 12), dtype=np.float32)
            else:
                params['left_hand_pose'] = data[prefix + 'lhand_pose'].reshape(1, -1).astype(np.float32)
                params['right_hand_pose'] = data[prefix + 'rhand_pose'].reshape(1, -1).astype(np.float32)
            
            # Camera translation
            cam_trans = data.get(f'person_{person_id}_cam_trans')
            
            return (params, cam_trans), None
            
        except KeyError as e:
            return None, f"missing_key:{str(e)}"
        except Exception as e:
            return None, f"error:{str(e)}"
    
    def reconstruct_mesh(self, params, cam_trans):
        """Reconstruct 3D mesh from parameters"""
        try:
            with torch.no_grad():
                tensors = {k: torch.from_numpy(v).float().to(self.device) 
                          for k, v in params.items()}
                output = self.model(**tensors)
                vertices = output.vertices[0].cpu().numpy()
                faces = self.model.faces
            
            if cam_trans is not None:
                vertices += cam_trans
            
            return vertices, faces, None
            
        except Exception as e:
            return None, None, f"reconstruction_error:{str(e)}"
    
    def render_mesh(self, vertices, faces, bg_color=(245, 245, 245)):
        """Render mesh to image"""
        try:
            os.environ['PYOPENGL_PLATFORM'] = 'egl'
            
            # Create mesh
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            
            # Center and scale
            bounds = mesh.bounds
            center = bounds.mean(axis=0)
            mesh.vertices -= center
            scale = 2.0 / (bounds[1] - bounds[0]).max()
            mesh.vertices *= scale
            
            # Material
            material = pyrender.MetallicRoughnessMaterial(
                baseColorFactor=[0.8, 0.8, 0.8, 1.0],
                metallicFactor=0.0,
                roughnessFactor=0.7
            )
            py_mesh = pyrender.Mesh.from_trimesh(mesh, material=material)
            
            # Scene
            scene = pyrender.Scene(bg_color=[c/255.0 for c in bg_color] + [1.0])
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
            renderer = pyrender.OffscreenRenderer(self.img_size, self.img_size)
            color, _ = renderer.render(scene)
            renderer.delete()
            
            return color, None
            
        except Exception as e:
            return None, f"render_error:{str(e)}"
    
    def render_frame(self, npz_path, person_id=0, use_fallback=True):
        """
        Complete pipeline: load → reconstruct → render
        
        Returns: (image, status_message)
        status can be: "success", "fallback:reason", "error:reason"
        """
        # Load parameters
        result, error = self.load_params(npz_path, person_id)
        
        if error:
            # Try fallback
            if use_fallback and self.last_valid_mesh:
                vertices, faces = self.last_valid_mesh
                img, err = self.render_mesh(vertices, faces)
                if err is None:
                    return img, f"fallback:{error}"
            return None, f"error:{error}"
        
        params, cam_trans = result
        
        # Reconstruct
        vertices, faces, error = self.reconstruct_mesh(params, cam_trans)
        
        if error:
            if use_fallback and self.last_valid_mesh:
                vertices, faces = self.last_valid_mesh
                img, err = self.render_mesh(vertices, faces)
                if err is None:
                    return img, f"fallback:{error}"
            return None, f"error:{error}"
        
        # Render
        img, error = self.render_mesh(vertices, faces)
        
        if error:
            return None, f"error:{error}"
        
        # Cache for future fallbacks
        self.last_valid_mesh = (vertices, faces)
        
        return img, "success"
```

### 5. app.py (Main Streamlit Application)

```python
"""
SignMesh - Main Streamlit Application
Video to 3D Mesh Rendering for Sign Language
"""
import streamlit as st
import numpy as np
from pathlib import Path
import os

# Local imports
import config
from file_manager import FileManager
from mesh_renderer import MeshRenderer

# Page configuration
st.set_page_config(
    page_title="SignMesh - Video to Mesh Rendering",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #007BFF;
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 10px;
    }
    .sub-header {
        text-align: center;
        color: #6C757D;
        font-size: 20px;
        margin-bottom: 40px;
    }
    .video-panel {
        background: white;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .stats-box {
        background: #F8F9FA;
        border-radius: 8px;
        padding: 15px;
        border-left: 4px solid #007BFF;
    }
    .success-msg {
        background: #D4EDDA;
        color: #155724;
        padding: 10px;
        border-radius: 4px;
        border: 1px solid #C3E6CB;
    }
    .warning-msg {
        background: #FFF3CD;
        color: #856404;
        padding: 10px;
        border-radius: 4px;
        border: 1px solid #FFEEBA;
    }
    .error-msg {
        background: #F8D7DA;
        color: #721C24;
        padding: 10px;
        border-radius: 4px;
        border: 1px solid #F5C6CB;
    }
</style>
""", unsafe_allow_html=True)

# Initialize managers
@st.cache_resource
def get_file_manager():
    return FileManager()

@st.cache_resource
def get_renderer(model_path, device):
    return MeshRenderer(model_path, device=device)

# Main app
def main():
    # Header
    st.markdown('<h1 class="main-header">SignMesh</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Video to 3D Mesh Rendering for Sign Language</p>', unsafe_allow_html=True)
    
    # Initialize managers
    file_mgr = get_file_manager()
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Show data source
        st.subheader("📁 Data Source")
        st.text(config.BASE_PATH)
        
        # Renderer settings
        st.subheader("🎨 Renderer Settings")
        
        device = st.selectbox("Device", ["cuda", "cpu"], 
                             index=0 if config.DEFAULT_DEVICE == 'cuda' else 1)
        
        img_size = st.slider("Image Size", 256, 1024, config.DEFAULT_IMG_SIZE, 64)
        
        use_fallback = st.checkbox("Enable Fallback", value=True,
                                   help="Use previous frame when person not detected")
        
        # Initialize renderer button
        if st.button("🔄 Initialize Renderer", type="primary"):
            with st.spinner("Loading SMPL-X model..."):
                try:
                    st.session_state.renderer = get_renderer(config.MODEL_PATH, device)
                    st.success("✅ Renderer initialized!")
                    if st.session_state.renderer.use_pca:
                        st.info("ℹ️ Using PCA hand model")
                except Exception as e:
                    st.error(f"❌ Failed: {e}")
        
        st.divider()
        
        # Statistics
        if st.button("📊 Scan Dataset"):
            with st.spinner("Scanning..."):
                for video_id in config.VIDEO_LIBRARY.keys():
                    stats = file_mgr.get_video_stats(video_id)
                    with st.expander(f"📹 {video_id}"):
                        if stats['exists']:
                            st.metric("Total Frames", stats['total_frames'])
                            st.metric("Valid Frames", stats['valid_frames'])
                            st.metric("Missing", stats['missing_frames'])
                        else:
                            st.warning("NPZ files not found")
    
    # Main content
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="video-panel">', unsafe_allow_html=True)
        st.subheader("📹 Original Video")
        
        # Video selector
        video_options = {k: v['title'] for k, v in config.VIDEO_LIBRARY.items()}
        selected_video = st.selectbox(
            "Select Video",
            options=list(video_options.keys()),
            format_func=lambda x: video_options[x]
        )
        
        # Display video
        video_path = file_mgr.get_video_path(selected_video)
        
        if video_path and os.path.exists(video_path):
            st.video(video_path)
        else:
            st.error(f"❌ Video not found: {video_path}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="video-panel">', unsafe_allow_html=True)
        st.subheader("🎨 Mesh Render")
        
        # Check if renderer is initialized
        if 'renderer' not in st.session_state:
            st.warning("⚠️ Please initialize the renderer in the sidebar")
        else:
            # Frame selector
            max_frames = config.VIDEO_LIBRARY[selected_video]['frames']
            frame_num = st.number_input(
                "Frame Number",
                min_value=1,
                max_value=max_frames,
                value=1,
                step=1
            )
            
            # Render button
            if st.button("🎬 Render Frame", type="primary"):
                npz_path = file_mgr.get_npz_path(selected_video, frame_num)
                
                if not os.path.exists(npz_path):
                    st.error(f"❌ NPZ file not found: frame_{frame_num:04d}")
                else:
                    with st.spinner("Rendering mesh..."):
                        img, status = st.session_state.renderer.render_frame(
                            npz_path,
                            person_id=0,
                            use_fallback=use_fallback
                        )
                        
                        # Display result
                        if img is not None:
                            st.image(img, use_column_width=True)
                            
                            # Status message
                            if status == "success":
                                st.markdown('<div class="success-msg">✅ Rendered successfully</div>', 
                                          unsafe_allow_html=True)
                            elif status.startswith("fallback"):
                                reason = status.split(":", 1)[1]
                                st.markdown(f'<div class="warning-msg">⚠️ Using fallback: {reason}</div>', 
                                          unsafe_allow_html=True)
                        else:
                            error = status.split(":", 1)[1] if ":" in status else status
                            st.markdown(f'<div class="error-msg">❌ Render failed: {error}</div>', 
                                      unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.divider()
    st.caption("SignMesh v1.0 - SMPL-X Mesh Rendering for Sign Language Videos")

if __name__ == "__main__":
    main()
```

### 6. batch_render.py (for pre-rendering all frames)

```python
"""
Batch render all frames for a video
"""
import argparse
from pathlib import Path
from tqdm import tqdm
import json
import os

# Must have mesh_renderer, file_manager, config accessible
from mesh_renderer import MeshRenderer
from file_manager import FileManager
import config
from PIL import Image

def main():
    parser = argparse.ArgumentParser(description='Batch render all frames')
    parser.add_argument('--video_id', required=True, help='Video ID to render')
    parser.add_argument('--output_dir', required=True, help='Output directory')
    parser.add_argument('--device', default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--use_fallback', action='store_true', help='Use fallback for missing frames')
    args = parser.parse_args()
    
    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    file_mgr = FileManager()
    renderer = MeshRenderer(config.MODEL_PATH, device=args.device)
    
    # Get video info
    if args.video_id not in config.VIDEO_LIBRARY:
        print(f"Error: Video ID '{args.video_id}' not found")
        return
    
    video_info = config.VIDEO_LIBRARY[args.video_id]
    total_frames = video_info['frames']
    
    print(f"Rendering {total_frames} frames for {args.video_id}")
    print(f"Output: {output_dir}")
    
    # Statistics
    stats = {
        'total': total_frames,
        'success': 0,
        'fallback': 0,
        'failed': 0,
        'errors': []
    }
    
    # Process frames
    for frame_num in tqdm(range(1, total_frames + 1)):
        npz_path = file_mgr.get_npz_path(args.video_id, frame_num)
        output_path = output_dir / f"mesh_{frame_num:04d}.png"
        
        if not os.path.exists(npz_path):
            stats['failed'] += 1
            stats['errors'].append({
                'frame': frame_num,
                'error': 'npz_not_found'
            })
            continue
        
        # Render
        img, status = renderer.render_frame(npz_path, use_fallback=args.use_fallback)
        
        if img is None:
            stats['failed'] += 1
            stats['errors'].append({
                'frame': frame_num,
                'error': status
            })
            continue
        
        # Save
        Image.fromarray(img).save(output_path)
        
        if status == "success":
            stats['success'] += 1
        elif status.startswith("fallback"):
            stats['fallback'] += 1
        else:
            stats['failed'] += 1
    
    # Save report
    report_path = output_dir / 'render_report.json'
    with open(report_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    # Print summary
    print("\n" + "="*70)
    print("RENDERING COMPLETE")
    print("="*70)
    print(f"Total: {stats['total']}")
    print(f"Success: {stats['success']} ({stats['success']/stats['total']*100:.1f}%)")
    print(f"Fallback: {stats['fallback']}")
    print(f"Failed: {stats['failed']}")
    print(f"\nReport: {report_path}")

if __name__ == '__main__':
    main()
```

### 7. .streamlit/config.toml

```toml
[theme]
primaryColor = "#007BFF"
backgroundColor = "#F8F9FA"
secondaryBackgroundColor = "#FFFFFF"
textColor = "#212529"
font = "sans serif"

[server]
maxUploadSize = 500
enableCORS = false
enableXsrfProtection = true

[browser]
gatherUsageStats = false
```

### 8. README.md

```markdown
# SignMesh - Video to Mesh Rendering

SMPL-X mesh reconstruction from sign language videos with real-time rendering.

## Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Data Paths

Edit `config.py` or set environment variables:
```python
BASE_PATH = "/path/to/your/data"
```

### 3. Data Structure
```
SignMesh/
├── videos/
├── npz_files/
│   ├── video1_speaking/
│   ├── video2_gestures/
│   └── ...
└── models/
    └── SMPLX_NEUTRAL.npz
```

### 4. Run Application
```bash
streamlit run app.py
```

## Usage

1. Initialize renderer in sidebar
2. Select video from dropdown
3. Choose frame number
4. Click "Render Frame"

## Batch Processing

Pre-render all frames:
```bash
python batch_render.py \
  --video_id video1_speaking \
  --output_dir ./rendered \
  --device cuda \
  --use_fallback
```

## Troubleshooting

### "No person detected"
- Frame has no person detection
- Enable fallback to use previous frame

### "Missing hand components"
- SMPL-X model incomplete
- App automatically falls back to PCA mode

## License
MIT
```

## Critical Implementation Notes for Claude Code

### 1. **ERROR HANDLING IS CRITICAL**
- ALWAYS check `person_ids` length before accessing any `person_0_*` keys
- Use try-except blocks for ALL NPZ loading operations
- Implement fallback mechanism (use last valid mesh)
- Never crash on missing frames

### 2. **Key Names are EXACT**
- `person_0_smplx_root_pose` (not `person_0_smplx_root_pose`)
- Always use string formatting: `f'person_{person_id}_smplx_{param_name}'`

### 3. **PCA Fallback is REQUIRED**
- Try `use_pca=False` first
- Catch `AttributeError` with 'hands_componentsl' in message
- Fall back to `use_pca=True` with 12 components
- Adjust hand pose dimensions accordingly

### 4. **Caching for Performance**
- Use `@st.cache_resource` for renderer initialization
- Cache last 30 rendered frames for smooth scrubbing
- Store last valid mesh for fallback

### 5. **User Feedback**
- Show clear status messages (success/fallback/error)
- Use color-coded notifications (green/yellow/red)
- Display statistics (success rate, fallback usage)

### 6. **Testing Edge Cases**
- First frame has no detection
- Multiple consecutive missing frames
- All frames missing (should fail gracefully)
- Corrupt NPZ files
- Missing video files

## Build Commands

```bash
# 1. Create directory structure
mkdir -p signmesh/{.streamlit,assets}
cd signmesh

# 2. Create all files using the specifications above

# 3. Install dependencies
pip install -r requirements.txt

# 4. Test configuration
python -c "import config; print(config.BASE_PATH)"

# 5. Run application
streamlit run app.py
```

## Expected Behavior

### On Successful Render:
- Display mesh image (720x720)
- Show green success message
- Cache mesh for fallback

### On Missing Person:
- Use fallback if enabled (show yellow warning)
- Show error if fallback disabled (red error)
- Never crash

### On Other Errors:
- Display specific error message
- Log to console for debugging
- Use fallback if available

## Performance Targets

- Render time: < 200ms per frame (GPU)
- UI responsiveness: < 50ms
- Memory usage: < 2GB for renderer
- Cache size: Last 30 frames (~600MB)

## Success Criteria

✅ Clean UI matching design specifications  
✅ Smooth video-mesh synchronization  
✅ Handles 5-15% missing frames gracefully  
✅ 90%+ success rate with fallback enabled  
✅ Clear error messages and user feedback  
✅ Production-ready error handling  
✅ Comprehensive documentation

This document contains EVERYTHING Claude Code needs to build the application correctly!
