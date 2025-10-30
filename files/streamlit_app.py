"""
Streamlit Video-to-Mesh Renderer
Handles missing frames and provides real-time mesh rendering
"""

import streamlit as st
import numpy as np
import torch
import trimesh
import pyrender
from PIL import Image
import smplx
import os
from pathlib import Path
import io
import base64

# Page config
st.set_page_config(
    page_title="ECOLANG - Video to Mesh Rendering",
    page_icon="🎬",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #007BFF;
        font-size: 36px;
        font-weight: 700;
        margin-bottom: 10px;
    }
    .sub-header {
        text-align: center;
        color: #6C757D;
        font-size: 18px;
        margin-bottom: 30px;
    }
    .video-container {
        background: white;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    .error-message {
        background: #FFF3CD;
        color: #856404;
        padding: 10px;
        border-radius: 4px;
        border: 1px solid #FFEEBA;
    }
    .success-message {
        background: #D4EDDA;
        color: #155724;
        padding: 10px;
        border-radius: 4px;
        border: 1px solid #C3E6CB;
    }
</style>
""", unsafe_allow_html=True)


class MeshRenderer:
    """Handles SMPL-X mesh reconstruction and rendering with error handling"""
    
    def __init__(self, model_path, device='cpu'):
        """Initialize the renderer"""
        self.device = torch.device(device)
        self.model_path = model_path
        self.model = None
        self.use_pca = False
        self.last_valid_mesh = None  # Cache for fallback
        
        # Try to load model
        self._init_model()
    
    def _init_model(self):
        """Initialize SMPL-X model with fallback options"""
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
            
        except Exception as e:
            # Fallback to PCA if hand components missing
            try:
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
                st.warning("Using PCA hand model (hand poses approximated)")
                
            except Exception as e2:
                st.error(f"Failed to load SMPL-X model: {e2}")
                raise
    
    def check_frame_valid(self, npz_path, person_id=0):
        """Check if NPZ file has valid SMPL-X parameters"""
        try:
            data = np.load(npz_path, allow_pickle=True)
            
            # Check if person exists
            person_ids = data.get('person_ids', np.array([]))
            if len(person_ids) == 0:
                return False, "No person detected"
            
            # Check if required keys exist
            prefix = f'person_{person_id}_smplx_'
            required_keys = ['root_pose', 'body_pose', 'shape']
            
            for key in required_keys:
                if prefix + key not in data.files:
                    return False, f"Missing {key}"
            
            return True, "Valid"
            
        except Exception as e:
            return False, str(e)
    
    def load_params_from_npz(self, npz_path, person_id=0):
        """Load SMPL-X parameters from NPZ with comprehensive error handling"""
        try:
            data = np.load(npz_path, allow_pickle=True)
            
            # Check if person exists
            person_ids = data.get('person_ids', np.array([]))
            if len(person_ids) == 0:
                return None, "No person detected in frame"
            
            if person_id >= len(person_ids):
                return None, f"Person ID {person_id} not found"
            
            prefix = f'person_{person_id}_smplx_'
            
            # Try to extract parameters
            try:
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
                if self.use_pca:
                    params['left_hand_pose'] = np.zeros((1, 12), dtype=np.float32)
                    params['right_hand_pose'] = np.zeros((1, 12), dtype=np.float32)
                else:
                    params['left_hand_pose'] = data[prefix + 'lhand_pose'].reshape(1, -1).astype(np.float32)
                    params['right_hand_pose'] = data[prefix + 'rhand_pose'].reshape(1, -1).astype(np.float32)
                
                # Get camera translation if available
                cam_trans = None
                if f'person_{person_id}_cam_trans' in data.files:
                    cam_trans = data[f'person_{person_id}_cam_trans']
                
                return (params, cam_trans), None
                
            except KeyError as e:
                return None, f"Missing parameter: {e}"
                
        except Exception as e:
            return None, f"Error loading NPZ: {e}"
    
    def reconstruct_mesh(self, params, cam_trans=None):
        """Reconstruct 3D mesh from parameters"""
        try:
            with torch.no_grad():
                tensors = {k: torch.from_numpy(v).float().to(self.device) 
                          for k, v in params.items()}
                
                output = self.model(**tensors)
                vertices = output.vertices[0].cpu().numpy()
                faces = self.model.faces
            
            # Apply camera translation
            if cam_trans is not None:
                vertices += cam_trans
            
            # Cache as last valid mesh
            self.last_valid_mesh = (vertices, faces)
            
            return vertices, faces, None
            
        except Exception as e:
            return None, None, f"Reconstruction error: {e}"
    
    def render_mesh(self, vertices, faces, img_size=720, bg_color=(245, 245, 245)):
        """Render mesh to image"""
        try:
            os.environ['PYOPENGL_PLATFORM'] = 'egl'
            
            # Create and process mesh
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            
            # Center and scale
            bounds = mesh.bounds
            center = bounds.mean(axis=0)
            mesh.vertices -= center
            scale = 2.0 / (bounds[1] - bounds[0]).max()
            mesh.vertices *= scale
            
            # Create pyrender mesh
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
            renderer = pyrender.OffscreenRenderer(img_size, img_size)
            color, _ = renderer.render(scene)
            renderer.delete()
            
            return color, None
            
        except Exception as e:
            return None, f"Rendering error: {e}"
    
    def render_frame(self, npz_path, person_id=0, img_size=720, use_fallback=True):
        """
        Complete pipeline: load → reconstruct → render
        
        Args:
            npz_path: Path to NPZ file
            person_id: Person ID to render
            img_size: Output image size
            use_fallback: If True, use last valid mesh when frame fails
            
        Returns:
            (image, error_message)
        """
        # Load parameters
        result = self.load_params_from_npz(npz_path, person_id)
        if result[1] is not None:  # Error
            if use_fallback and self.last_valid_mesh is not None:
                # Use cached mesh
                vertices, faces = self.last_valid_mesh
                img, err = self.render_mesh(vertices, faces, img_size)
                if err is None:
                    return img, f"Using previous frame (current frame: {result[1]})"
            return None, result[1]
        
        params, cam_trans = result[0]
        
        # Reconstruct mesh
        vertices, faces, err = self.reconstruct_mesh(params, cam_trans)
        if err is not None:
            if use_fallback and self.last_valid_mesh is not None:
                vertices, faces = self.last_valid_mesh
                img, err2 = self.render_mesh(vertices, faces, img_size)
                if err2 is None:
                    return img, f"Using previous frame ({err})"
            return None, err
        
        # Render
        img, err = self.render_mesh(vertices, faces, img_size)
        if err is not None:
            return None, err
        
        return img, None


def create_placeholder_image(width, height, text):
    """Create placeholder image for missing frames"""
    img = np.ones((height, width, 3), dtype=np.uint8) * 240
    return img


def scan_npz_directory(npz_dir):
    """Scan directory and categorize frames"""
    npz_files = sorted(Path(npz_dir).glob("*.npz"))
    
    valid_frames = []
    invalid_frames = []
    
    for npz_file in npz_files:
        try:
            data = np.load(npz_file, allow_pickle=True)
            person_ids = data.get('person_ids', np.array([]))
            
            if len(person_ids) > 0:
                valid_frames.append(str(npz_file))
            else:
                invalid_frames.append((str(npz_file), "No person detected"))
        except Exception as e:
            invalid_frames.append((str(npz_file), str(e)))
    
    return valid_frames, invalid_frames


# Main Streamlit App
def main():
    st.markdown('<h1 class="main-header">ECOLANG - Video to Mesh Rendering</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Real-time SMPL-X mesh reconstruction from video frames</p>', unsafe_allow_html=True)
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        model_path = st.text_input("SMPL-X Model Path", value="/content")
        npz_dir = st.text_input("NPZ Directory", value="/content/npz_frames")
        
        img_size = st.slider("Image Size", 256, 1024, 720, 64)
        use_fallback = st.checkbox("Use Fallback (repeat last valid frame)", value=True)
        
        device = st.selectbox("Device", ["cpu", "cuda"])
        
        if st.button("🔄 Initialize Renderer"):
            with st.spinner("Loading SMPL-X model..."):
                try:
                    st.session_state.renderer = MeshRenderer(model_path, device)
                    st.success("✅ Renderer initialized!")
                except Exception as e:
                    st.error(f"❌ Failed to initialize: {e}")
    
    # Main content
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="video-container">', unsafe_allow_html=True)
        st.subheader("📹 Original Video")
        
        video_file = st.file_uploader("Upload Video", type=['mp4', 'avi', 'mov'])
        
        if video_file:
            st.video(video_file)
        else:
            st.info("Upload a video file to begin")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="video-container">', unsafe_allow_html=True)
        st.subheader("🎨 Mesh Render")
        
        if 'renderer' not in st.session_state:
            st.warning("⚠️ Please initialize the renderer in the sidebar first")
        else:
            # Frame selector
            frame_num = st.number_input("Frame Number", min_value=1, max_value=1800, value=1)
            
            npz_file = Path(npz_dir) / f"frame_{frame_num:04d}_params.npz"
            
            if st.button("🎬 Render Frame"):
                if not npz_file.exists():
                    st.error(f"❌ NPZ file not found: {npz_file}")
                else:
                    with st.spinner("Rendering mesh..."):
                        img, error = st.session_state.renderer.render_frame(
                            str(npz_file),
                            person_id=0,
                            img_size=img_size,
                            use_fallback=use_fallback
                        )
                        
                        if error is not None:
                            if "Using previous frame" in error:
                                st.warning(f"⚠️ {error}")
                            else:
                                st.error(f"❌ {error}")
                                # Show placeholder
                                img = create_placeholder_image(img_size, img_size, "No Detection")
                        
                        if img is not None:
                            st.image(img, use_column_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Statistics section
    st.markdown("---")
    st.subheader("📊 Dataset Statistics")
    
    if st.button("🔍 Scan NPZ Directory"):
        if not Path(npz_dir).exists():
            st.error(f"Directory not found: {npz_dir}")
        else:
            with st.spinner("Scanning..."):
                valid, invalid = scan_npz_directory(npz_dir)
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Frames", len(valid) + len(invalid))
                
                with col2:
                    st.metric("Valid Frames", len(valid))
                
                with col3:
                    st.metric("Invalid Frames", len(invalid))
                
                if len(invalid) > 0:
                    with st.expander("⚠️ View Invalid Frames"):
                        for npz_file, reason in invalid[:50]:  # Show first 50
                            st.text(f"{Path(npz_file).name}: {reason}")


if __name__ == "__main__":
    main()
