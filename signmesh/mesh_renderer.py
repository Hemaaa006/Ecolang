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
