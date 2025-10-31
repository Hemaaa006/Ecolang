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
