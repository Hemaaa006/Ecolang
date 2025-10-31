"""
ECOLANG - Main Streamlit Application
3D Mesh Rendering from Video Parameters
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
    page_title="ECOLANG - 3D Mesh Rendering",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed"
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
    st.markdown('<h1 class="main-header">ECOLANG</h1>', unsafe_allow_html=True)

    # Initialize managers
    file_mgr = get_file_manager()

    # Auto-initialize renderer
    if 'renderer' not in st.session_state:
        with st.spinner("Loading SMPL-X model..."):
            try:
                st.session_state.renderer = get_renderer(config.MODEL_PATH, config.DEFAULT_DEVICE)
            except Exception as e:
                st.error(f"Failed to initialize renderer: {e}")
                return

    # Main content
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="video-panel">', unsafe_allow_html=True)
        st.subheader("Original Video")

        # Video selector
        video_options = {k: v['title'] for k, v in config.VIDEO_LIBRARY.items()}
        selected_video = st.selectbox(
            "Select Video",
            options=list(video_options.keys()),
            format_func=lambda x: video_options[x],
            key="video_selector"
        )

        # Display video
        video_path = file_mgr.get_video_path(selected_video)

        if video_path and os.path.exists(video_path):
            st.video(video_path)
        else:
            st.error(f"Video not found: {video_path}")

        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="video-panel">', unsafe_allow_html=True)
        st.subheader("Mesh Render")

        # Frame selector
        max_frames = config.VIDEO_LIBRARY[selected_video]['frames']
        frame_num = st.number_input(
            "Frame Number",
            min_value=1,
            max_value=max_frames,
            value=1,
            step=1,
            key="frame_selector"
        )

        # Auto-render on video selection change
        npz_path = file_mgr.get_npz_path(selected_video, frame_num)

        if not os.path.exists(npz_path):
            st.error(f"NPZ file not found: frame_{frame_num:04d}")
        else:
            with st.spinner("Rendering mesh..."):
                img, status = st.session_state.renderer.render_frame(
                    npz_path,
                    person_id=0,
                    use_fallback=True
                )

                # Display result
                if img is not None:
                    st.image(img, use_column_width=True)

                    # Status message
                    if status == "success":
                        st.markdown('<div class="success-msg">Rendered successfully</div>',
                                  unsafe_allow_html=True)
                    elif status.startswith("fallback"):
                        reason = status.split(":", 1)[1]
                        st.markdown(f'<div class="warning-msg">Using fallback: {reason}</div>',
                                  unsafe_allow_html=True)
                else:
                    error = status.split(":", 1)[1] if ":" in status else status
                    st.markdown(f'<div class="error-msg">Render failed: {error}</div>',
                                      unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
