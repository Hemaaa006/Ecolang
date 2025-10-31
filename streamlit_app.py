"""
ECOLANG - Video to 3D Mesh Rendering
Clean interface with on-demand mesh generation via Google Colab API
"""
import streamlit as st
import httpx
import base64
from io import BytesIO
from PIL import Image
import cv2
import numpy as np
import os
import sys
from pathlib import Path
import time

# Add signmesh directory to path
signmesh_path = Path(__file__).parent / "signmesh"
if str(signmesh_path) not in sys.path:
    sys.path.insert(0, str(signmesh_path))

try:
    import config
except ImportError:
    st.error("Configuration module not found. Please ensure signmesh/config.py exists.")
    st.stop()

# Page configuration - MUST be first Streamlit command
st.set_page_config(
    page_title="ECOLANG",
    page_icon="🎬",
    layout="wide"
)

# Minimal, clean CSS
st.markdown("""
<style>
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Main header */
    .main-header {
        text-align: center;
        color: #2E86AB;
        font-size: 56px;
        font-weight: 700;
        margin-bottom: 50px;
        letter-spacing: 4px;
        text-transform: uppercase;
    }

    /* Video panels */
    .video-panel {
        background: white;
        border-radius: 8px;
        padding: 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        margin-bottom: 20px;
    }

    /* Panel titles */
    .panel-title {
        font-size: 18px;
        font-weight: 600;
        margin-bottom: 15px;
        color: #333;
        text-align: center;
    }

    /* Progress text */
    .progress-text {
        text-align: center;
        color: #666;
        font-size: 14px;
        margin: 10px 0;
    }

    /* Remove extra padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 0rem;
    }
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown('<h1 class="main-header">ECOLANG</h1>', unsafe_allow_html=True)

# Get Colab API URL from secrets or environment
try:
    COLAB_API_URL = st.secrets["COLAB_API_URL"]
except:
    COLAB_API_URL = os.environ.get('COLAB_API_URL', config.COLAB_API_URL)

# Initialize session state
if 'mesh_video_path' not in st.session_state:
    st.session_state.mesh_video_path = None
if 'current_video' not in st.session_state:
    st.session_state.current_video = None
if 'generating' not in st.session_state:
    st.session_state.generating = False

# Video selection dropdown (centered, no label)
col_spacer1, col_dropdown, col_spacer2 = st.columns([1, 2, 1])
with col_dropdown:
    video_options = {k: v['title'] for k, v in config.VIDEO_LIBRARY.items()}
    selected_video = st.selectbox(
        "Select Video",
        options=list(video_options.keys()),
        format_func=lambda x: video_options[x],
        label_visibility="collapsed"
    )

# Auto-trigger mesh generation when video changes
if selected_video != st.session_state.current_video:
    st.session_state.current_video = selected_video
    st.session_state.mesh_video_path = None
    st.session_state.generating = False
    st.rerun()

# Display videos side by side
col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="video-panel">', unsafe_allow_html=True)
    st.markdown('<p class="panel-title">Original Video</p>', unsafe_allow_html=True)

    # Get video URL from config
    video_info = config.VIDEO_LIBRARY[selected_video]
    video_url = video_info['github_url']

    # Display original video
    st.video(video_url)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="video-panel">', unsafe_allow_html=True)
    st.markdown('<p class="panel-title">Mesh Render</p>', unsafe_allow_html=True)

    # Check if we need to generate mesh
    if st.session_state.mesh_video_path is None and not st.session_state.generating:
        st.session_state.generating = True

        # Create progress UI elements
        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            # Test API health
            status_text.markdown('<p class="progress-text">Connecting to Colab API...</p>', unsafe_allow_html=True)

            with httpx.Client(timeout=10.0) as client:
                response = client.get(f"{COLAB_API_URL}/health")
                if response.status_code != 200:
                    st.error("❌ Colab API is not responding")
                    st.info("Please ensure the Colab notebook is running and the API URL is correct in secrets.")
                    st.stop()

            status_text.markdown('<p class="progress-text">✓ Connected to Colab API</p>', unsafe_allow_html=True)
            time.sleep(0.5)

            # Generate all mesh frames
            mesh_frames = []
            total_frames = video_info['frames']

            # Use a persistent client for better performance
            with httpx.Client(timeout=60.0) as client:
                for frame_num in range(1, total_frames + 1):
                    status_text.markdown(
                        f'<p class="progress-text">Generating frame {frame_num} of {total_frames}</p>',
                        unsafe_allow_html=True
                    )

                    try:
                        # Call Colab API
                        response = client.post(
                            f"{COLAB_API_URL}/render_frame",
                            json={
                                "video_id": selected_video,
                                "frame_number": frame_num
                            }
                        )

                        if response.status_code == 200:
                            data = response.json()
                            # Decode base64 image
                            img_data = base64.b64decode(data['image'])
                            img = Image.open(BytesIO(img_data))
                            mesh_frames.append(np.array(img))
                        else:
                            # Use fallback: repeat last valid frame
                            if mesh_frames:
                                mesh_frames.append(mesh_frames[-1])
                            else:
                                # No valid frames yet, create black frame
                                mesh_frames.append(np.zeros((720, 720, 3), dtype=np.uint8))

                        # Update progress
                        progress_bar.progress(frame_num / total_frames)

                    except Exception as e:
                        st.warning(f"Frame {frame_num} failed: {str(e)}")
                        if mesh_frames:
                            mesh_frames.append(mesh_frames[-1])

            # Convert frames to video
            status_text.markdown('<p class="progress-text">Compiling video...</p>', unsafe_allow_html=True)

            output_dir = Path("mesh_videos")
            output_dir.mkdir(exist_ok=True)
            output_path = output_dir / f"mesh_{selected_video}.mp4"

            # Create video from frames
            fps = video_info.get('fps', 30)
            height, width = mesh_frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

            for frame in mesh_frames:
                out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            out.release()

            # Save to session state
            st.session_state.mesh_video_path = str(output_path)
            st.session_state.generating = False

            # Clear progress UI
            status_text.empty()
            progress_bar.empty()

            # Success message
            st.success("✓ Mesh video generated successfully!")
            time.sleep(1)
            st.rerun()

        except httpx.ConnectError:
            st.error("❌ Cannot connect to Colab API")
            st.info(f"""
            **Troubleshooting:**
            1. Make sure the Colab notebook is running
            2. Check that the ngrok URL is correct in Streamlit secrets
            3. Current API URL: `{COLAB_API_URL}`
            """)
            st.session_state.generating = False
            st.stop()

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.info("Please check the Colab API logs for more details")
            st.session_state.generating = False
            st.stop()

    # Display mesh video if available
    if st.session_state.mesh_video_path and os.path.exists(st.session_state.mesh_video_path):
        st.video(st.session_state.mesh_video_path)
    elif st.session_state.generating:
        st.info("Generating mesh video...")

    st.markdown('</div>', unsafe_allow_html=True)

# Footer (minimal)
st.markdown("---")
st.markdown(
    '<p style="text-align: center; color: #999; font-size: 12px;">ECOLANG v1.0 - Powered by SMPL-X & Google Colab</p>',
    unsafe_allow_html=True
)
