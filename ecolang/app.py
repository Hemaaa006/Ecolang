"""
ECOLANG - Main Streamlit Application
3D Mesh Rendering from Video Parameters using Colab API
"""
import streamlit as st
import numpy as np
from pathlib import Path
import os

# Local imports
import config
from api_client import get_api_client

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
    .video-panel {
        background: white;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        margin-bottom: 20px;
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
    .info-msg {
        background: #D1ECF1;
        color: #0C5460;
        padding: 10px;
        border-radius: 4px;
        border: 1px solid #BEE5EB;
    }
</style>
""", unsafe_allow_html=True)

# Main app
def main():
    # Header
    st.markdown('<h1 class="main-header">ECOLANG</h1>', unsafe_allow_html=True)

    # Initialize API client
    api_client = get_api_client()

    # Check API health on first load
    if 'api_healthy' not in st.session_state:
        with st.spinner("Connecting to Colab backend..."):
            healthy, info = api_client.health_check()
            st.session_state.api_healthy = healthy
            st.session_state.api_info = info

    # Show API status
    if not st.session_state.api_healthy:
        st.error(f"Cannot connect to Colab backend: {st.session_state.api_info}")
        st.info("Please ensure your Colab notebook is running and the ngrok URL is correct in Streamlit secrets")
        return
    else:
        # Show subtle API status indicator
        with st.expander("Backend Status"):
            st.markdown(f'<div class="info-msg">Connected to Colab API</div>', unsafe_allow_html=True)
            if isinstance(st.session_state.api_info, dict):
                st.json(st.session_state.api_info)

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

        # Display video info
        video_info = config.VIDEO_LIBRARY[selected_video]
        st.video(video_info['github_url'])

        st.caption(f"Duration: {video_info['duration']} | FPS: {video_info['fps']} | Total Frames: {video_info['frames']}")

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

        # Render button
        if st.button("Render Frame", type="primary", use_container_width=True):
            with st.spinner("Rendering mesh via Colab API..."):
                img, status = api_client.render_frame(selected_video, frame_num)

                # Display result
                if img is not None:
                    st.image(img, use_column_width=True)

                    # Status message
                    if status == "success":
                        st.markdown('<div class="success-msg">Rendered successfully</div>',
                                  unsafe_allow_html=True)
                    elif status.startswith("fallback"):
                        reason = status.split(":", 1)[1] if ":" in status else ""
                        st.markdown(f'<div class="warning-msg">Using fallback: {reason}</div>',
                                  unsafe_allow_html=True)
                else:
                    error = status.split(":", 1)[1] if ":" in status else status
                    st.markdown(f'<div class="error-msg">Render failed: {error}</div>',
                                      unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
