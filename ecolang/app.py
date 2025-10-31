"""
ECOLANG - Main Streamlit Application
3D Mesh Rendering from Video Parameters using Colab API
"""
import streamlit as st
import time

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
        margin-bottom: 30px;
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
        margin-top: 10px;
    }
    .error-msg {
        background: #F8D7DA;
        color: #721C24;
        padding: 10px;
        border-radius: 4px;
        border: 1px solid #F5C6CB;
        margin-top: 10px;
    }
    .info-msg {
        background: #D1ECF1;
        color: #0C5460;
        padding: 10px;
        border-radius: 4px;
        border: 1px solid #BEE5EB;
    }
    iframe {
        border-radius: 8px;
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

        # Display video using iframe for Google Drive
        video_info = config.VIDEO_LIBRARY[selected_video]

        # Embed Google Drive video
        video_html = f"""
        <iframe src="{video_info['video_url']}"
                width="100%"
                height="400"
                frameborder="0"
                allow="autoplay"
                allowfullscreen>
        </iframe>
        """
        st.markdown(video_html, unsafe_allow_html=True)

        st.caption(f"Duration: {video_info['duration']} | FPS: {video_info['fps']} | Total Frames: {video_info['frames']}")

        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="video-panel">', unsafe_allow_html=True)
        st.subheader("Mesh Rendered Video")

        # Render button
        if st.button("Render Video", type="primary", use_container_width=True):
            # Create placeholder for progress
            progress_bar = st.progress(0)
            status_text = st.empty()
            video_placeholder = st.empty()

            # Start rendering
            status_text.text("Starting video rendering...")

            # Initialize progress tracking
            start_time = time.time()
            max_frames = video_info['frames']

            # Call API to start rendering
            with st.spinner("Rendering video..."):
                video_url, status = api_client.render_video(selected_video)

                if video_url:
                    # Success - show rendered video
                    progress_bar.progress(100)
                    status_text.text("Rendering complete!")

                    # Display rendered video
                    video_placeholder.markdown(f"""
                    <iframe src="{video_url}"
                            width="100%"
                            height="400"
                            frameborder="0"
                            allow="autoplay"
                            allowfullscreen>
                    </iframe>
                    """, unsafe_allow_html=True)

                    st.markdown('<div class="success-msg">Video rendered successfully!</div>',
                              unsafe_allow_html=True)
                else:
                    # Error
                    progress_bar.empty()
                    status_text.empty()
                    error = status.split(":", 1)[1] if ":" in status else status
                    st.markdown(f'<div class="error-msg">Render failed: {error}</div>',
                                      unsafe_allow_html=True)
        else:
            # Show placeholder message
            st.info("Click 'Render Video' to generate 3D mesh rendering")
            st.caption("The rendering process will convert all video frames to 3D mesh animations")

        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
