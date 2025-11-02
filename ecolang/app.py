"""
ECOLANG - Main Streamlit Application
Side-by-side video playback: Original vs Rendered
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

# Custom CSS - Minimalistic Design
st.markdown("""
<style>
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Main container */
    .block-container {
        padding-top: 3rem;
        padding-bottom: 3rem;
        max-width: 100%;
    }

    /* Page title - centered */
    .page-title {
        text-align: center;
        font-size: 3.5rem;
        font-weight: 700;
        color: #1a1a1a;
        margin-bottom: 3rem;
        letter-spacing: 0.5px;
    }

    /* Column containers */
    [data-testid="column"] {
        padding: 0 1.5rem;
    }

    /* Section headers - centered */
    .section-header {
        text-align: center;
        font-size: 1.5rem;
        font-weight: 600;
        color: #2c3e50;
        margin-bottom: 1.5rem;
        padding-bottom: 0.75rem;
        border-bottom: 2px solid #e0e0e0;
    }

    /* Video containers */
    .video-container {
        background: #ffffff;
        border-radius: 16px;
        padding: 2rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        min-height: 700px;
    }

    /* Video iframes */
    iframe {
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }

    /* Dropdown styling */
    .stSelectbox {
        margin-bottom: 1.5rem;
    }

    /* Button styling */
    .stButton > button {
        width: 100%;
        border-radius: 8px;
        height: 3rem;
        font-size: 1.1rem;
        font-weight: 600;
        margin-top: 1rem;
        background-color: #007BFF;
        color: white;
        border: none;
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        background-color: #0056b3;
        box-shadow: 0 4px 12px rgba(0,123,255,0.3);
    }

    /* Progress bar */
    .stProgress > div > div > div {
        background-color: #007BFF;
    }

    /* Empty placeholder styling */
    .placeholder-box {
        background: #f8f9fa;
        border: 2px dashed #dee2e6;
        border-radius: 12px;
        height: 600px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: #6c757d;
        font-size: 1.1rem;
        font-weight: 500;
    }

    /* Remove extra padding */
    .element-container {
        margin-bottom: 0;
    }

    /* Selectbox label */
    label {
        font-size: 0.95rem;
        font-weight: 500;
        color: #495057;
    }
</style>
""", unsafe_allow_html=True)

# Main app
def main():
    # Page Title
    st.markdown('<h1 class="page-title">ECOLANG</h1>', unsafe_allow_html=True)

    # Initialize API client
    api_client = get_api_client()

    # Check API health
    if 'api_healthy' not in st.session_state:
        with st.spinner("Connecting to backend..."):
            healthy, info = api_client.health_check()
            st.session_state.api_healthy = healthy
            st.session_state.api_info = info

    if not st.session_state.api_healthy:
        st.error(f"⚠️ Backend connection failed: {st.session_state.api_info}")
        st.info("Please ensure your Colab notebook is running with the correct ngrok URL.")
        return

    # Initialize session state
    if 'rendered_video_url' not in st.session_state:
        st.session_state.rendered_video_url = None
    if 'current_video_id' not in st.session_state:
        st.session_state.current_video_id = None
    if 'rendering_in_progress' not in st.session_state:
        st.session_state.rendering_in_progress = False
    if 'progress_data' not in st.session_state:
        st.session_state.progress_data = None

    # Two-column layout
    col1, col2 = st.columns([1, 1], gap="large")

    # LEFT COLUMN - Original Video
    with col1:
        st.markdown('<div class="video-container">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">Original Video</div>', unsafe_allow_html=True)

        # Video selector
        video_options = {k: v['title'] for k, v in config.VIDEO_LIBRARY.items()}
        selected_video = st.selectbox(
            "Select Video",
            options=list(video_options.keys()),
            format_func=lambda x: video_options[x],
            key="video_selector",
            label_visibility="visible"
        )

        # Reset rendered video if selection changes
        if st.session_state.current_video_id != selected_video:
            st.session_state.rendered_video_url = None
            st.session_state.current_video_id = selected_video
            st.session_state.rendering_in_progress = False
            st.session_state.progress_data = None

        # Display original video
        video_info = config.VIDEO_LIBRARY[selected_video]
        video_html = f"""
        <iframe src="{video_info['video_url']}"
                width="100%"
                height="600"
                frameborder="0"
                allow="autoplay"
                allowfullscreen>
        </iframe>
        """
        st.markdown(video_html, unsafe_allow_html=True)

        # Render button at bottom
        render_clicked = st.button("🎬 Render Video", type="primary", key="render_btn")

        st.markdown('</div>', unsafe_allow_html=True)

    # RIGHT COLUMN - Rendered Video + Progress
    with col2:
        st.markdown('<div class="video-container">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">Mesh Rendered Video</div>', unsafe_allow_html=True)

        # Show rendered video if available
        if st.session_state.rendered_video_url:
            rendered_html = f"""
            <iframe src="{st.session_state.rendered_video_url}"
                    width="100%"
                    height="600"
                    frameborder="0"
                    allow="autoplay"
                    allowfullscreen>
            </iframe>
            """
            st.markdown(rendered_html, unsafe_allow_html=True)

        # Show progress if rendering
        elif st.session_state.rendering_in_progress:
            st.markdown('<div class="placeholder-box">Rendering in progress...</div>', unsafe_allow_html=True)

            # Progress display area
            st.markdown("---")
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Poll for progress
            video_info = config.VIDEO_LIBRARY[selected_video]
            max_frames = video_info['frames']

            while st.session_state.rendering_in_progress:
                progress_data = api_client.get_render_progress(selected_video)

                if progress_data:
                    current = progress_data.get('current', 0)
                    total = progress_data.get('total', max_frames)
                    job_status = progress_data.get('status', 'unknown')

                    if total > 0:
                        progress_pct = int((current / total) * 100)
                        progress_bar.progress(progress_pct)
                        status_text.text(f"Frame {current} of {total} ({progress_pct}%)")

                    if job_status == "complete":
                        progress_bar.progress(100)
                        status_text.text("Complete!")

                        rendered_url = progress_data.get('video_url')
                        if rendered_url:
                            st.session_state.rendered_video_url = rendered_url
                            st.session_state.rendering_in_progress = False
                            time.sleep(1)
                            st.rerun()
                        break

                    elif job_status == "error":
                        error_msg = progress_data.get("error", "Unknown error")
                        st.error(f"Error: {error_msg}")
                        st.session_state.rendering_in_progress = False
                        break

                time.sleep(2)

        # Empty state
        else:
            st.markdown('<div class="placeholder-box">Select a video and click "Render Video"</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # Handle render button click
    if render_clicked and not st.session_state.rendering_in_progress:
        st.session_state.rendering_in_progress = True
        st.session_state.rendered_video_url = None

        # Start rendering
        video_url, status = api_client.render_video(selected_video)

        if "success" not in status:
            error = status.split(":", 1)[1] if ":" in status else status
            st.error(f"Failed to start: {error}")
            st.session_state.rendering_in_progress = False
        else:
            st.rerun()

if __name__ == "__main__":
    main()
