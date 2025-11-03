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
    /* Hide Streamlit branding and default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Remove default padding that causes duplication */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 1rem;
        max-width: 100% !important;
    }

    /* Page title - centered */
    .page-title {
        text-align: center;
        font-size: 3rem;
        font-weight: 700;
        color: #1a1a1a;
        margin-bottom: 2.5rem;
        letter-spacing: 0.5px;
    }

    /* Section headers */
    h3 {
        text-align: center;
        font-size: 1.4rem;
        font-weight: 600;
        color: #2c3e50;
        margin-bottom: 1.25rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e0e0e0;
    }

    /* Column spacing */
    [data-testid="column"] {
        padding: 0 1rem;
    }

    /* Video wrapper for aspect ratio */
    .video-wrapper {
        position: relative;
        width: 100%;
        padding-bottom: 80%; /* Taller player for better visibility */
        margin-bottom: 1.5rem;
        background: #000;
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }

    .video-wrapper iframe {
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        border: none;
    }

    /* Placeholder box */
    .placeholder-wrapper {
        position: relative;
        width: 100%;
        padding-bottom: 75%;
        margin-bottom: 1.5rem;
        background: #f8f9fa;
        border: 2px dashed #dee2e6;
        border-radius: 12px;
        overflow: hidden;
    }

    .placeholder-content {
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        display: flex;
        align-items: center;
        justify-content: center;
        color: #6c757d;
        font-size: 1.1rem;
        font-weight: 500;
        text-align: center;
        padding: 2rem;
    }

    /* Selectbox styling */
    .stSelectbox {
        margin-bottom: 1.5rem;
    }

    .stSelectbox label {
        font-size: 0.95rem;
        font-weight: 500;
        color: #495057;
    }

    /* Button styling */
    .stButton > button {
        width: 100%;
        border-radius: 8px;
        height: 3rem;
        font-size: 1.05rem;
        font-weight: 600;
        background-color: #007BFF;
        color: white;
        border: none;
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        background-color: #0056b3;
        box-shadow: 0 4px 12px rgba(0,123,255,0.3);
        transform: translateY(-1px);
    }

    .stButton > button:active {
        transform: translateY(0);
    }

    /* Progress bar */
    .stProgress {
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }

    .stProgress > div > div > div {
        background-color: #007BFF;
    }

    /* Status text */
    .element-container p {
        text-align: center;
        color: #495057;
        font-weight: 500;
    }

    /* Error messages */
    .stError {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #f5c6cb;
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
        st.error(f"Backend connection failed: {st.session_state.api_info}")
        st.info("Please ensure your Colab notebook is running with the correct ngrok URL.")
        return

    # Initialize session state
    if 'rendered_video_url' not in st.session_state:
        st.session_state.rendered_video_url = None
    if 'current_video_id' not in st.session_state:
        st.session_state.current_video_id = None
    if 'rendering_in_progress' not in st.session_state:
        st.session_state.rendering_in_progress = False

    # Two-column layout
    col1, col2 = st.columns([1, 1], gap="large")

    # LEFT COLUMN - Original Video
    with col1:
        st.subheader("Original Video")

        # Video selector
        video_options = {k: v['title'] for k, v in config.VIDEO_LIBRARY.items()}
        selected_video = st.selectbox(
            "Select Video",
            options=list(video_options.keys()),
            format_func=lambda x: video_options[x],
            key="video_selector"
        )

        # Reset rendered video if selection changes
        if st.session_state.current_video_id != selected_video:
            st.session_state.rendered_video_url = None
            st.session_state.current_video_id = selected_video
            st.session_state.rendering_in_progress = False

        # Display original video with proper aspect ratio
        video_info = config.VIDEO_LIBRARY[selected_video]
        # Build original URL with optional sync/autoplay token
        orig_url = video_info['video_url']
        sync_token = st.session_state.get('sync_play_token')
        if 'drive.google.com' in orig_url:
            if '/file/d/' in orig_url:
                fid = orig_url.split('/file/d/')[1].split('/')[0]
                orig_url = f"https://drive.google.com/file/d/{fid}/preview"
            if sync_token:
                joiner = '&' if '?' in orig_url else '?'
                orig_url = f"{orig_url}{joiner}autoplay=1&mute=1&sync={sync_token}"

        video_html = f"""
        <div class=\"video-wrapper\">
            <iframe src=\"{orig_url}\" allow=\"autoplay\" allowfullscreen></iframe>
        </div>
        """
        st.markdown(video_html, unsafe_allow_html=True)

        # Render button (no emoji) and full width
        render_clicked = st.button("Render Video", type="primary", key="render_btn", use_container_width=True)

    # RIGHT COLUMN - Rendered Video + Progress
    with col2:
        st.subheader("Mesh Rendered Video")
        # Single placeholder to avoid accumulating multiple players
        video_area = st.empty()

        # Show rendered video if available (with autoplay)
        if st.session_state.rendered_video_url:
            # Modify Google Drive URL to enable autoplay
            rendered_url = st.session_state.rendered_video_url
            if "drive.google.com" in rendered_url:
                # Convert to embed format with autoplay
                if "/file/d/" in rendered_url:
                    file_id = rendered_url.split("/file/d/")[1].split("/")[0]
                    rendered_url = f"https://drive.google.com/file/d/{file_id}/preview"
                elif "preview" in rendered_url and "autoplay" not in rendered_url:
                    rendered_url = rendered_url

            # If a sync play was requested, force reload + autoplay
            sync_token = st.session_state.get('sync_play_token')
            if sync_token:
                joiner = '&' if '?' in rendered_url else '?'
                rendered_url = f"{rendered_url}{joiner}autoplay=1&mute=1&sync={sync_token}"

            rendered_html = f"""
            <div class=\"video-wrapper\">
                <iframe src=\"{rendered_url}\" allow=\"autoplay\" allowfullscreen></iframe>
            </div>
            """
            video_area.markdown(rendered_html, unsafe_allow_html=True)

        # Show progress if rendering
        elif st.session_state.rendering_in_progress:
            # Placeholder during rendering
            video_area.markdown("""
            <div class=\"placeholder-wrapper\">
                <div class=\"placeholder-content\">
                    Rendering in progress...
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Progress display
            st.markdown("---")
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Poll for progress
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
                        status_text.markdown(f"**Frame {current} of {total}** ({progress_pct}%)")

                    if job_status == "complete":
                        progress_bar.progress(100)
                        status_text.markdown("**Complete!**")

                        # Prefer direct Drive preview URL; fallback to file_id; else serve from backend
                        rendered_url = progress_data.get('video_url')
                        if not rendered_url:
                            file_id = progress_data.get('file_id')
                            if file_id:
                                rendered_url = f"https://drive.google.com/file/d/{file_id}/preview?autoplay=1"

                        # If still no URL, construct backend-served embed endpoint
                        if not rendered_url:
                            try:
                                base_url = api_client.api_url.rstrip('/')
                                rendered_url = f"{base_url}/rendered/{selected_video}"
                            except Exception:
                                rendered_url = None

                        if rendered_url:
                            st.session_state.rendered_video_url = rendered_url
                            st.session_state.rendering_in_progress = False
                            time.sleep(0.5)
                            st.rerun()
                        else:
                            # Couldn't get a public URL; show where it was saved
                            local_path = progress_data.get('drive_path') or progress_data.get('local_path')
                            if local_path:
                                status_text.markdown(f"Saved to: {local_path}")
                            st.session_state.rendering_in_progress = False
                        break

                    elif job_status == "error":
                        error_msg = progress_data.get("error", "Unknown error")
                        st.error(f"Error: {error_msg}")
                        st.session_state.rendering_in_progress = False
                        break

                time.sleep(2)

        # Empty state
        else:
            video_area.markdown("""
            <div class=\"placeholder-wrapper\">
                <div class=\"placeholder-content\">
                    Select a video and click \"Render Video\"
                </div>
            </div>
            """, unsafe_allow_html=True)

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

    # Global in-sync control (appears only when we have a rendered video)
    if st.session_state.get('rendered_video_url'):
        c1, c2, c3 = st.columns([1,2,1])
        with c2:
            if st.button("Play In Sync", key="sync_play_btn", use_container_width=True):
                st.session_state.sync_play_token = str(int(time.time()*1000))
                st.experimental_rerun()

# Note: main() is invoked by the Streamlit entrypoint script (streamlit_app.py).
