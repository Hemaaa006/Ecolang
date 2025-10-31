#!/usr/bin/env python3
"""Update app.py with fixed UI layout"""

import os
import sys

# Navigate to the ecolang directory
ecolang_path = r"C:\Users\dell\Desktop\Steamlit_render app\ecolang"

if not os.path.exists(ecolang_path):
    # Try alternate spelling
    ecolang_path = r"C:\Users\dell\Desktop\Steamlit_render app\ecolang"

if not os.path.exists(ecolang_path):
    print(f"ERROR: Cannot find ecolang directory")
    sys.exit(1)

app_py_path = os.path.join(ecolang_path, "app.py")

# New app.py content
content = '''"""
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

    # Check API health
    if 'api_healthy' not in st.session_state:
        with st.spinner("Connecting to Colab backend..."):
            healthy, info = api_client.health_check()
            st.session_state.api_healthy = healthy
            st.session_state.api_info = info

    if not st.session_state.api_healthy:
        st.error(f"Cannot connect to Colab backend: {st.session_state.api_info}")
        st.info("Please ensure your Colab notebook is running and ngrok URL is in Streamlit secrets")
        return

    # Initialize session state for rendered video
    if 'rendered_video_url' not in st.session_state:
        st.session_state.rendered_video_url = None
    if 'current_video_id' not in st.session_state:
        st.session_state.current_video_id = None
    if 'rendering_in_progress' not in st.session_state:
        st.session_state.rendering_in_progress = False

    # Main content - Side by side layout
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

        # Reset rendered video if selection changes
        if st.session_state.current_video_id != selected_video:
            st.session_state.rendered_video_url = None
            st.session_state.current_video_id = selected_video

        # Display original video
        video_info = config.VIDEO_LIBRARY[selected_video]
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

        # Render button - moved under original video
        if st.button("Render Video", type="primary", use_container_width=True):
            st.session_state.rendering_in_progress = True
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Start rendering
            status_text.text("Starting rendering...")
            video_url, status = api_client.render_video(selected_video)

            if "success" in status:
                # Poll for progress
                max_frames = video_info['frames']

                while True:
                    progress_data = api_client.get_render_progress(selected_video)

                    if progress_data:
                        current = progress_data.get('current', 0)
                        total = progress_data.get('total', max_frames)
                        job_status = progress_data.get('status', 'unknown')

                        if total > 0:
                            progress_pct = int((current / total) * 100)
                            progress_bar.progress(progress_pct)
                            status_text.text(f"Rendering: {current}/{total} frames ({progress_pct}%)")

                        if job_status == "complete":
                            progress_bar.progress(100)
                            status_text.text("Rendering complete!")

                            # Get video URL
                            rendered_url = progress_data.get('video_url')

                            if rendered_url:
                                # Save to session state
                                st.session_state.rendered_video_url = rendered_url
                                st.session_state.rendering_in_progress = False
                                st.success("Video rendered successfully!")
                                st.rerun()  # Refresh to show video
                            else:
                                st.warning("Video rendered but Drive URL not available")
                                local_path = progress_data.get('local_path', 'Unknown')
                                st.info(f"Path: {local_path}")
                            break

                        elif job_status == "error":
                            error_msg = progress_data.get("error", "Unknown error")
                            st.error(f"Rendering error: {error_msg}")
                            st.session_state.rendering_in_progress = False
                            break

                    time.sleep(2)  # Poll every 2 seconds
            else:
                error = status.split(":", 1)[1] if ":" in status else status
                st.error(f"Failed to start rendering: {error}")
                st.session_state.rendering_in_progress = False

        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="video-panel">', unsafe_allow_html=True)
        st.subheader("Mesh Rendered Video")

        # Show rendered video if available
        if st.session_state.rendered_video_url:
            rendered_html = f"""
            <iframe src="{st.session_state.rendered_video_url}"
                    width="100%"
                    height="400"
                    frameborder="0"
                    allow="autoplay"
                    allowfullscreen>
            </iframe>
            """
            st.markdown(rendered_html, unsafe_allow_html=True)
        else:
            # Placeholder when no rendered video
            st.info("Select a video and click 'Render Video' to see the 3D mesh rendering")

        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
'''

# Write the updated file
try:
    with open(app_py_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"✓ Successfully updated {app_py_path}")
    print("\nChanges made:")
    print("  1. Fixed video display - shows iframe for rendered video")
    print("  2. Moved 'Render Video' button under original video")
    print("  3. Removed extra text (duration, FPS, rendering time captions)")
    print("  4. Videos now aligned side-by-side")
except Exception as e:
    print(f"ERROR: {e}")
    sys.exit(1)
