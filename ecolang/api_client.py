"""
API Client for Colab Backend
Handles communication with the Colab mesh rendering API
"""
import requests
import base64
import numpy as np
from PIL import Image
import io
import streamlit as st


class ColabAPIClient:
    """Client for communicating with Colab rendering backend"""

    def __init__(self, api_url):
        """
        Initialize API client

        Args:
            api_url: Base URL of the Colab API (e.g., https://xxx.ngrok-free.app)
        """
        self.api_url = api_url.rstrip('/')
        self.session = requests.Session()

    def health_check(self):
        """Check if API is alive and responsive"""
        try:
            response = self.session.get(f"{self.api_url}/health", timeout=5)
            if response.status_code == 200:
                return True, response.json()
            return False, f"API returned status {response.status_code}"
        except requests.exceptions.RequestException as e:
            return False, f"Connection error: {str(e)}"

    def render_video(self, video_id, progress_callback=None):
        """
        Request full video rendering from Colab API

        Args:
            video_id: Video identifier (e.g., 'ch07_speakerview_012')
            progress_callback: Optional callback function(current, total, message)

        Returns:
            tuple: (video_url, status_message)
        """
        try:
            # Make API request
            response = self.session.post(
                f"{self.api_url}/render_video",
                json={"video_id": video_id},
                timeout=300,  # 5 minutes for full video
                stream=True
            )

            if response.status_code == 200:
                result = response.json()

                if result.get('success'):
                    return result.get('video_url'), "success"
                else:
                    return None, f"error:{result.get('error', 'Unknown error')}"
            else:
                return None, f"error:API returned status {response.status_code}"

        except requests.exceptions.Timeout:
            return None, "error:Request timed out"
        except requests.exceptions.RequestException as e:
            return None, f"error:Connection error - {str(e)}"
        except Exception as e:
            return None, f"error:{str(e)}"

    def get_render_progress(self, video_id):
        """
        Get rendering progress for a video

        Args:
            video_id: Video identifier

        Returns:
            dict: Progress information {current_frame, total_frames, status}
        """
        try:
            response = self.session.get(
                f"{self.api_url}/render_progress/{video_id}",
                timeout=5
            )

            if response.status_code == 200:
                return response.json()
            return None

        except Exception:
            return None


@st.cache_resource
def get_api_client():
    """Get or create API client (cached)"""
    # Try to get API URL from Streamlit secrets
    api_url = st.secrets.get("COLAB_API_URL", None)

    if not api_url:
        st.error("COLAB_API_URL not configured in Streamlit secrets!")
        st.info("Please add your Colab ngrok URL to Streamlit Cloud secrets")
        st.stop()

    return ColabAPIClient(api_url)
