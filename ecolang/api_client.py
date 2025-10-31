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

    def render_frame(self, video_id, frame_number):
        """
        Request frame rendering from Colab API

        Args:
            video_id: Video identifier (e.g., 'ch07_speakerview_012')
            frame_number: Frame number to render (1-indexed)

        Returns:
            tuple: (image_array, status_message)
        """
        try:
            # Make API request
            response = self.session.post(
                f"{self.api_url}/render",
                json={
                    "video_id": video_id,
                    "frame_number": frame_number
                },
                timeout=30  # Rendering can take time
            )

            if response.status_code == 200:
                result = response.json()

                if result.get('success'):
                    # Decode base64 image
                    img_data = base64.b64decode(result['image'])
                    img = Image.open(io.BytesIO(img_data))
                    return np.array(img), "success"
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
