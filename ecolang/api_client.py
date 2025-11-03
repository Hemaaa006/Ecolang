"""
API Client for Colab Backend
Handles communication with the Colab mesh rendering API
"""
import requests
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

    def render_video(self, video_id, force=False):
        """
        Request full video rendering from Colab API.

        Args:
            video_id: Video identifier (e.g., 'ch07_speakerview_012')
            force: When True, cancel any running render before starting this one.

        Returns:
            tuple: (video_url, status_message, response_data)
        """
        try:
            response = self.session.post(
                f"{self.api_url}/render_video",
                json={"video_id": video_id, "force": force},
                timeout=300,
                stream=False
            )

            try:
                result = response.json()
            except ValueError:
                result = {}

            if response.status_code == 200:
                if result.get('success'):
                    return result.get('video_url'), "success", result
                else:
                    return None, f"error:{result.get('error', 'Unknown error')}", result

            error_key = result.get("error") if isinstance(result, dict) else None
            return None, f"error:{error_key or f'status_{response.status_code}'}", result

        except requests.exceptions.Timeout:
            return None, "error:Request timed out", {}
        except requests.exceptions.RequestException as e:
            return None, f"error:Connection error - {str(e)}", {}
        except Exception as e:
            return None, f"error:{str(e)}", {}

    def get_video_library(self, refresh=False):
        """Return discovered videos with status metadata."""
        try:
            params = {"refresh": "true"} if refresh else None
            response = self.session.get(
                f"{self.api_url}/video_library",
                params=params,
                timeout=10
            )
            if response.status_code == 200:
                return True, response.json()
            return False, {"error": f"status_{response.status_code}"}
        except requests.exceptions.RequestException as exc:
            return False, {"error": str(exc)}

    def get_render_manifest(self, refresh=False):
        """Fetch raw manifest data keyed by video_id."""
        try:
            params = {"refresh": "true"} if refresh else None
            response = self.session.get(
                f"{self.api_url}/render_manifest",
                params=params,
                timeout=10
            )
            if response.status_code == 200:
                return True, response.json()
            return False, {"error": f"status_{response.status_code}"}
        except requests.exceptions.RequestException as exc:
            return False, {"error": str(exc)}

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

    def get_original_stream_url(self, video_id):
        """Return absolute URL for original video streaming."""
        return f"{self.api_url}/original_file/{video_id}"

    def get_rendered_stream_url(self, video_id):
        """Return absolute URL for rendered video streaming."""
        return f"{self.api_url}/rendered_file/{video_id}"


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
