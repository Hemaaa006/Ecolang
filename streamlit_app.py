"""
ECOLANG - Main Entry Point for Streamlit Cloud
3D Mesh Rendering from Video Parameters
"""
import sys
from pathlib import Path

# Add ecolang directory to Python path
ecolang_path = Path(__file__).parent / "ecolang"
if str(ecolang_path) not in sys.path:
    sys.path.insert(0, str(ecolang_path))

# Import and run the main app from ecolang
try:
    from app import main
    # Call main() directly - Streamlit handles the execution
    main()
except ImportError as e:
    import streamlit as st
    st.error(f"Failed to import ECOLANG app: {e}")
    st.info("Make sure the ecolang directory contains app.py and all required modules")
    st.stop()
