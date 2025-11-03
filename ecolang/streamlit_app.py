"""
Minimal Streamlit entrypoint that delegates to the main app.
Prevents duplicate UIs by centralizing rendering in app.main().
"""
import sys
from pathlib import Path

# Ensure local imports resolve
pkg_dir = Path(__file__).parent
if str(pkg_dir) not in sys.path:
    sys.path.insert(0, str(pkg_dir))

from app import main

if __name__ == "__main__":
    main()

