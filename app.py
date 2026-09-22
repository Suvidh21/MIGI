"""
DR. MIGI — Streamlit Cloud Entrypoint
======================================
Streamlit Community Cloud requires app.py in the project root.
This file adds the project root to sys.path and then executes
frontend/app.py using exec(), which Streamlit handles properly.
"""

import os
import sys

# Ensure project root is on Python path for all backend imports
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Execute the real Streamlit app
_app_path = os.path.join(PROJECT_ROOT, "frontend", "app.py")
with open(_app_path, "r", encoding="utf-8") as _f:
    exec(_f.read(), {"__name__": "__main__", "__file__": _app_path})
