"""
Launcher script for the Multi-Modal Deepfake Detection Web Application & API.

Usage:
    python start_web.py
"""

import sys
import uvicorn

if __name__ == "__main__":
    print("=" * 70)
    print("  DeepGuard AI - Multi-Modal Deepfake Detection Platform")
    print("  Server starting at: http://127.0.0.1:8000")
    print("  API Docs available at: http://127.0.0.1:8000/docs")
    print("=" * 70)
    uvicorn.run("src.api.app:app", host="0.0.0.0", port=8000, reload=False)
