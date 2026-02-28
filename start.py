"""
Scry — Production start script

Builds the React frontend then starts the FastAPI server.

Usage:
    python start.py            — build frontend + start server
    python start.py --no-build — skip frontend build, just start server
"""

import subprocess
import sys
import os
from pathlib import Path

ROOT_DIR     = Path(__file__).parent
FRONTEND_DIR = ROOT_DIR / "web" / "frontend"
WEB_DIR      = ROOT_DIR / "web"

def run(cmd, cwd=None, check=True):
    print(f"\n>>> {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=cwd, check=check)
    return result.returncode

def build_frontend():
    print("\n── Building React frontend ──────────────────────────")

    if run("node --version", check=False) != 0:
        print("ERROR: Node.js not found. Install Node.js 18+ to build the frontend.")
        sys.exit(1)

    run("npm install", cwd=FRONTEND_DIR)
    run("npm run build", cwd=FRONTEND_DIR)
    print("── Frontend build complete ──────────────────────────")

def start_server():
    port  = os.getenv("PORT", "8000")
    host  = os.getenv("HOST", "127.0.0.1")
    debug = os.getenv("DEBUG", "false").lower() == "true"

    print(f"\n── Starting Scry on http://{host}:{port} ────────────")

    # core/, detectors/, ml/ live at ROOT (scry/) not inside web/
    # Add both ROOT and WEB to PYTHONPATH so all imports resolve
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    paths = [str(ROOT_DIR), str(WEB_DIR)]
    if existing:
        paths.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(paths)

    os.chdir(WEB_DIR)
    os.execvpe("uvicorn", [
        "uvicorn", "app:app",
        "--host", host,
        "--port", str(port),
        "--workers", "1",
        *(["--reload"] if debug else []),
    ], env)

if __name__ == "__main__":
    skip_build = "--no-build" in sys.argv

    if not skip_build:
        build_frontend()

    start_server()