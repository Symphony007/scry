"""
start.py — Application entry point

Purpose:
    Builds the React frontend and then starts the FastAPI backend server.
    This is the ONLY file you run to launch the entire application.

Inputs:
    Optional CLI flag: --no-build (skips the npm build step)

Outputs:
    A running uvicorn server at http://127.0.0.1:8000

How it fits in:
    run `python start.py` -> builds frontend -> starts backend
    The backend (web/app.py) then serves both the API and the React UI.
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
    print("\n\u2500\u2500 Building React frontend \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500")

    if run("node --version", check=False) != 0:
        print("ERROR: Node.js not found. Install Node.js 18+ to build the frontend.")
        sys.exit(1)

    run("npm install", cwd=FRONTEND_DIR)
    run("npm run build", cwd=FRONTEND_DIR)
    print("\u2500\u2500 Frontend build complete \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500")

def start_server():
    port  = os.getenv("PORT", "8000")
    host  = os.getenv("HOST", "127.0.0.1")
    debug = os.getenv("DEBUG", "false").lower() == "true"

    print(f"\n\u2500\u2500 Starting Scry on http://{host}:{port} \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500")

    # core/, detectors/, ml/ live at ROOT (scry/) not inside web/.
    # Both ROOT and WEB are added to PYTHONPATH so all imports resolve
    # regardless of where uvicorn is launched from.
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    paths = [str(ROOT_DIR), str(WEB_DIR)]
    if existing:
        paths.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(paths)

    os.chdir(WEB_DIR)
    reload_flag = ["--reload"] if debug else []
    os.execvpe("uvicorn", [
        "uvicorn", "app:app",
        "--host", host,
        "--port", port,
        "--workers", "1",
        *reload_flag,
    ], env)

if __name__ == "__main__":
    skip_build = "--no-build" in sys.argv

    if not skip_build:
        build_frontend()

    start_server()