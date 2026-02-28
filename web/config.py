"""
Scry — Configuration

All settings are read from environment variables.
In development, values are loaded from a .env file in the project root.
In production (Render), set these as environment variables in the dashboard.

Usage:
    from config import settings
    print(settings.max_upload_bytes)
"""

import os
import tempfile
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root (two levels up from web/)
_env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=_env_path)


class Settings:
    # Server
    host: str        = os.getenv("HOST", "0.0.0.0")
    port: int        = int(os.getenv("PORT", "8000"))
    debug: bool      = os.getenv("DEBUG", "false").lower() == "true"

    # Logging
    log_level: str   = os.getenv("LOG_LEVEL", "info").lower()

    # File uploads — default 10 MB
    max_upload_mb: int    = int(os.getenv("MAX_UPLOAD_MB", "10"))
    max_upload_bytes: int = max_upload_mb * 1024 * 1024

    # Temp directory for processing uploads
    temp_dir: Path   = Path(
        os.getenv("TEMP_DIR", str(Path(tempfile.gettempdir()) / "scry_uploads"))
    )

    # CORS — comma-separated allowed origins, default "*" for public tool
    cors_origins: list = [
        o.strip()
        for o in os.getenv("CORS_ORIGINS", "*").split(",")
    ]

    # App metadata
    app_version: str = "0.1.0"
    app_title: str   = "Scry — Steganography Engine"


settings = Settings()

# Ensure temp dir exists at startup
settings.temp_dir.mkdir(parents=True, exist_ok=True)