"""
Scry — FastAPI backend

Endpoints:
    POST /api/detect        — upload image, run full detection pipeline
    POST /api/embed         — embed a message into an image
    POST /api/decode        — decode a hidden message from an image
    GET  /api/health        — health check

Static files (React build) are served from /static.
In production, the React app is built and served by FastAPI directly.
"""

import os
import sys
from pathlib import Path as _Path

# Add both project root (for core/, detectors/, ml/) and web/ (for config)
# to sys.path so all imports resolve regardless of how uvicorn is launched
_ROOT = _Path(__file__).parent.parent   # scry/
_WEB  = _Path(__file__).parent          # scry/web/
for _p in [str(_WEB), str(_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import uuid
import logging
from pathlib import Path
from PIL import Image

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.background import BackgroundTask

from config import settings

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level   = getattr(logging, settings.log_level.upper(), logging.INFO),
    format  = "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt = "%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("scry")

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title       = settings.app_title,
    description = "Detects and embeds steganographic content in images.",
    version     = settings.app_version,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = settings.cors_origins,
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

TEMP_DIR = settings.temp_dir

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def save_upload(upload: UploadFile) -> Path:
    """
    Read upload into a temp file.
    Raises HTTPException 413 if file exceeds MAX_UPLOAD_MB.
    """
    data   = upload.file.read()
    size   = len(data)

    if size > settings.max_upload_bytes:
        mb = size / (1024 * 1024)
        raise HTTPException(
            status_code = 413,
            detail      = (
                f"File too large ({mb:.1f} MB). "
                f"Maximum allowed size is {settings.max_upload_mb} MB."
            )
        )

    suffix = Path(upload.filename).suffix.lower() or ".png"
    path   = TEMP_DIR / f"{uuid.uuid4().hex}{suffix}"

    with open(path, "wb") as f:
        f.write(data)

    logger.debug(f"Saved upload: {path.name} ({size / 1024:.1f} KB)")
    return path


def cleanup(*paths) -> None:
    """Delete one or more temp files, silently ignoring missing files."""
    for path in paths:
        try:
            if path and Path(str(path)).exists():
                os.unlink(str(path))
                logger.debug(f"Cleaned up temp file: {Path(str(path)).name}")
        except Exception as exc:
            logger.warning(f"Failed to clean up {path}: {exc}")


def run_detection_pipeline(image_path: Path) -> dict:
    import numpy as np
    from PIL import Image as PILImage
    from core.format_handler import classify as classify_format
    from ml.classifier import get_classifier
    from detectors.chi_square  import ChiSquareDetector
    from detectors.entropy     import EntropyDetector
    from detectors.rs_analysis import RSAnalysisDetector
    from detectors.histogram   import HistogramDetector
    from detectors.aggregator  import build_type_aware_aggregator

    logger.info(f"Running detection pipeline on: {image_path.name}")

    fmt_info    = classify_format(str(image_path))
    img         = PILImage.open(str(image_path)).convert("RGB")
    arr         = np.array(img)
    clf         = get_classifier()
    type_result = clf.classify(arr)

    detectors = [
        ChiSquareDetector(),
        EntropyDetector(),
        RSAnalysisDetector(),
        HistogramDetector(),
    ]
    detector_results = [d.analyze(arr) for d in detectors]

    agg        = build_type_aware_aggregator(type_result)
    agg_result = agg.aggregate(detector_results)

    logger.info(
        f"Detection complete — verdict: {agg_result.final_verdict.value} "
        f"({agg_result.final_probability:.2%})"
    )

    return {
        "format": {
            "actual_format"      : fmt_info.actual_format.value,
            "compression"        : fmt_info.compression.value,
            "embedding_domain"   : fmt_info.embedding_domain.value,
            "width"              : fmt_info.width,
            "height"             : fmt_info.height,
            "bit_depth"          : fmt_info.bit_depth,
            "color_space"        : fmt_info.color_space,
            "has_alpha"          : fmt_info.has_alpha,
            "has_metadata"       : fmt_info.has_metadata,
            "extension_mismatch" : fmt_info.extension_mismatch,
            "is_supported"       : fmt_info.is_supported,
        },
        "image_type": {
            "type"               : type_result.image_type,
            "confidence"         : round(type_result.confidence, 4),
            "method"             : type_result.method,
            "class_probabilities": {
                k: round(v, 4)
                for k, v in type_result.class_probabilities.items()
            },
            "reliability_notes"  : type_result.reliability_notes,
        },
        "detectors": [
            {
                "name"        : r.detector,
                "probability" : round(r.probability, 4),
                "confidence"  : round(r.confidence, 4),
                "verdict"     : r.verdict.value,
                "reliability" : r.reliability.value,
                "notes"       : r.notes,
                "weight_used" : round(agg_result.weights_used.get(r.detector, 0.0), 2),
            }
            for r in detector_results
        ],
        "verdict": {
            "final_verdict"      : agg_result.final_verdict.value,
            "final_probability"  : round(agg_result.final_probability, 4),
            "confidence"         : round(agg_result.confidence, 4),
            "payload_estimate"   : round(agg_result.payload_estimate, 2)
                                   if agg_result.payload_estimate is not None
                                   else None,
            "notes"              : agg_result.notes,
        },
        "warnings": (
            ["Extension mismatch detected — file may have been renamed."]
            if fmt_info.extension_mismatch else []
        ),
    }


# ---------------------------------------------------------------------------
# API Routes
# ---------------------------------------------------------------------------

@app.get("/api/health")
def health():
    return {
        "status"  : "ok",
        "version" : settings.app_version,
        "debug"   : settings.debug,
    }


@app.post("/api/detect")
async def detect(file: UploadFile = File(...)):
    path = None
    try:
        logger.info(f"Detect request — file: {file.filename}")
        path   = save_upload(file)
        result = run_detection_pipeline(path)
        return JSONResponse(content=result)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Detection failed: {e}")
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")
    finally:
        if path:
            cleanup(path)


@app.post("/api/embed")
async def embed(
    file    : UploadFile = File(...),
    message : str        = Form(...),
    method  : str        = Form("lsb_matching"),
):
    src_path    = None
    png_path    = None
    dst_path    = None
    output_path = None

    VALID_METHODS = {"lsb_replacement", "lsb_matching", "metadata", "dwt"}

    try:
        logger.info(f"Embed request — file: {file.filename}, method: {method}")

        if method not in VALID_METHODS:
            raise HTTPException(
                status_code = 400,
                detail      = f"Unknown method '{method}'. Valid: {sorted(VALID_METHODS)}"
            )

        src_path = save_upload(file)

        from core.format_handler import classify, EmbeddingDomain
        info = classify(str(src_path))

        if not info.is_supported:
            raise HTTPException(
                status_code = 400,
                detail      = (
                    f"Format '{info.actual_format.value}' is not supported. "
                    f"Please upload a PNG, JPEG, WebP, or TIFF image."
                )
            )

        # ------------------------------------------------------------------
        # Metadata
        # ------------------------------------------------------------------
        if method == "metadata":
            from core.metadata_embedder import embed_metadata

            src_suffix = src_path.suffix.lower()
            out_suffix = ".png" if src_suffix in (".webp", ".tiff", ".tif") else src_suffix
            dst_path   = TEMP_DIR / f"{uuid.uuid4().hex}_stego{out_suffix}"

            result      = embed_metadata(str(src_path), message, str(dst_path))
            output_path = Path(result["output_path"])
            download_name = Path(file.filename).stem + output_path.suffix.lower()

        # ------------------------------------------------------------------
        # DWT
        # ------------------------------------------------------------------
        elif method == "dwt":
            from core.dwt_embedder import embed_dwt

            if src_path.suffix.lower() in ('.jpg', '.jpeg'):
                png_path = TEMP_DIR / f"{uuid.uuid4().hex}_converted.png"
                Image.open(str(src_path)).convert("RGB").save(str(png_path), format="PNG")
                embed_src = str(png_path)
            else:
                embed_src = str(src_path)

            dst_path      = TEMP_DIR / f"{uuid.uuid4().hex}_stego.png"
            result        = embed_dwt(embed_src, message, str(dst_path))
            output_path   = Path(result["output_path"])
            download_name = Path(file.filename).stem + "_stego.png"

        # ------------------------------------------------------------------
        # Spatial LSB
        # ------------------------------------------------------------------
        else:
            if info.embedding_domain == EmbeddingDomain.DCT:
                png_path = TEMP_DIR / f"{uuid.uuid4().hex}_converted.png"
                Image.open(str(src_path)).convert("RGB").save(str(png_path), format="PNG")
                embed_src = str(png_path)
            else:
                embed_src = str(src_path)

            dst_path = TEMP_DIR / f"{uuid.uuid4().hex}_stego.png"

            if method == "lsb_matching":
                from core.lsb_matching_embedder import embed_matching
                embed_matching(embed_src, message, str(dst_path))
            else:
                from core.embedder import embed as lsb_replace
                lsb_replace(embed_src, message, str(dst_path))

            output_path   = dst_path
            download_name = Path(file.filename).stem + "_stego.png"

        logger.info(f"Embed complete — output: {output_path.name}")

        files_to_cleanup = [p for p in [src_path, png_path] if p]
        if dst_path and Path(str(dst_path)) != output_path:
            files_to_cleanup.append(dst_path)

        return FileResponse(
            path       = str(output_path),
            media_type = "application/octet-stream",
            filename   = download_name,
            background = BackgroundTask(cleanup, *files_to_cleanup, output_path),
        )

    except HTTPException:
        cleanup(src_path, png_path, dst_path)
        raise
    except Exception as e:
        logger.exception(f"Embedding failed: {e}")
        cleanup(src_path, png_path, dst_path)
        if output_path and output_path != dst_path:
            cleanup(output_path)
        raise HTTPException(status_code=500, detail=f"Embedding failed: {str(e)}")


@app.post("/api/decode")
async def decode_endpoint(file: UploadFile = File(...)):
    path = None
    try:
        logger.info(f"Decode request — file: {file.filename}")
        path = save_upload(file)

        from core.decoder import decode
        result = decode(str(path))

        if result.success:
            logger.info(f"Decode successful — method: {result.method_used}")
        else:
            logger.info(f"Decode found no payload — {result.error}")

        return JSONResponse(content={
            "success"         : result.success,
            "message"         : result.message if result.success else "",
            "method_used"     : result.method_used,
            "format_detected" : result.format_detected,
            "error"           : result.error,
            "warnings"        : result.warnings,
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Decode failed: {e}")
        raise HTTPException(status_code=500, detail=f"Decode failed: {str(e)}")
    finally:
        if path:
            cleanup(path)


# ---------------------------------------------------------------------------
# Serve React frontend
# ---------------------------------------------------------------------------

STATIC_DIR = Path(__file__).parent.parent / "static"

if STATIC_DIR.exists():
    logger.info(f"Serving static frontend from: {STATIC_DIR}")
    app.mount("/assets", StaticFiles(directory=str(STATIC_DIR / "assets")), name="assets")

    @app.get("/")
    async def serve_root():
        return FileResponse(str(STATIC_DIR / "index.html"))

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        file_path = STATIC_DIR / full_path
        if file_path.exists() and file_path.is_file():
            return FileResponse(str(file_path))
        return FileResponse(str(STATIC_DIR / "index.html"))
else:
    logger.warning(
        f"Static directory not found at {STATIC_DIR}. "
        f"Run 'npm run build' in web/frontend/ to generate it."
    )