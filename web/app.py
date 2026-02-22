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
import uuid
import tempfile
import traceback
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title       = "Scry — Steganography Detection Engine",
    description = "Detects and embeds steganographic content in images.",
    version     = "0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# Temp directory for uploaded files
TEMP_DIR = Path(tempfile.gettempdir()) / "scry_uploads"
TEMP_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def save_upload(upload: UploadFile) -> Path:
    """Save an uploaded file to a temp path with its original extension."""
    suffix = Path(upload.filename).suffix.lower() or ".png"
    path   = TEMP_DIR / f"{uuid.uuid4().hex}{suffix}"
    with open(path, "wb") as f:
        f.write(upload.file.read())
    return path


def cleanup(path: Path) -> None:
    try:
        if path and path.exists():
            os.unlink(path)
    except Exception:
        pass


def run_detection_pipeline(image_path: Path) -> dict:
    """
    Run the full Scry detection pipeline on an image file.

    Pipeline:
        1. Format classification
        2. Image type classification
        3. All four statistical detectors
        4. Type-aware aggregation
        5. Return structured result

    Returns a dict safe for JSON serialization.
    """
    import numpy as np
    from PIL import Image as PILImage

    from core.format_handler import classify as classify_format
    from ml.type_classifier  import ImageTypeClassifier
    from detectors.chi_square  import ChiSquareDetector
    from detectors.entropy     import EntropyDetector
    from detectors.rs_analysis import RSAnalysisDetector
    from detectors.histogram   import HistogramDetector
    from detectors.aggregator  import build_type_aware_aggregator

    # Step 1: Format info
    fmt_info = classify_format(str(image_path))

    # Step 2: Load image as numpy array
    img = PILImage.open(str(image_path)).convert("RGB")
    arr = np.array(img)

    # Step 3: Image type classification
    clf      = ImageTypeClassifier()
    type_result = clf.classify(arr)

    # Step 4: Run all four detectors
    detectors = [
        ChiSquareDetector(),
        EntropyDetector(),
        RSAnalysisDetector(),
        HistogramDetector(),
    ]
    detector_results = [d.analyze(arr) for d in detectors]

    # Step 5: Type-aware aggregation
    agg        = build_type_aware_aggregator(type_result)
    agg_result = agg.aggregate(detector_results)

    # Step 6: Serialize everything to JSON-safe dict
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
    return {"status": "ok", "version": "0.1.0"}


@app.post("/api/detect")
async def detect(file: UploadFile = File(...)):
    """
    Upload an image and run the full steganography detection pipeline.
    Returns format info, image type classification, per-detector results,
    and a final weighted verdict.
    """
    path = None
    try:
        path   = save_upload(file)
        result = run_detection_pipeline(path)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(
            status_code = 500,
            detail      = f"Detection failed: {str(e)}"
        )
    finally:
        if path:
            cleanup(path)

@app.post("/api/embed")
async def embed(
    file    : UploadFile = File(...),
    message : str        = Form(...),
):
    src_path = None
    dst_path = None
    try:
        src_path = save_upload(file)

        from core.format_handler import classify, EmbeddingDomain
        info = classify(str(src_path))
        print(f"[DEBUG] actual_format={info.actual_format.value} domain={info.embedding_domain.value} supported={info.is_supported}")

        if info.embedding_domain == EmbeddingDomain.DCT:
            # DCT embedding via jpegio not available on Windows/Python 3.14.
            # Fall back to spatial LSB on PNG — lossless, reliable, fully tested.
            from core.embedder import embed as spatial_embed
            dst_path = TEMP_DIR / f"{uuid.uuid4().hex}_stego.png"
            spatial_embed(str(src_path), message, str(dst_path))
            download_name = Path(file.filename).stem + "_stego.png"

        elif info.embedding_domain == EmbeddingDomain.SPATIAL:
            from core.embedder import embed as spatial_embed
            dst_path = TEMP_DIR / f"{uuid.uuid4().hex}_stego.png"
            spatial_embed(str(src_path), message, str(dst_path))
            download_name = Path(file.filename).stem + "_stego.png"

        else:
            raise HTTPException(
                status_code = 400,
                detail      = f"Format '{info.actual_format.value}' is not supported for embedding."
            )

        return FileResponse(
            path       = str(dst_path),
            media_type = "application/octet-stream",
            filename   = download_name,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code = 500,
            detail      = f"Embedding failed: {str(e)}"
        )
    finally:
        if src_path:
            cleanup(src_path)

@app.post("/api/decode")
async def decode_endpoint(file: UploadFile = File(...)):
    """
    Attempt to decode a hidden message from an uploaded image.
    Returns success status, decoded message, and diagnostic info.
    """
    path = None
    try:
        path = save_upload(file)

        from core.decoder import decode
        result = decode(str(path))

        return JSONResponse(content={
            "success"         : result.success,
            "message"         : result.message if result.success else "",
            "method_used"     : result.method_used,
            "format_detected" : result.format_detected,
            "error"           : result.error,
            "warnings"        : result.warnings,
        })

    except Exception as e:
        raise HTTPException(
            status_code = 500,
            detail      = f"Decode failed: {str(e)}"
        )
    finally:
        if path:
            cleanup(path)


# ---------------------------------------------------------------------------
# Serve React frontend
# ---------------------------------------------------------------------------

STATIC_DIR = Path(__file__).parent.parent / "static"

if STATIC_DIR.exists():
    app.mount("/assets", StaticFiles(directory=str(STATIC_DIR / "assets")), name="assets")

    @app.get("/")
    async def serve_root():
        return FileResponse(str(STATIC_DIR / "index.html"))

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        """Catch-all route — serves React SPA for all non-API routes."""
        file_path = STATIC_DIR / full_path
        if file_path.exists() and file_path.is_file():
            return FileResponse(str(file_path))
        return FileResponse(str(STATIC_DIR / "index.html"))