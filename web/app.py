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
from pathlib import Path
from PIL import Image

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

TEMP_DIR = Path(tempfile.gettempdir()) / "scry_uploads"
TEMP_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def save_upload(upload: UploadFile) -> Path:
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
    import numpy as np
    from PIL import Image as PILImage
    from core.format_handler import classify as classify_format
    from ml.type_classifier  import ImageTypeClassifier
    from detectors.chi_square  import ChiSquareDetector
    from detectors.entropy     import EntropyDetector
    from detectors.rs_analysis import RSAnalysisDetector
    from detectors.histogram   import HistogramDetector
    from detectors.aggregator  import build_type_aware_aggregator

    fmt_info    = classify_format(str(image_path))
    img         = PILImage.open(str(image_path)).convert("RGB")
    arr         = np.array(img)
    clf         = ImageTypeClassifier()
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
    path = None
    try:
        path   = save_upload(file)
        result = run_detection_pipeline(path)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")
    finally:
        if path: cleanup(path)


@app.post("/api/embed")
async def embed(
    file    : UploadFile = File(...),
    message : str        = Form(...),
    method  : str        = Form("lsb_matching"),
):
    src_path = None
    dst_path = None
    png_path = None

    VALID_METHODS = {"lsb_replacement", "lsb_matching", "metadata", "dwt"}

    try:
        if method not in VALID_METHODS:
            raise HTTPException(
                status_code = 400,
                detail      = f"Unknown method '{method}'. Valid: {sorted(VALID_METHODS)}"
            )

        src_path = save_upload(file)

        from core.format_handler import classify, EmbeddingDomain
        info = classify(str(src_path))

        # ------------------------------------------------------------------
        # Metadata — handle first, before any format conversion
        # JPEG inputs are converted to PNG — PNG tEXt chunks are more
        # reliable than JPEG EXIF for the embed/decode round-trip.
        # ------------------------------------------------------------------
        if method == "metadata":
            from core.metadata_embedder import embed_metadata
            suffix = Path(src_path).suffix.lower()

            if suffix in ('.jpg', '.jpeg'):
                png_path  = TEMP_DIR / f"{uuid.uuid4().hex}_converted.png"
                Image.open(str(src_path)).convert("RGB").save(
                    str(png_path), format="PNG"
                )
                embed_src  = str(png_path)
                out_suffix = ".png"
            else:
                embed_src  = str(src_path)
                out_suffix = suffix

            dst_path      = TEMP_DIR / f"{uuid.uuid4().hex}_stego{out_suffix}"
            embed_metadata(embed_src, message, str(dst_path))
            download_name = Path(file.filename).stem + f"_stego{out_suffix}"

        # ------------------------------------------------------------------
        # DWT — always outputs PNG regardless of input format
        # ------------------------------------------------------------------
        elif method == "dwt":
            from core.dwt_embedder import embed_dwt

            # JPEG files have near-zero HH sub-bands after lossy compression
            # so DWT finds no stable coefficients to embed into.
            # Convert to PNG first so the full frequency content is available.
            suffix = Path(src_path).suffix.lower()
            if suffix in ('.jpg', '.jpeg'):
                png_path = TEMP_DIR / f"{uuid.uuid4().hex}_converted.png"
                Image.open(str(src_path)).convert("RGB").save(
                    str(png_path), format="PNG"
                )
                embed_src = str(png_path)
            else:
                embed_src = str(src_path)

            dst_path      = TEMP_DIR / f"{uuid.uuid4().hex}_stego.png"
            result        = embed_dwt(embed_src, message, str(dst_path))
            dst_path      = Path(result["output_path"])
            download_name = Path(file.filename).stem + "_stego.png"

        # ------------------------------------------------------------------
        # Spatial methods (lsb_matching, lsb_replacement)
        # JPEG needs converting to PNG first
        # ------------------------------------------------------------------
        else:
            if info.embedding_domain == EmbeddingDomain.DCT:
                png_path = TEMP_DIR / f"{uuid.uuid4().hex}_converted.png"
                Image.open(str(src_path)).convert("RGB").save(
                    str(png_path), format="PNG"
                )
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

            download_name = Path(file.filename).stem + "_stego.png"

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
        if src_path: cleanup(src_path)
        if png_path: cleanup(png_path)
        if dst_path: cleanup(dst_path)


@app.post("/api/decode")
async def decode_endpoint(file: UploadFile = File(...)):
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
        if path: cleanup(path)


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
        file_path = STATIC_DIR / full_path
        if file_path.exists() and file_path.is_file():
            return FileResponse(str(file_path))
        return FileResponse(str(STATIC_DIR / "index.html"))