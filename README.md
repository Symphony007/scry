# Scry

**Steganography embedding and detection engine — a student research project.**

Live: [https://scry-8hcq.onrender.com](https://scry-8hcq.onrender.com)

Scry is a full-stack tool for hiding messages inside images (steganography) and for detecting whether a given image has been tampered with using the same techniques. It was built as a learning project to explore the intersection of signal processing, statistical analysis, and machine learning — and to understand, from first principles, why steganography is hard to detect reliably.

---

## Table of Contents

1. [What it does](#what-it-does)
2. [Architecture overview](#architecture-overview)
3. [Embedding methods](#embedding-methods)
4. [Detection pipeline](#detection-pipeline)
5. [Machine learning layer](#machine-learning-layer)
6. [The data folder](#the-data-folder)
7. [Local setup](#local-setup)
8. [Current limitations](#current-limitations)
9. [What's next](#whats-next)

---

## What it does

Scry has three core functions:

**Embed** — Hide a text message inside an image without visibly altering it. Supports four different methods with different trade-offs between capacity, imperceptibility, and robustness.

**Decode** — Given a potentially stego image, attempt to extract any hidden message from it. The decoder tries every known method in sequence and returns the first successful result.

**Detect** — Run a statistical and ML-based analysis pipeline on an image to estimate the probability that it contains hidden data. This tab shows a "Coming Soon" placeholder in the current UI. The detection pipeline, statistical detectors, and ML models all exist and work — but they are intentionally not exposed yet. Confidence calibration is incomplete, and showing uncalibrated scores to users would be misleading. The pipeline is documented in full below so the work is visible, even though the tab isn't.

---

## Architecture overview

```
scry/
├── core/               # Embedding and decoding logic
│   ├── embedder.py         — LSB Replacement
│   ├── lsb_matching_embedder.py — LSB Matching
│   ├── dwt_embedder.py     — DWT (wavelet domain)
│   ├── dct_embedder.py     — DCT (frequency domain)
│   ├── metadata_embedder.py — Metadata / zero-pixel-change
│   ├── embedder_selector.py — Chooses best method for format
│   ├── decoder.py          — Universal decoder (tries all methods)
│   ├── format_handler.py   — Format classification and routing
│   └── utils.py            — Shared image I/O utilities
│
├── detectors/          # Statistical stego detectors
│   ├── chi_square.py
│   ├── rs_analysis.py
│   ├── histogram.py
│   ├── entropy.py
│   └── aggregator.py       — Weighted score aggregation
│
├── ml/                 # Machine learning classifiers
│   ├── type_classifier.py  — Classifies image type (photo, AI, screenshot, etc.)
│   ├── type_features.py    — Feature extraction for type classification
│   └── classifier.py       — Stego detection model interface
│
├── scripts/            # Training and data preparation scripts
├── web/
│   ├── app.py              — FastAPI backend
│   ├── config.py           — Settings
│   └── frontend/           — React frontend (Vite)
│
├── tests/              # pytest test suite
├── visualizers/        # LSB plane visualizer
└── data/               # Training data (gitignored — see below)
```

The backend is FastAPI. The frontend is React + Vite, built and served as static files directly by FastAPI in production. There is no separate Node server in deployment.

**Why FastAPI?** It handles file uploads cleanly, has async support for potentially slow ML inference calls, and generates OpenAPI docs automatically — useful for debugging the API during development.

**Why serve the React build from FastAPI rather than a separate CDN?** This is a student project deployed on Render's free tier. Running two separate services would require two free-tier instances and a more complex deployment config. Serving static files from the same FastAPI process is simpler and sufficient at this scale.

**Why React + Vite?** The UI has three independent panels (Embed, Decode, Detect) with their own state, and React's component model handles that more cleanly than vanilla JS would. Vite gives fast HMR during development.

---

## Embedding methods

### LSB Replacement
The simplest method. Each pixel's least significant bit is overwritten with one bit of the message. At 1 bit per channel and 3 channels (RGB), this gives 3 bits per pixel. For a 512×512 image that's roughly 98,000 characters of capacity.

**Why it's included:** It is the foundational steganography technique. Almost everything in the literature is measured against it. Understanding its weaknesses (it creates detectable statistical patterns in the LSB plane) is the starting point for understanding why better methods exist.

**Limitation:** Detectable by chi-square analysis. The act of replacing LSBs with message bits shifts the histogram of pixel values in a statistically measurable way.

### LSB Matching (±1 Embedding)
Instead of overwriting the LSB directly, the pixel value is randomly incremented or decremented by 1 to achieve the desired bit value. This breaks the paired-pixel statistical signature that chi-square exploits.

**Why it's better:** Chi-square analysis becomes blind to it because the modification is no longer deterministic. The histogram distortion is significantly reduced.

**Why it's still detectable:** RS Analysis can still find it. The spatial correlation of natural images is disturbed by random ±1 noise, and RS Analysis measures exactly that.

**Supported formats:** PNG and TIFF only. JPEG is lossy — quantization would destroy the embedded bits. WebP is converted to PNG before embedding.

### DWT (Discrete Wavelet Transform)
Embeds data in the high-frequency wavelet coefficients of the image. The image is decomposed into subbands; the message is hidden in the detail coefficients, which correspond to edges and textures rather than smooth regions.

**Why wavelet domain?** Human vision is less sensitive to changes in high-frequency detail regions than in smooth low-frequency regions. Modifications there are harder to see and, depending on threshold tuning, can survive mild processing.

**Implementation note:** Uses `scipy.signal` for the wavelet transform. A custom implementation rather than a library like PyWavelets was chosen to maintain explicit control over the coefficient selection and modification strategy.

**Limitation:** Capacity is lower than LSB methods. The embedding is only done in the HL and LH subbands to preserve perceptual quality, which limits how much data can be stored.

### DCT (Discrete Cosine Transform)
Embeds data in the mid-frequency DCT coefficients of 8×8 pixel blocks, similar conceptually to how JPEG compression works. A format header (4 bits) encodes the target format so the decoder knows how to read it back.

**Why mid-frequency?** Low frequencies carry the bulk of visible image information. High frequencies are often discarded by compression. Mid-frequency coefficients are perceptually less critical but stable enough to survive reasonable image handling.

**Limitation:** This is a custom DCT embedder, not the F5 algorithm. F5 is the established research-grade DCT-based embedder with better steganographic security. The custom implementation is simpler and sufficient for demonstrating the concept.

### Metadata (Zero Pixel Modification)
Embeds the message in the image's metadata fields — specifically PNG `tEXt` chunks and JPEG EXIF `UserComment`. The pixel data is completely untouched.

**Why include this?** It demonstrates that not all steganography lives in the pixel data. Statistical detectors that operate on pixel values (chi-square, RS analysis, entropy, histogram analysis) are completely blind to this method — they will always score near zero on a metadata-embedded image regardless of payload size. This has practical implications: a detection pipeline that only runs pixel-level analysis gives a false sense of security.

**Format behaviour:**
- PNG → stores in `tEXt` chunk, output is PNG
- JPEG → stores in EXIF UserComment via `piexif`, output is JPEG
- TIFF and WebP → converted to PNG first for reliability; output is PNG

---

## Detection pipeline

> ### ⚠️ Not yet exposed in the UI
>
> The Detect tab currently shows a "Coming Soon" placeholder. This is a deliberate decision, not a missing feature.
>
> The four statistical detectors described below are fully implemented, tested, and passing. The aggregation layer is wired up. The type-aware weighting system works. But the stego detection models sitting behind all of this are not yet confidence-calibrated. An uncalibrated classifier will output probabilities like 0.97 or 0.03 on genuinely ambiguous images — numbers that look precise but aren't. Presenting that to a user as a detection result would be dishonest.
>
> The tab stays hidden until calibration is complete. Everything below documents what is already built.

Detection runs four statistical detectors in parallel, aggregates their scores using type-aware weights, and produces a final verdict.

### Chi-Square Analysis
Examines the histogram of pixel value pairs. LSB Replacement creates a characteristic signature: values that differ only in their LSB (e.g. 100 and 101) become artificially equalized in frequency. Chi-square measures the deviation from expected natural distribution.

**Blind to:** LSB Matching, DWT, DCT, Metadata.

### RS Analysis (Regular-Singular Analysis)
Groups pixels into blocks and classifies them as Regular, Singular, or Unusable based on a discriminant function. In natural images, the ratio of Regular to Singular groups follows a predictable pattern. Embedding data — especially LSB Matching — disturbs spatial correlation and shifts this ratio measurably.

**More robust than chi-square** for LSB Matching, but computationally heavier.

### Histogram Analysis
Compares the histogram of the image against an expected smooth distribution. Steganographic noise tends to flatten and slightly regularize histograms in ways that diverge from natural image statistics.

### Entropy Analysis
Measures local and global entropy across image regions. Natural images have spatially varying entropy (smooth regions are low entropy, textured regions are high). Steganographic embedding tends to increase entropy uniformly, especially in smooth regions where it was previously very low.

### Aggregation and type-aware weighting

The four detector scores are combined using a weighted average. Critically, the weights are not fixed — they are adjusted based on the image type classification (see ML layer below).

**Why type-aware weights?** A screenshot has very different natural statistics than a photograph. A screenshot is full of flat solid-colour regions with near-zero entropy; an entropy detector will flag it as suspicious even when clean. If the same fixed weights are used for all image types, the false positive rate on screenshots and AI-generated images is unacceptably high. Type-aware weighting compensates for this.

---

## Machine learning layer

> ### ⚠️ Partially active — read carefully
>
> The ML layer has two components in very different states of completion, and it matters to be precise about what is and isn't running.
>
> The **image type classifier** is complete, trained, and running in production on every single request right now. It is not visible in the UI but it is active. Every time a user uploads an image to embed or decode, the type classifier runs and its output adjusts how the detection weights would be applied. Users benefit from it silently.
>
> The **stego detection classifier** is a different story. The training data exists. The feature extraction pipeline exists. The model files exist. The code to call them exists. But the classifier is not connected to any user-facing endpoint. It is not called during Embed or Decode. It is only relevant to the Detect tab — which is hidden. The gap is not capability, it is calibration. Until that is done, the stego classifier is real work that simply isn't wired up to anything a user can see yet.

### Image type classifier (complete, in production)

Classifies each input image into one of five types: `photographic`, `ai_generated`, `screenshot`, `scanned`, `synthetic`.

This runs on every image before detection and its output drives the detector weight adjustments described above.

**Features used:** Statistical texture features — entropy, gradient statistics, frequency domain features, colour distribution, local variance patterns. Deliberately no deep features: the classifier needs to be fast, lightweight, and interpretable. A scikit-learn Random Forest is used.

**Training data:** ~500–1000 images per class, manually curated and stored in `data/benchmark_images/by_type/`. The data is gitignored because of size; the trained model file lives in `data/models/`.

### Stego detection classifier (trained, not yet exposed)

A second classifier trained to distinguish clean images from stego images across the four embedding methods.

**Training setup:** The `stego_prepared/` folder contains a balanced dataset of clean images and images embedded with each of the four methods (LSB Replacement, LSB Matching, DWT, Metadata). Features are extracted by `scripts/extract_stego_features.py` into `stego_features/` and `data/features.npz`. The model is trained by `scripts/train_stego_models.py`.

**What it aims to do:** Rather than treating all stego detection as a single binary problem, the goal is a per-method classifier — one model per embedding method — each trained on the specific statistical signatures of that method. The aggregation layer can then weight detector scores not just by image type but also by which embedding methods are most plausible given the image's characteristics.

**Why it's not in the UI yet:** The models exist and produce predictions, but confidence calibration is incomplete. Uncalibrated classifiers often output extreme probabilities (0.98 or 0.02) even on borderline cases, which would mislead users. Proper calibration (Platt scaling or isotonic regression via scikit-learn's `CalibratedClassifierCV`) is the next step before exposing this to users.

---

## The data folder

The `data/` directory is gitignored entirely due to size. Its structure is:

```
data/
├── benchmark_images/           # Raw image dataset for type classifier training
│   └── by_type/
│       ├── ai_generated/
│       ├── photographic/
│       ├── scanned/
│       ├── screenshot/
│       └── synthetic/
│
├── models/                     # Trained model files (.pkl / .joblib)
│
├── prepared/                   # Preprocessed images for type classifier
│   └── test/
│       ├── ai_generated/
│       ├── photographic/
│       ├── scanned/
│       ├── screenshot/
│       └── synthetic/
│
├── train/                      # Training split for type classifier
│   ├── ai_generated/
│   ├── photographic/
│   ├── scanned/
│   ├── screenshot/
│   ├── synthetic/
│   └── labels.csv
│
├── stego_prepared/             # Stego detection dataset
│   ├── test/
│   │   ├── clean/
│   │   ├── dwt/
│   │   ├── lsb_match/
│   │   └── lsb_replace/
│   └── train/
│       ├── clean/
│       ├── dwt/
│       ├── lsb_match/
│       └── lsb_replace/
│
├── stego_features/             # Extracted features for stego classifier
├── stego_labels.csv
├── feature_extraction_log.txt
└── features.npz                # Compiled feature matrix for training
```

To reproduce the training pipeline from scratch:
```bash
python scripts/prepare_dataset.py
python scripts/generate_stego_dataset.py
python scripts/extract_features.py
python scripts/extract_stego_features.py
python scripts/train_models.py
python scripts/train_stego_models.py
```

---

## Local setup

**Requirements:** Python 3.10+, Node 18+

```bash
# Clone and install Python dependencies
git clone <repo>
cd scry
pip install -r requirements.txt

# Run the backend
python start.py
# or directly:
uvicorn web.app:app --reload --port 8000
```

To run the frontend in development with hot reload:
```bash
cd web/frontend
npm install
npm run dev
# Vite dev server on http://localhost:5173
# Proxies /api/* to http://localhost:8000
```

To build and serve the production frontend through FastAPI:
```bash
cd web/frontend
npm run build
# Built files land in scry/static/
# FastAPI serves them automatically on next startup
```

**Run tests:**
```bash
pytest tests/ -v
```

---

## Current limitations

**Metadata steganography is undetectable by all four statistical detectors.** This is by design — the pixels are untouched — but it means a user who gets a "clean" detection result cannot rule out metadata-based hiding. A proper metadata scanner (checking for unexpected `tEXt` chunk keys or unusual EXIF fields) would be needed to close this gap.

**The stego detection ML is not calibrated.** The models are trained and produce reasonable accuracy in testing, but raw classifier probabilities are not reliable confidence scores. Exposing uncalibrated probabilities to users would be misleading, so the Detect tab is hidden until calibration is done.

**No adversarial robustness.** All current embedding methods are fragile — resizing, recompressing, or colour-adjusting the image will destroy the hidden message. This is a fundamental limitation of LSB-class methods. The plan was to explore adversarial/RL-based embedding that can survive transformations, but this has not been started.

**Capacity is not adaptive.** The user provides a message and the embedder either succeeds or raises an error if the message is too long. There is no UI feedback about how much of the image's capacity is being used before the user tries.

**Free tier deployment.** Render's free tier spins down after inactivity. The first request after a period of inactivity may take 30–60 seconds while the instance spins back up. This is a deployment constraint, not a code issue.

---

## What's next

**Immediate (before calling the project complete):**
- Calibrate the stego detection models and expose the Detect tab
- Add a metadata field scanner to complement the pixel-level detectors
- Show capacity usage to the user in the Embed panel

**Medium term:**
- Benchmark framework: systematic accuracy tables across image types and payload sizes, false positive rates per detector, payload detection curves
- Confidence calibration report: compare raw vs calibrated model probabilities
- F5 algorithm: replace the custom DCT embedder with the research-standard F5 for better steganographic security

**Long term (research direction):**
- Adversarial embedding: explore whether a reinforcement learning agent can learn embedding modifications that maximally confuse the statistical detectors while minimising perceptual distortion
- This would close the loop — the embedder and detector become adversaries, and each improves by training against the other

---

*Built as a student project. The goal was to understand steganography from the ground up — not to build a production security tool. Every architectural decision above reflects that learning objective.*