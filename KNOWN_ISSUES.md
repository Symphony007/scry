# Known Issues & Documented Failure Modes

## The Mandrill Problem
The USC-SIPI Mandrill benchmark image (4.2.03.tiff) has a naturally balanced
pixel pair distribution that scores 1.0 on chi-square even when clean.
Root cause: high-precision analog film scanning mimics the statistical
signature of steganography.

**Rule:** The Mandrill must never be used as a sole ground-truth test for
statistical detectors. Chi-square and histogram detectors are unreliable
on ALL scanned/analog images by extension.

## AI-Generated Image Problem
Diffusion model outputs (Stable Diffusion, DALL-E, Midjourney) have
near-perfect LSB entropy by construction. The entropy detector is useless
on them. RS analysis reliability is also affected by the unusual noise profile.

## Screenshot / GPU-Render Problem
Game screenshots and UI captures have structured pixel patterns that produce
false positives in entropy-based detection and false negatives in histogram
combing.

## Small Payload Problem
All statistical detectors fail on payloads below approximately 3% of image
capacity. This is inherent to statistical analysis — it must be documented,
not engineered around.

## JPEG Compression Incompatibility
LSB replacement in the spatial domain is destroyed by JPEG DCT compression.
Detection on JPEG files produces false positives from compression artifacts.
Both embedding and detection require a separate DCT-domain code path for JPEG.
