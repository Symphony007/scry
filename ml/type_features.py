"""
Feature extractor for image type classification.

Extracts discriminating features that separate the six image types:
    Photographic, Scanned/Analog, AI-Generated, Screenshot/UI,
    Synthetic/Constructed, Mixed/Unknown

Every feature is computed from raw pixel values only — no metadata
assumptions. The feature vector is a fixed-length numpy array so it
can be fed directly into scikit-learn classifiers.
"""

import numpy as np
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Image type labels — used as classifier output classes
# ---------------------------------------------------------------------------

class ImageType:
    PHOTOGRAPHIC = "Photographic"
    SCANNED      = "Scanned"
    AI_GENERATED = "AI-Generated"
    SCREENSHOT   = "Screenshot"
    SYNTHETIC    = "Synthetic"
    UNKNOWN      = "Unknown"

ALL_IMAGE_TYPES = [
    ImageType.PHOTOGRAPHIC,
    ImageType.SCANNED,
    ImageType.AI_GENERATED,
    ImageType.SCREENSHOT,
    ImageType.SYNTHETIC,
    ImageType.UNKNOWN,
]


@dataclass
class FeatureVector:
    """
    Named container for all extracted features.
    The to_array() method returns a fixed-length numpy array
    in a consistent order for use with scikit-learn.

    Features are grouped by what property they measure:
        LSB statistics      — randomness, pair balance
        Pixel distribution  — histogram shape, multimodality
        Noise profile       — spatial frequency, smoothness
        Edge statistics     — sharpness distribution
        Color statistics    — channel correlations, saturation
        Compression artifacts — blocking, ringing
    """

    # --- LSB statistics ---
    lsb_entropy          : float = 0.0   # overall LSB randomness [0,1]
    lsb_mean             : float = 0.0   # mean LSB value [0,1]
    lsb_pair_balance     : float = 0.0   # mean |even_count - odd_count| / total
    lsb_spatial_variance : float = 0.0   # variance of LSB values across blocks

    # --- Pixel distribution ---
    pixel_mean           : float = 0.0   # mean pixel intensity
    pixel_std            : float = 0.0   # std of pixel intensity
    pixel_skewness       : float = 0.0   # histogram skewness
    pixel_kurtosis       : float = 0.0   # histogram kurtosis
    histogram_peaks      : float = 0.0   # number of dominant histogram peaks
    pixel_range_usage    : float = 0.0   # fraction of 0-255 range actually used

    # --- Noise profile ---
    noise_variance       : float = 0.0   # high-frequency noise level
    noise_spatial_corr   : float = 0.0   # spatial autocorrelation of noise
    smooth_region_frac   : float = 0.0   # fraction of image in smooth regions
    gradient_mean        : float = 0.0   # mean gradient magnitude
    gradient_std         : float = 0.0   # std of gradient magnitude

    # --- Edge statistics ---
    edge_density         : float = 0.0   # fraction of pixels classified as edges
    edge_sharpness       : float = 0.0   # mean edge gradient magnitude
    edge_linearity       : float = 0.0   # how straight edges are (UI indicator)

    # --- Color statistics ---
    channel_correlation  : float = 0.0   # mean R-G-B inter-channel correlation
    saturation_mean      : float = 0.0   # mean color saturation
    saturation_std       : float = 0.0   # std of saturation
    gray_fraction        : float = 0.0   # fraction of near-gray pixels

    # --- Compression / blocking artifacts ---
    block_boundary_delta : float = 0.0   # 8x8 block boundary discontinuity
    high_freq_energy     : float = 0.0   # energy in high spatial frequencies

    # Feature count — must match the number of fields above
    FEATURE_COUNT = 24

    def to_array(self) -> np.ndarray:
        """Return features as a fixed-length float64 numpy array."""
        return np.array([
            self.lsb_entropy,
            self.lsb_mean,
            self.lsb_pair_balance,
            self.lsb_spatial_variance,
            self.pixel_mean,
            self.pixel_std,
            self.pixel_skewness,
            self.pixel_kurtosis,
            self.histogram_peaks,
            self.pixel_range_usage,
            self.noise_variance,
            self.noise_spatial_corr,
            self.smooth_region_frac,
            self.gradient_mean,
            self.gradient_std,
            self.edge_density,
            self.edge_sharpness,
            self.edge_linearity,
            self.channel_correlation,
            self.saturation_mean,
            self.saturation_std,
            self.gray_fraction,
            self.block_boundary_delta,
            self.high_freq_energy,
        ], dtype=np.float64)

    @staticmethod
    def feature_names() -> list[str]:
        """Return feature names in the same order as to_array()."""
        return [
            "lsb_entropy", "lsb_mean", "lsb_pair_balance", "lsb_spatial_variance",
            "pixel_mean", "pixel_std", "pixel_skewness", "pixel_kurtosis",
            "histogram_peaks", "pixel_range_usage",
            "noise_variance", "noise_spatial_corr", "smooth_region_frac",
            "gradient_mean", "gradient_std",
            "edge_density", "edge_sharpness", "edge_linearity",
            "channel_correlation", "saturation_mean", "saturation_std", "gray_fraction",
            "block_boundary_delta", "high_freq_energy",
        ]


# ---------------------------------------------------------------------------
# Individual feature computation functions
# ---------------------------------------------------------------------------

def _lsb_entropy(image: np.ndarray) -> float:
    """Shannon entropy of the LSB plane across all channels."""
    lsb  = (image & 1).astype(np.float64)
    p1   = np.mean(lsb)
    p0   = 1.0 - p1
    if p1 == 0.0 or p1 == 1.0:
        return 0.0
    return float(-(p0 * np.log2(p0) + p1 * np.log2(p1)))


def _lsb_pair_balance(image: np.ndarray) -> float:
    """
    Mean absolute imbalance between adjacent pixel value pair counts.
    Low value = pairs are balanced (stego-like or AI-generated).
    High value = pairs are unbalanced (natural photographic).
    """
    scores = []
    for ch in range(3):
        counts = np.bincount(image[:, :, ch].flatten(), minlength=256).astype(np.float64)
        for i in range(0, 256, 2):
            total = counts[i] + counts[i + 1]
            if total > 0:
                scores.append(abs(counts[i] - counts[i + 1]) / total)
    return float(np.mean(scores)) if scores else 0.0


def _lsb_spatial_variance(image: np.ndarray, block_size: int = 16) -> float:
    """
    Variance of per-block LSB entropy across the image.
    Low variance = uniform LSB distribution (AI-generated, synthetic).
    High variance = structured regions mixed with random (photographic, scanned).
    """
    lsb    = (image & 1).astype(np.float64)
    h, w   = lsb.shape[:2]
    entropies = []
    for r in range(0, h - block_size + 1, block_size):
        for c in range(0, w - block_size + 1, block_size):
            block = lsb[r:r+block_size, c:c+block_size].flatten()
            p1    = np.mean(block)
            p0    = 1.0 - p1
            if 0.0 < p1 < 1.0:
                entropies.append(-(p0 * np.log2(p0) + p1 * np.log2(p1)))
            else:
                entropies.append(0.0)
    return float(np.var(entropies)) if entropies else 0.0


def _pixel_skewness(gray: np.ndarray) -> float:
    """Skewness of the grayscale pixel distribution."""
    flat  = gray.flatten().astype(np.float64)
    mean  = np.mean(flat)
    std   = np.std(flat)
    if std < 1e-8:
        return 0.0
    return float(np.mean(((flat - mean) / std) ** 3))


def _pixel_kurtosis(gray: np.ndarray) -> float:
    """Excess kurtosis of the grayscale pixel distribution."""
    flat = gray.flatten().astype(np.float64)
    mean = np.mean(flat)
    std  = np.std(flat)
    if std < 1e-8:
        return 0.0
    return float(np.mean(((flat - mean) / std) ** 4) - 3.0)


def _histogram_peaks(gray: np.ndarray) -> float:
    """
    Count dominant peaks in the grayscale histogram.
    Screenshots / UI images tend to have a small number of very sharp peaks.
    Natural photos have a smooth, broad distribution.
    """
    counts = np.bincount(gray.flatten(), minlength=256).astype(np.float64)
    # Smooth the histogram slightly to avoid noise peaks
    kernel  = np.ones(5) / 5.0
    smoothed = np.convolve(counts, kernel, mode="same")
    # A peak is a local maximum above 1% of the max count
    threshold = 0.01 * smoothed.max()
    peaks = 0
    for i in range(1, 255):
        if smoothed[i] > smoothed[i-1] and smoothed[i] > smoothed[i+1]:
            if smoothed[i] > threshold:
                peaks += 1
    return float(peaks)


def _noise_variance(gray: np.ndarray) -> float:
    """
    Estimate high-frequency noise by subtracting a smoothed version.
    High noise = scanned film grain or high-ISO photo.
    Low noise  = AI-generated, screenshot, synthetic.
    """
    from scipy.ndimage import uniform_filter
    smoothed = uniform_filter(gray.astype(np.float64), size=3)
    residual = gray.astype(np.float64) - smoothed
    return float(np.var(residual))


def _noise_spatial_corr(gray: np.ndarray) -> float:
    """
    Spatial autocorrelation of the noise residual at lag 1.
    Film grain has low autocorrelation (random).
    Structured noise (JPEG artifacts, GPU rendering) has higher correlation.
    """
    from scipy.ndimage import uniform_filter
    smoothed = uniform_filter(gray.astype(np.float64), size=3)
    residual = gray.astype(np.float64) - smoothed
    flat     = residual.flatten()
    if len(flat) < 2:
        return 0.0
    # Pearson correlation between residual[:-1] and residual[1:]
    a = flat[:-1] - np.mean(flat[:-1])
    b = flat[1:]  - np.mean(flat[1:])
    denom = np.std(a) * np.std(b)
    if denom < 1e-8:
        return 0.0
    return float(np.mean(a * b) / denom)


def _smooth_region_fraction(gray: np.ndarray, threshold: float = 5.0) -> float:
    """
    Fraction of image pixels in smooth (low-gradient) regions.
    High fraction = synthetic, AI, screenshot (large flat areas).
    Low fraction  = natural photographic content.
    """
    gy = np.abs(np.diff(gray.astype(np.float64), axis=0))
    gx = np.abs(np.diff(gray.astype(np.float64), axis=1))
    # Trim to same size
    h = min(gy.shape[0], gx.shape[0])
    w = min(gy.shape[1], gx.shape[1])
    grad_mag = np.sqrt(gy[:h, :w] ** 2 + gx[:h, :w] ** 2)
    return float(np.mean(grad_mag < threshold))


def _gradient_stats(gray: np.ndarray) -> tuple[float, float]:
    """Mean and std of the gradient magnitude across the image."""
    gy = np.abs(np.diff(gray.astype(np.float64), axis=0))
    gx = np.abs(np.diff(gray.astype(np.float64), axis=1))
    h  = min(gy.shape[0], gx.shape[0])
    w  = min(gy.shape[1], gx.shape[1])
    grad_mag = np.sqrt(gy[:h, :w] ** 2 + gx[:h, :w] ** 2)
    return float(np.mean(grad_mag)), float(np.std(grad_mag))


def _edge_stats(gray: np.ndarray) -> tuple[float, float, float]:
    """
    Edge density, mean edge sharpness, and edge linearity.
    Screenshot/UI images have high linearity (straight edges).
    Natural photos have irregular, curved edges.
    """
    # Simple Sobel-like gradient
    gy = np.diff(gray.astype(np.float64), axis=0)
    gx = np.diff(gray.astype(np.float64), axis=1)
    h  = min(gy.shape[0], gx.shape[0])
    w  = min(gy.shape[1], gx.shape[1])
    mag = np.sqrt(gy[:h, :w] ** 2 + gx[:h, :w] ** 2)

    threshold   = np.percentile(mag, 90)
    edge_mask   = mag > threshold
    edge_density  = float(np.mean(edge_mask))
    edge_sharpness = float(np.mean(mag[edge_mask])) if edge_mask.any() else 0.0

    linearity = float(np.clip(
        (np.mean(np.abs(gy[:h, :w])) + np.mean(np.abs(gx[:h, :w]))) / (np.mean(mag) + 1e-8),
        0.0, 10.0
    ) / 10.0)

    return edge_density, edge_sharpness, linearity


def _channel_correlation(image: np.ndarray) -> float:
    """
    Mean inter-channel Pearson correlation (R-G, G-B, R-B).
    High correlation = grayscale-ish or desaturated (scanned docs, synthetic).
    Low correlation  = colorful natural content.
    """
    r = image[:, :, 0].flatten().astype(np.float64)
    g = image[:, :, 1].flatten().astype(np.float64)
    b = image[:, :, 2].flatten().astype(np.float64)

    def corr(a, b):
        da, db = a - a.mean(), b - b.mean()
        denom  = np.std(da) * np.std(db)
        return float(np.mean(da * db) / denom) if denom > 1e-8 else 1.0

    return float(np.mean([corr(r, g), corr(g, b), corr(r, b)]))


def _saturation_stats(image: np.ndarray) -> tuple[float, float]:
    """
    Mean and std of pixel saturation in [0, 1].
    Computed as (max_channel - min_channel) / (max_channel + 1).
    """
    r = image[:, :, 0].astype(np.float64)
    g = image[:, :, 1].astype(np.float64)
    b = image[:, :, 2].astype(np.float64)
    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    sat  = (cmax - cmin) / (cmax + 1.0)
    return float(np.mean(sat)), float(np.std(sat))


def _gray_fraction(image: np.ndarray, tolerance: int = 10) -> float:
    """
    Fraction of pixels where R ≈ G ≈ B within tolerance.
    High fraction = grayscale / scanned document / synthetic.
    """
    r, g, b = image[:,:,0].astype(int), image[:,:,1].astype(int), image[:,:,2].astype(int)
    is_gray = (np.abs(r - g) < tolerance) & \
              (np.abs(g - b) < tolerance) & \
              (np.abs(r - b) < tolerance)
    return float(np.mean(is_gray))


def _block_boundary_delta(gray: np.ndarray, block_size: int = 8) -> float:
    """
    Mean pixel difference at 8x8 block boundaries vs within blocks.
    High delta = strong JPEG blocking artifacts.
    Low delta  = smooth image (AI-generated, PNG, synthetic).
    """
    h, w       = gray.shape
    gray_f     = gray.astype(np.float64)
    boundary_diffs = []
    interior_diffs = []

    # Horizontal boundaries
    for r in range(block_size, h, block_size):
        if r < h:
            boundary_diffs.extend(np.abs(gray_f[r, :] - gray_f[r-1, :]).tolist())

    # Interior horizontal differences (mid-block)
    for r in range(0, h - 1):
        if r % block_size != block_size - 1:
            interior_diffs.extend(np.abs(gray_f[r, :] - gray_f[r+1, :]).tolist())

    if not boundary_diffs or not interior_diffs:
        return 0.0

    return float(np.mean(boundary_diffs) - np.mean(interior_diffs))


def _high_freq_energy(gray: np.ndarray) -> float:
    """
    Energy in the high-frequency portion of the DCT spectrum.
    Computed on the full image DCT — not per-block.
    High value = lots of fine detail or noise (scanned, photographic).
    Low value  = smooth image (AI, synthetic, screenshot large areas).
    """
    from scipy.fft import dct
    gray_f  = gray.astype(np.float64) - 128.0
    # Compute 2D DCT via separable 1D DCTs
    dct_h   = dct(gray_f,   type=2, norm="ortho", axis=1)
    dct_2d  = dct(dct_h,    type=2, norm="ortho", axis=0)

    h, w    = dct_2d.shape
    # High frequency = bottom-right quadrant of DCT spectrum
    hf_block = dct_2d[h//2:, w//2:]
    total    = np.sum(dct_2d ** 2) + 1e-8
    return float(np.sum(hf_block ** 2) / total)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def extract_features(image: np.ndarray) -> FeatureVector:
    """
    Extract the full feature vector from an RGB image array.

    Args:
        image: RGB image as numpy array (H x W x 3, uint8)

    Returns:
        FeatureVector with all 24 features populated.
        On any per-feature error, that feature is set to 0.0
        and computation continues — never raises.
    """
    gray = np.mean(image, axis=2).astype(np.uint8)

    def safe(fn, *args, default=0.0):
        try:
            return fn(*args)
        except Exception:
            return default

    lsb_ent   = safe(_lsb_entropy,         image)
    lsb_mean  = float(np.mean((image & 1).astype(np.float64)))
    lsb_pair  = safe(_lsb_pair_balance,    image)
    lsb_svar  = safe(_lsb_spatial_variance, image)

    pix_mean  = float(np.mean(image.astype(np.float64)))
    pix_std   = float(np.std(image.astype(np.float64)))
    pix_skew  = safe(_pixel_skewness,      gray)
    pix_kurt  = safe(_pixel_kurtosis,      gray)
    hist_peaks= safe(_histogram_peaks,     gray)
    range_use = float(
        (int(image.max()) - int(image.min()) + 1) / 256.0
    )

    noise_var  = safe(_noise_variance,      gray)
    noise_corr = safe(_noise_spatial_corr,  gray)
    smooth_frac= safe(_smooth_region_fraction, gray)
    grad_mean, grad_std = safe(_gradient_stats, gray, default=(0.0, 0.0))

    edge_den, edge_sharp, edge_lin = safe(_edge_stats, gray, default=(0.0, 0.0, 0.0))

    ch_corr    = safe(_channel_correlation, image)
    sat_mean, sat_std = safe(_saturation_stats, image, default=(0.0, 0.0))
    gray_frac  = safe(_gray_fraction,       image)

    block_delta= safe(_block_boundary_delta, gray)
    hf_energy  = safe(_high_freq_energy,    gray)

    return FeatureVector(
        lsb_entropy          = lsb_ent,
        lsb_mean             = lsb_mean,
        lsb_pair_balance     = lsb_pair,
        lsb_spatial_variance = lsb_svar,
        pixel_mean           = pix_mean,
        pixel_std            = pix_std,
        pixel_skewness       = pix_skew,
        pixel_kurtosis       = pix_kurt,
        histogram_peaks      = hist_peaks,
        pixel_range_usage    = range_use,
        noise_variance       = noise_var,
        noise_spatial_corr   = noise_corr,
        smooth_region_frac   = smooth_frac,
        gradient_mean        = grad_mean,
        gradient_std         = grad_std,
        edge_density         = edge_den,
        edge_sharpness       = edge_sharp,
        edge_linearity       = edge_lin,
        channel_correlation  = ch_corr,
        saturation_mean      = sat_mean,
        saturation_std       = sat_std,
        gray_fraction        = gray_frac,
        block_boundary_delta = block_delta,
        high_freq_energy     = hf_energy,
    )