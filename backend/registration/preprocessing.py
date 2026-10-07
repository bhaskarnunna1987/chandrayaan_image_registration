"""
preprocessing.py
================
Sensor-Aware and Illumination-Robust Preprocessing for Lunar Imagery.

Supports:
- Modality-aware normalization (OHRC, TMC, IIRS, LRO NAC, Generic)
- Dynamic range expansion & noise mitigation
- Multi-scale CLAHE (Contrast-Limited Adaptive Histogram Equalization)
- Frequency-domain Homomorphic Filtering (attenuates low-frequency illumination gradients,
  enhances high-frequency reflectance details such as crater rims and ejecta)
- Structural and normalized gradient representations (improves stability under contrasting solar angles)
"""

from typing import Tuple, Optional, Dict, Any
import numpy as np
import cv2


def normalize_to_uint8(img: np.ndarray) -> np.ndarray:
    """
    Safely normalizes input imagery (8-bit, 10-bit, 12-bit, 16-bit or float)
    into standard 8-bit [0, 255] grayscale without saturation clipping.
    """
    if img is None or img.size == 0:
        raise ValueError("Input image is empty or invalid.")

    # Convert RGB/BGR to Grayscale if multi-channel
    if len(img.shape) == 3:
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        elif img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif img.shape[2] == 1:
            img = img[:, :, 0]

    img_f = img.astype(np.float32)
    # Robust percentile-based dynamic range clipping (0.5% to 99.5%) to remove sensor dead/hot pixels
    p_low, p_high = np.percentile(img_f, (0.5, 99.5))
    if p_high > p_low:
        img_clipped = np.clip(img_f, p_low, p_high)
        norm = (img_clipped - p_low) / (p_high - p_low) * 255.0
    else:
        norm = np.zeros_like(img_f)

    return np.uint8(np.clip(norm, 0, 255))


def apply_clahe(img_u8: np.ndarray, clip_limit: float = 3.0, tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE).
    Enhances local crater rim visibility in shadow zones without amplifying global noise.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(img_u8)


def homomorphic_filter(img_u8: np.ndarray,
                       d0: float = 25.0,
                       gamma_l: float = 0.5,
                       gamma_h: float = 1.8,
                       c: float = 1.0) -> np.ndarray:
    """
    Frequency-domain Homomorphic Filtering.
    Image formation model: I(x, y) = L(x, y) * R(x, y)
    Where L is low-frequency incident solar illumination, and R is high-frequency surface reflectance.
    ln(I) = ln(L) + ln(R)
    A high-pass filter attenuates L (solar angle gradients) while boosting R (crater topography).
    """
    img_f = img_u8.astype(np.float32) + 1.0  # Avoid log(0)
    img_log = np.log(img_f)

    rows, cols = img_log.shape
    crow, ccol = rows // 2, cols // 2

    # Construct Gaussian high-pass filter H(u, v)
    u = np.arange(rows) - crow
    v = np.arange(cols) - ccol
    u_grid, v_grid = np.meshgrid(u, v, indexing='ij')
    d_sq = u_grid**2 + v_grid**2

    # Transfer function: H = (gamma_h - gamma_l) * (1 - exp(-c * D^2 / D0^2)) + gamma_l
    h = (gamma_h - gamma_l) * (1.0 - np.exp(-c * d_sq / (d0**2 + 1e-6))) + gamma_l

    # FFT shift
    dft = np.fft.fft2(img_log)
    dft_shift = np.fft.fftshift(dft)

    # Filter in frequency domain
    filtered = dft_shift * h

    # Inverse FFT
    f_ishift = np.fft.ifftshift(filtered)
    img_back = np.fft.ifft2(f_ishift)
    img_exp = np.real(np.exp(img_back))

    # Normalize back to uint8
    min_val, max_val = np.percentile(img_exp, (1.0, 99.0))
    if max_val > min_val:
        res = (img_exp - min_val) / (max_val - min_val) * 255.0
    else:
        res = img_exp
    return np.uint8(np.clip(res, 0, 255))


def compute_structural_representation(img_u8: np.ndarray) -> np.ndarray:
    """
    Generates a structural, illumination-robust representation combining
    Scharr gradient magnitude with normalized local energy.
    This emphasizes physical surface edges (crater rims, rilles, boulders)
    while remaining less sensitive to raw intensity shifts caused by Sun angles.
    """
    blurred = cv2.GaussianBlur(img_u8, (3, 3), 0.8)

    # Scharr operators give superior gradient rotational symmetry over Sobel
    grad_x = cv2.Scharr(blurred, cv2.CV_32F, 1, 0)
    grad_y = cv2.Scharr(blurred, cv2.CV_32F, 0, 1)

    mag = np.sqrt(grad_x**2 + grad_y**2)
    # Normalize gradient magnitude with local mean suppression
    local_mean = cv2.boxFilter(mag, -1, (7, 7))
    norm_grad = mag / (local_mean + 1e-4)

    # Scale to uint8
    p99 = np.percentile(norm_grad, 99.0)
    if p99 > 0:
        norm_grad = np.clip(norm_grad / p99 * 255.0, 0, 255)
    return np.uint8(norm_grad)


def sensor_aware_preprocess(img: np.ndarray,
                            sensor_type: str = "GENERIC",
                            apply_homomorphic: bool = True,
                            apply_structural: bool = False) -> Dict[str, Any]:
    """
    Pipeline entry point for sensor-aware and illumination-robust preprocessing.
    
    Parameters:
    - img: input image (ndarray)
    - sensor_type: 'OHRC', 'TMC', 'IIRS', 'LRO_NAC', or 'GENERIC'
    - apply_homomorphic: whether to apply homomorphic filter for solar angle invariance
    - apply_structural: whether to convert to gradient/structural representation
    
    Returns:
    - Dictionary with 'processed' image, 'clahe' image, 'display_preview', and telemetry metadata.
    """
    u8 = normalize_to_uint8(img)
    meta = {
        "sensor_type": sensor_type,
        "input_shape": list(img.shape),
        "processed_shape": list(u8.shape),
        "steps_applied": ["dynamic_range_normalization"]
    }

    # Modality specific adjustments
    if sensor_type.upper() == "OHRC":
        # High resolution panchromatic (~0.25m-0.32m): fine texture preservation, subtle denoise
        u8 = cv2.bilateralFilter(u8, d=5, sigmaColor=35, sigmaSpace=5)
        clahe_clip = 2.5
        meta["steps_applied"].append("ohrc_bilateral_denoise")
    elif sensor_type.upper() == "TMC":
        # Moderate resolution panchromatic stereo (~5m): crater morphology enhancement
        clahe_clip = 3.5
        meta["steps_applied"].append("tmc_contrast_boost")
    elif sensor_type.upper() == "IIRS":
        # Hyperspectral / near-infrared: higher noise floor, broader local smoothing
        u8 = cv2.medianBlur(u8, 3)
        clahe_clip = 4.0
        meta["steps_applied"].append("iirs_spectral_band_smoothing")
    else:
        clahe_clip = 3.0

    clahe_img = apply_clahe(u8, clip_limit=clahe_clip)
    meta["steps_applied"].append(f"clahe_clip_{clahe_clip}")

    current = clahe_img
    if apply_homomorphic:
        current = homomorphic_filter(current, d0=30.0, gamma_l=0.6, gamma_h=1.6)
        meta["steps_applied"].append("homomorphic_filter")

    if apply_structural:
        current = compute_structural_representation(current)
        meta["steps_applied"].append("structural_scharr_energy")

    return {
        "processed": current,
        "clahe": clahe_img,
        "raw_normalized": u8,
        "metadata": meta
    }
