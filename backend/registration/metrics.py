"""
metrics.py
==========
Quantitative Evaluation Metrics for Lunar Image Registration.

Calculates:
- Matching metrics: keypoint counts, candidate matches, inlier matches, inlier ratio
- Reprojection metrics: RMSE, mean error, median error, max error
- Spatial metrics: spatial coverage percentage, spatial Shannon entropy
- Sub-pixel metrics: mean refinement displacement, refinement success rate
- Image-level similarity: Normalized Cross-Correlation (NCC) and SSIM (pre vs post)
- Processing latency per pipeline stage
"""

from typing import Dict, Any, Optional
import time
import numpy as np
import cv2


def compute_ncc(img1: np.ndarray, img2: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    """
    Computes Normalized Cross-Correlation (NCC) in range [-1.0, 1.0].
    Strictly evaluated inside the valid intersection mask.
    """
    if mask is None:
        mask = np.ones(img1.shape[:2], dtype=bool)

    if np.sum(mask) < 50:
        return 0.0

    v1 = img1[mask].astype(np.float64)
    v2 = img2[mask].astype(np.float64)

    m1, m2 = np.mean(v1), np.mean(v2)
    v1_c = v1 - m1
    v2_c = v2 - m2

    norm1 = np.sqrt(np.sum(v1_c**2))
    norm2 = np.sqrt(np.sum(v2_c**2))

    if norm1 < 1e-6 or norm2 < 1e-6:
        return 0.0

    return float(np.sum(v1_c * v2_c) / (norm1 * norm2))


def compute_ssim(img1: np.ndarray, img2: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    """
    Computes Structural Similarity Index (SSIM) in range [0, 1.0].
    Uses a standard 11x11 Gaussian window with constants c1=(0.01*255)^2, c2=(0.03*255)^2.
    """
    if mask is None:
        mask = np.ones(img1.shape[:2], dtype=bool)

    if np.sum(mask) < 50:
        return 0.0

    im1 = img1.astype(np.float64)
    im2 = img2.astype(np.float64)

    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2

    k = cv2.getGaussianKernel(11, 1.5)
    window = k @ k.T

    mu1 = cv2.filter2D(im1, -1, window)
    mu2 = cv2.filter2D(im2, -1, window)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.filter2D(im1 * im1, -1, window) - mu1_sq
    sigma2_sq = cv2.filter2D(im2 * im2, -1, window) - mu2_sq
    sigma12 = cv2.filter2D(im1 * im2, -1, window) - mu1_mu2

    ssim_map = ((2.0 * mu1_mu2 + c1) * (2.0 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))

    valid_vals = ssim_map[mask]
    return float(np.clip(np.mean(valid_vals), 0.0, 1.0)) if len(valid_vals) > 0 else 0.0


class BenchmarkTimer:
    """Convenience timer context for granular pipeline latency profiling."""

    def __init__(self):
        self.timings: Dict[str, float] = {}
        self._start_times: Dict[str, float] = {}

    def start(self, stage_name: str):
        self._start_times[stage_name] = time.perf_counter()

    def stop(self, stage_name: str):
        if stage_name in self._start_times:
            elapsed_ms = (time.perf_counter() - self._start_times[stage_name]) * 1000.0
            self.timings[stage_name] = round(elapsed_ms, 2)

    def get_summary(self) -> Dict[str, float]:
        total_ms = sum(self.timings.values())
        summary = dict(self.timings)
        summary["total_pipeline_ms"] = round(total_ms, 2)
        summary["total_pipeline_sec"] = round(total_ms / 1000.0, 3)
        return summary


def evaluate_registration_metrics(img_ref: np.ndarray,
                                 img_src: np.ndarray,
                                 img_warped: np.ndarray,
                                 overlap_mask: np.ndarray,
                                 verification_result: Dict[str, Any],
                                 spatial_result: Dict[str, Any],
                                 subpixel_result: Dict[str, Any],
                                 timing_summary: Dict[str, float],
                                 ground_truth_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Compiles complete quantitative scientific metrics report.
    """
    # Pre-registration similarity (comparing rescaled raw source to reference)
    h_r, w_r = img_ref.shape[:2]
    src_resized = cv2.resize(img_src, (w_r, h_r), interpolation=cv2.INTER_AREA)

    ncc_pre = compute_ncc(img_ref, src_resized)
    ssim_pre = compute_ssim(img_ref, src_resized)

    # Post-registration similarity in valid overlap zone
    ncc_post = compute_ncc(img_ref, img_warped, overlap_mask)
    ssim_post = compute_ssim(img_ref, img_warped, overlap_mask)

    metrics = {
        "matching": {
            "inlier_count": verification_result.get("inlier_count", 0),
            "inlier_ratio_pct": round(verification_result.get("inlier_ratio", 0.0) * 100.0, 2),
            "model_type": verification_result.get("model_type", "UNKNOWN"),
            "inlier_rmse_px": round(float(verification_result.get("inlier_rmse", 0.0)), 3),
            "mean_reproj_error_px": round(float(verification_result.get("mean_error", 0.0)), 3),
            "median_reproj_error_px": round(float(verification_result.get("median_error", 0.0)), 3)
        },
        "spatial": {
            "spatial_coverage_pct": spatial_result.get("spatial_coverage_pct", 0.0),
            "spatial_entropy": spatial_result.get("spatial_entropy", 0.0),
            "normalized_entropy": spatial_result.get("normalized_entropy", 0.0),
            "occupied_cells": spatial_result.get("occupied_cells", 0),
            "total_cells": spatial_result.get("total_cells", 64),
            "selected_tie_points": spatial_result.get("selected_count", 0),
            "conventional_comparison": spatial_result.get("conventional_comparison", {})
        },
        "subpixel": {
            "refined_tie_points": subpixel_result.get("refined_count", 0),
            "refinement_rate_pct": subpixel_result.get("refinement_rate", 0.0),
            "mean_subpixel_shift_px": subpixel_result.get("mean_subpixel_shift", 0.0),
            "max_subpixel_shift_px": subpixel_result.get("max_subpixel_shift", 0.0)
        },
        "image_similarity": {
            "pre_registration": {
                "ncc": round(ncc_pre, 4),
                "ssim": round(ssim_pre, 4)
            },
            "post_registration": {
                "ncc": round(ncc_post, 4),
                "ssim": round(ssim_post, 4)
            },
            "ncc_gain": round(ncc_post - ncc_pre, 4),
            "ssim_gain": round(ssim_post - ssim_pre, 4)
        },
        "performance": timing_summary
    }

    # If ground truth was provided (e.g. from synthetic benchmark)
    if ground_truth_data is not None:
        metrics["ground_truth_evaluation"] = {
            "is_demonstration_data": True,
            "synthetic_scale": ground_truth_data.get("scale", 1.0),
            "synthetic_rotation_deg": ground_truth_data.get("rotation_deg", 0.0),
            "synthetic_translation": ground_truth_data.get("translation", (0.0, 0.0)),
            "synthetic_sun_elevation_diff": ground_truth_data.get("sun_elevation_diff", 0.0),
            "synthetic_sun_azimuth_diff": ground_truth_data.get("sun_azimuth_diff", 0.0),
            "true_transformation_rmse": ground_truth_data.get("ground_truth_rmse", None)
        }
    else:
        metrics["ground_truth_evaluation"] = {
            "is_demonstration_data": False,
            "note": "Real imagery: metrics reflect internal consistency & inlier reprojection residuals."
        }

    return metrics
