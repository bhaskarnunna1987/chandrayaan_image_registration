"""
registration.py
===============
Image Warping, Transformation Estimation, and Registration Visualization.

Provides:
- Re-estimation of transformation matrix using sub-pixel refined correspondences
- Forward/inverse mapping with bicubic interpolation
- Dynamic overlap mask calculation
- Multiple visual verification modes:
  1. Warped registered source
  2. Alpha-blended overlay (with boundary masking)
  3. Checkerboard alternating tile composite
  4. False-color residual difference map (colormap highlighting alignment errors)
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np
import cv2


class ImageRegistrar:
    """
    Handles transformation matrix fitting, image warping, and composite visualizations.
    """

    def __init__(self, interpolation: int = cv2.INTER_CUBIC):
        self.interpolation = interpolation

    def fit_transformation(self,
                           pts_src: np.ndarray,
                           pts_ref: np.ndarray,
                           model_type: str = "AFFINE") -> Optional[np.ndarray]:
        """
        Calculates least-squares transformation matrix from refined tie points.
        """
        if len(pts_src) < (3 if model_type in ["SIMILARITY", "AFFINE"] else 4):
            return None

        if model_type == "SIMILARITY":
            M, _ = cv2.estimateAffinePartial2D(pts_src, pts_ref, method=cv2.LMEDS)
            return M
        elif model_type == "AFFINE":
            M, _ = cv2.estimateAffine2D(pts_src, pts_ref, method=cv2.LMEDS)
            return M
        elif model_type == "HOMOGRAPHY":
            H, _ = cv2.findHomography(pts_src, pts_ref, method=cv2.LMEDS)
            return H
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def warp_image(self,
                   img_src: np.ndarray,
                   matrix: np.ndarray,
                   model_type: str,
                   target_shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Warps source image into reference coordinate system.

        Returns:
        - warped_img: ndarray of shape target_shape
        - overlap_mask: boolean ndarray indicating valid warped pixels
        """
        h_t, w_t = target_shape[:2]

        if model_type in ["SIMILARITY", "AFFINE"]:
            warped = cv2.warpAffine(
                img_src, matrix[:2, :3], (w_t, h_t),
                flags=self.interpolation,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            )
            # Binary mask of valid pixels
            mask_src = np.ones(img_src.shape[:2], dtype=np.uint8) * 255
            overlap_mask = cv2.warpAffine(
                mask_src, matrix[:2, :3], (w_t, h_t),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            ) > 0
        else:  # HOMOGRAPHY
            warped = cv2.warpPerspective(
                img_src, matrix, (w_t, h_t),
                flags=self.interpolation,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            )
            mask_src = np.ones(img_src.shape[:2], dtype=np.uint8) * 255
            overlap_mask = cv2.warpPerspective(
                mask_src, matrix, (w_t, h_t),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            ) > 0

        return warped, overlap_mask

    @staticmethod
    def create_alpha_blend(img_ref: np.ndarray,
                           img_warped: np.ndarray,
                           overlap_mask: np.ndarray,
                           alpha: float = 0.5) -> np.ndarray:
        """
        Creates an alpha-blended overlay inside the valid overlap region.
        Outside the overlap region, shows the reference image.
        """
        ref_f = img_ref.astype(np.float32)
        warp_f = img_warped.astype(np.float32)

        blend_f = ref_f.copy()
        mask_3d = overlap_mask if len(ref_f.shape) == 2 else np.repeat(overlap_mask[:, :, np.newaxis], ref_f.shape[2], axis=2)

        blend_f[mask_3d] = (alpha * ref_f[mask_3d] + (1.0 - alpha) * warp_f[mask_3d])
        return np.uint8(np.clip(blend_f, 0, 255))

    @staticmethod
    def create_checkerboard(img_ref: np.ndarray,
                            img_warped: np.ndarray,
                            overlap_mask: np.ndarray,
                            tile_size: int = 48) -> np.ndarray:
        """
        Creates an alternating checkerboard pattern between reference and registered source.
        This provides immediate visual verification of boundary continuity across crater rims.
        """
        h, w = img_ref.shape[:2]
        composite = img_ref.copy()

        # Generate checkerboard boolean mask
        r_indices = (np.arange(h) // tile_size) % 2
        c_indices = (np.arange(w) // tile_size) % 2
        checker_mask = (r_indices[:, None] ^ c_indices[None, :]) > 0

        # Replace checker tiles with warped source where valid
        apply_mask = checker_mask & overlap_mask
        if len(composite.shape) == 3:
            for ch in range(composite.shape[2]):
                composite[apply_mask, ch] = img_warped[apply_mask, ch]
        else:
            composite[apply_mask] = img_warped[apply_mask]

        return composite

    @staticmethod
    def create_difference_map(img_ref: np.ndarray,
                              img_warped: np.ndarray,
                              overlap_mask: np.ndarray) -> np.ndarray:
        """
        Computes absolute intensity difference in the overlap region and maps it
        to a JET colormap for clear visual inspection of residual geometric error.
        """
        diff = np.abs(img_ref.astype(np.float32) - img_warped.astype(np.float32))
        diff[~overlap_mask] = 0

        # Contrast-enhanced display
        p95 = np.percentile(diff[overlap_mask], 95) if np.any(overlap_mask) else 255.0
        norm_diff = np.clip(diff / (p95 + 1e-4) * 255.0, 0, 255).astype(np.uint8)

        # Apply colormap
        colored = cv2.applyColorMap(norm_diff, cv2.COLORMAP_JET)
        # Black out non-overlap
        colored[~overlap_mask] = [15, 20, 25]  # Dark background
        return colored

    @staticmethod
    def create_side_by_side(img_src: np.ndarray,
                            img_ref: np.ndarray,
                            max_dim: int = 600) -> np.ndarray:
        """Creates a side-by-side scaled preview of source and reference images."""
        def scale_aspect(im):
            h, w = im.shape[:2]
            scale = max_dim / float(max(h, w))
            new_w, new_h = int(w * scale), int(h * scale)
            return cv2.resize(im, (new_w, new_h), interpolation=cv2.INTER_AREA)

        s_scaled = scale_aspect(img_src)
        r_scaled = scale_aspect(img_ref)

        h = max(s_scaled.shape[0], r_scaled.shape[0])
        pad_s = np.zeros((h, s_scaled.shape[1]), dtype=s_scaled.dtype)
        pad_r = np.zeros((h, r_scaled.shape[1]), dtype=r_scaled.dtype)
        pad_s[:s_scaled.shape[0], :] = s_scaled
        pad_r[:r_scaled.shape[0], :] = r_scaled

        return np.hstack([pad_s, pad_r])
