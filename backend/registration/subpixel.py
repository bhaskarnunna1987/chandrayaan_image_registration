"""
subpixel.py
===========
Sub-Pixel Correspondence Refinement for Lunar Imagery.

Primary Innovation #3:
Detectors output integer or coarse keypoint coordinates. For high-precision
lunar mapping, digital elevation models, and change detection, tie points must
be resolved to fractional-pixel accuracy.

Methodology:
1. For each correspondence pair (xs, ys) -> (xr, yr), extract a local patch from
   the source image, aligned with the reference orientation and scale using the
   estimated geometric transformation matrix M.
2. Perform Normalized Cross-Correlation (NCC) against a local search window
   around (xr, yr) in the reference image.
3. Locate the discrete correlation peak.
4. Fit a continuous 2D quadratic paraboloid surface to the 3x3 correlation matrix:
   f(x, y) = a*x^2 + b*y^2 + c*x*y + d*x + e*y + g
5. Analytically compute the continuous extremum (dx, dy).
6. Yields true fractional coordinates: (xr + dx, yr + dy).
"""

from typing import Tuple, List, Dict, Any, Optional
import numpy as np
import cv2


class SubPixelRefiner:
    """
    Refines tie points to sub-pixel precision using orientation-aware local patch
    Normalized Cross-Correlation (NCC) and quadratic paraboloid peak fitting.
    """

    def __init__(self,
                 patch_size: int = 21,
                 search_radius: int = 6,
                 min_correlation: float = 0.30):
        """
        Parameters:
        - patch_size: size of template patch in reference space (default 21x21)
        - search_radius: search distance in pixels around initial match (default +/- 6 px)
        - min_correlation: minimum NCC peak required to accept refinement
        """
        self.patch_size = patch_size if patch_size % 2 == 1 else patch_size + 1
        self.search_radius = search_radius
        self.search_window = self.patch_size + 2 * search_radius
        self.min_correlation = min_correlation

    @staticmethod
    def fit_paraboloid_peak(c_3x3: np.ndarray) -> Tuple[float, float, float]:
        """
        Fits a 2D quadratic surface:
        f(x, y) = a*x^2 + b*y^2 + c*x*y + d*x + e*y + g
        to a 3x3 correlation matrix centered at (0, 0) with indices [-1, 0, 1].

        Returns:
        - dx: fractional sub-pixel x-offset [-0.5, 0.5]
        - dy: fractional sub-pixel y-offset [-0.5, 0.5]
        - peak_val: estimated peak value at the sub-pixel extremum
        """
        z = c_3x3

        # Analytical least-squares coefficients for grid {-1, 0, 1} x {-1, 0, 1}:
        d = (z[1, 2] - z[1, 0]) / 2.0
        e = (z[2, 1] - z[0, 1]) / 2.0

        a = (z[1, 2] - 2.0 * z[1, 1] + z[1, 0]) / 2.0
        b = (z[2, 1] - 2.0 * z[1, 1] + z[0, 1]) / 2.0
        c = (z[2, 2] - z[2, 0] - z[0, 2] + z[0, 0]) / 4.0

        det = 4.0 * a * b - c * c
        if abs(det) < 1e-6 or a >= 0 or b >= 0:
            # Fall back to 1D parabolic peaks along orthogonal axes
            denom_x = (2.0 * z[1, 1] - z[1, 0] - z[1, 2])
            dx = (z[1, 2] - z[1, 0]) / (2.0 * denom_x) if abs(denom_x) > 1e-6 else 0.0

            denom_y = (2.0 * z[1, 1] - z[0, 1] - z[2, 1])
            dy = (z[2, 1] - z[0, 1]) / (2.0 * denom_y) if abs(denom_y) > 1e-6 else 0.0

            dx = float(np.clip(dx, -0.5, 0.5))
            dy = float(np.clip(dy, -0.5, 0.5))
            return dx, dy, float(z[1, 1])

        dx = (c * e - 2.0 * b * d) / det
        dy = (c * d - 2.0 * a * e) / det

        # Sub-pixel shift must remain within the peak pixel basin
        dx = float(np.clip(dx, -0.75, 0.75))
        dy = float(np.clip(dy, -0.75, 0.75))

        peak_val = float(a * dx**2 + b * dy**2 + c * dx * dy + d * dx + e * dy + z[1, 1])
        return dx, dy, peak_val

    def extract_aligned_patch(self,
                              img_src: np.ndarray,
                              pt_src: Tuple[float, float],
                              matrix: Optional[np.ndarray],
                              model_type: str) -> Optional[np.ndarray]:
        """
        Extracts a patch from img_src centered at pt_src, rotated and scaled
        to match the reference orientation using the estimated transformation.
        """
        xs, ys = pt_src
        p_rad = self.patch_size // 2

        if matrix is None:
            # Fallback to axis-aligned patch
            xs_i, ys_i = int(round(xs)), int(round(ys))
            h, w = img_src.shape[:2]
            if xs_i - p_rad < 0 or xs_i + p_rad >= w or ys_i - p_rad < 0 or ys_i + p_rad >= h:
                return None
            return img_src[ys_i - p_rad : ys_i + p_rad + 1, xs_i - p_rad : xs_i + p_rad + 1]

        # Construct local affine transformation that maps patch coordinates [-p_rad, p_rad]
        # around pt_src to reference image orientation
        try:
            if model_type in ["SIMILARITY", "AFFINE"]:
                A = matrix[:2, :2]
            else:  # HOMOGRAPHY
                # Local affine approximation: Jacobian at (xs, ys)
                h11, h12, h13 = matrix[0]
                h21, h22, h23 = matrix[1]
                h31, h32, h33 = matrix[2]
                den = h31 * xs + h32 * ys + h33
                if abs(den) < 1e-6:
                    return None
                A = np.array([
                    [(h11 * den - (h11 * xs + h12 * ys + h13) * h31) / (den**2),
                     (h12 * den - (h11 * xs + h12 * ys + h13) * h32) / (den**2)],
                    [(h21 * den - (h21 * xs + h22 * ys + h23) * h31) / (den**2),
                     (h22 * den - (h21 * xs + h22 * ys + h23) * h32) / (den**2)]
                ], dtype=np.float32)

            # Invert A to map output patch grid [-p_rad, p_rad] -> source coordinates
            det = np.linalg.det(A)
            if abs(det) < 1e-4:
                return None
            A_inv = np.linalg.inv(A)

            # Center of output patch is (p_rad, p_rad)
            M_patch = np.zeros((2, 3), dtype=np.float32)
            M_patch[:2, :2] = A_inv
            M_patch[:, 2] = np.array([xs, ys]) - A_inv @ np.array([p_rad, p_rad])

            patch = cv2.warpAffine(
                img_src, M_patch, (self.patch_size, self.patch_size),
                flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT
            )
            return patch
        except Exception:
            return None

    def refine_point(self,
                     pt_src: Tuple[float, float],
                     pt_ref: Tuple[float, float],
                     img_src: np.ndarray,
                     img_ref: np.ndarray,
                     matrix: Optional[np.ndarray] = None,
                     model_type: str = "AFFINE") -> Dict[str, Any]:
        """
        Refines a single correspondence pair to sub-pixel coordinates.
        """
        xs, ys = pt_src
        xr, yr = pt_ref

        h_r, w_r = img_ref.shape[:2]
        rad_p = self.patch_size // 2
        rad_s = self.search_window // 2

        xr_int, yr_int = int(round(xr)), int(round(yr))

        # Check reference search window bounds
        if (xr_int - rad_s < 0 or xr_int + rad_s >= w_r or
            yr_int - rad_s < 0 or yr_int + rad_s >= h_r):
            return {
                "refined": False,
                "pt_src_subpix": (xs, ys),
                "pt_ref_subpix": (xr, yr),
                "displacement": (0.0, 0.0),
                "displacement_mag": 0.0,
                "ncc_score": 0.0,
                "reason": "Reference search window out of image bounds"
            }

        # Extract aligned template patch from source image
        patch = self.extract_aligned_patch(img_src, (xs, ys), matrix, model_type)
        if patch is None or patch.shape != (self.patch_size, self.patch_size):
            return {
                "refined": False,
                "pt_src_subpix": (xs, ys),
                "pt_ref_subpix": (xr, yr),
                "displacement": (0.0, 0.0),
                "displacement_mag": 0.0,
                "ncc_score": 0.0,
                "reason": "Failed to extract source template patch"
            }

        # Extract reference search window
        window = img_ref[yr_int - rad_s : yr_int + rad_s + 1, xr_int - rad_s : xr_int + rad_s + 1]

        # Compute Normalized Cross-Correlation
        ncc_map = cv2.matchTemplate(window, patch, cv2.TM_CCOEFF_NORMED)

        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(ncc_map)
        peak_x, peak_y = max_loc

        if max_val < self.min_correlation:
            return {
                "refined": False,
                "pt_src_subpix": (xs, ys),
                "pt_ref_subpix": (xr, yr),
                "displacement": (0.0, 0.0),
                "displacement_mag": 0.0,
                "ncc_score": float(max_val),
                "reason": f"Correlation peak ({max_val:.2f}) below threshold ({self.min_correlation})"
            }

        # Sub-pixel peak fitting using 3x3 patch around max_loc
        map_h, map_w = ncc_map.shape
        if 0 < peak_x < map_w - 1 and 0 < peak_y < map_h - 1:
            c_3x3 = ncc_map[peak_y - 1 : peak_y + 2, peak_x - 1 : peak_x + 2]
            dx, dy, fitted_peak = self.fit_paraboloid_peak(c_3x3)
        else:
            dx, dy = 0.0, 0.0
            fitted_peak = float(max_val)

        # Coordinate calculation:
        # Window starts at (xr_int - rad_s, yr_int - rad_s)
        # Match patch top-left in window is (peak_x, peak_y)
        # Patch center relative to top-left is (rad_p, rad_p)
        # Continuous center position = (xr_int - rad_s) + peak_x + dx + rad_p
        refined_xr = float((xr_int - rad_s) + peak_x + dx + rad_p)
        refined_yr = float((yr_int - rad_s) + peak_y + dy + rad_p)

        shift_x = refined_xr - xr
        shift_y = refined_yr - yr
        shift_mag = float(np.sqrt(shift_x**2 + shift_y**2))

        # Reject wild outlier jumps (> search_radius)
        if shift_mag > self.search_radius:
            return {
                "refined": False,
                "pt_src_subpix": (xs, ys),
                "pt_ref_subpix": (xr, yr),
                "displacement": (0.0, 0.0),
                "displacement_mag": 0.0,
                "ncc_score": float(fitted_peak),
                "reason": "Subpixel shift exceeded search radius"
            }

        return {
            "refined": True,
            "pt_src_subpix": (round(float(xs), 4), round(float(ys), 4)),
            "pt_ref_subpix": (round(refined_xr, 4), round(refined_yr, 4)),
            "displacement": (round(shift_x, 4), round(shift_y, 4)),
            "displacement_mag": round(shift_mag, 4),
            "ncc_score": round(float(fitted_peak), 4),
            "reason": "Success"
        }

    def refine_all(self,
                   pts_src: np.ndarray,
                   pts_ref: np.ndarray,
                   img_src: np.ndarray,
                   img_ref: np.ndarray,
                   matrix: Optional[np.ndarray] = None,
                   model_type: str = "AFFINE") -> Dict[str, Any]:
        """
        Batch refines array of correspondences to sub-pixel coordinates.
        """
        n_pts = len(pts_src)
        refined_src = []
        refined_ref = []
        displacements = []
        scores = []
        success_count = 0

        for i in range(n_pts):
            res = self.refine_point(
                (float(pts_src[i, 0]), float(pts_src[i, 1])),
                (float(pts_ref[i, 0]), float(pts_ref[i, 1])),
                img_src, img_ref,
                matrix=matrix,
                model_type=model_type
            )
            refined_src.append(res["pt_src_subpix"])
            refined_ref.append(res["pt_ref_subpix"])
            displacements.append(res["displacement"])
            scores.append(res["ncc_score"])
            if res["refined"]:
                success_count += 1

        ref_src_arr = np.array(refined_src, dtype=np.float32)
        ref_ref_arr = np.array(refined_ref, dtype=np.float32)
        disp_arr = np.array(displacements, dtype=np.float32)
        mags = np.linalg.norm(disp_arr, axis=1)

        return {
            "pts_src_subpix": ref_src_arr,
            "pts_ref_subpix": ref_ref_arr,
            "displacements": disp_arr,
            "displacement_magnitudes": mags,
            "ncc_scores": np.array(scores, dtype=np.float32),
            "total_points": n_pts,
            "refined_count": success_count,
            "refinement_rate": round(success_count / max(n_pts, 1) * 100.0, 1),
            "mean_subpixel_shift": round(float(np.mean(mags)), 4) if len(mags) > 0 else 0.0,
            "max_subpixel_shift": round(float(np.max(mags)), 4) if len(mags) > 0 else 0.0
        }
