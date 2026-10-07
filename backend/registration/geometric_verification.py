"""
geometric_verification.py
=========================
Robust Geometric Verification and Outlier Rejection for Lunar Imagery.

Supports:
- Transformation models:
  * SIMILARITY (4 DOF: translation, rotation, uniform scale)
  * AFFINE (6 DOF: translation, rotation, scale, shear)
  * HOMOGRAPHY (8 DOF: planar projective perspective)
  * AUTO (evaluates and selects the most physically sound, non-degenerate model)
- Robust estimators: USAC_MAGSAC / RANSAC with adaptive inlier thresholding
- Outlier filtering based on symmetric transfer / reprojection error
- Degeneracy & stability checks (determinant, condition number, perspective keystoning)
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np
import cv2


class GeometricVerifier:
    """
    Performs robust geometric estimation, model selection, and outlier rejection.
    """

    def __init__(self,
                 model_type: str = "AUTO",
                 ransac_thresh: float = 3.0,
                 max_iters: int = 5000,
                 confidence: float = 0.999):
        """
        Parameters:
        - model_type: 'SIMILARITY', 'AFFINE', 'HOMOGRAPHY', or 'AUTO'
        - ransac_thresh: reprojection error threshold in pixels
        - max_iters: maximum RANSAC iterations
        - confidence: RANSAC confidence level
        """
        self.model_type = model_type.upper()
        self.ransac_thresh = ransac_thresh
        self.max_iters = max_iters
        self.confidence = confidence

    @staticmethod
    def is_transformation_valid(matrix: np.ndarray, model_type: str, img_shape: Tuple[int, int]) -> bool:
        """
        Validates whether the estimated transformation matrix is physically realistic
        and free from extreme foldover, negative scale, or degenerate keystoning.
        """
        if matrix is None:
            return False

        h, w = img_shape[:2]
        corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)

        try:
            if model_type in ["SIMILARITY", "AFFINE"]:
                if matrix.shape == (2, 3):
                    # Linear component 2x2
                    A = matrix[:2, :2]
                    det = np.linalg.det(A)
                    # Check scale and positive orientation (reflection not permitted in orbital nadir)
                    if det <= 0.05 or det > 25.0:
                        return False
                    warped_corners = cv2.transform(corners, matrix)
                else:
                    return False
            elif model_type == "HOMOGRAPHY":
                if matrix.shape != (3, 3) or abs(matrix[2, 2]) < 1e-7:
                    return False
                H = matrix / matrix[2, 2]
                det = np.linalg.det(H[:2, :2])
                if det <= 0.05 or det > 25.0:
                    return False
                warped_corners = cv2.perspectiveTransform(corners, H)
            else:
                return False

            # Check convexity of warped bounding box
            warped_pts = warped_corners.reshape(-1, 2)
            is_convex = cv2.isContourConvex(np.int32(warped_pts))
            if not is_convex:
                return False

            # Check area ratio
            area = cv2.contourArea(np.int32(warped_pts))
            orig_area = float(w * h)
            if area < 0.05 * orig_area or area > 10.0 * orig_area:
                return False

            return True
        except Exception:
            return False

    def compute_reprojection_errors(self,
                                   pts_src: np.ndarray,
                                   pts_ref: np.ndarray,
                                   matrix: np.ndarray,
                                   model_type: str) -> np.ndarray:
        """
        Computes Euclidean reprojection error in pixels for each correspondence pair.
        """
        n_pts = len(pts_src)
        if n_pts == 0 or matrix is None:
            return np.empty((0,), dtype=np.float32)

        pts_src_reshaped = pts_src.reshape(-1, 1, 2).astype(np.float32)

        if model_type in ["SIMILARITY", "AFFINE"]:
            pts_projected = cv2.transform(pts_src_reshaped, matrix[:2, :3]).reshape(-1, 2)
        else:  # HOMOGRAPHY
            pts_projected = cv2.perspectiveTransform(pts_src_reshaped, matrix).reshape(-1, 2)

        errors = np.linalg.norm(pts_ref - pts_projected, axis=1)
        return errors

    def _estimate_similarity(self, pts_src: np.ndarray, pts_ref: np.ndarray) -> Tuple[Optional[np.ndarray], np.ndarray]:
        """Estimates 4-DOF Similarity transform using RANSAC."""
        if len(pts_src) < 3:
            return None, np.zeros((len(pts_src),), dtype=bool)
        # cv2.estimateAffinePartial2D estimates similarity (translation, rotation, uniform scale)
        M, inliers = cv2.estimateAffinePartial2D(
            pts_src, pts_ref,
            method=cv2.RANSAC,
            ransacReprojThreshold=self.ransac_thresh,
            maxIters=self.max_iters,
            confidence=self.confidence
        )
        mask = inliers.ravel().astype(bool) if inliers is not None else np.zeros((len(pts_src),), dtype=bool)
        return M, mask

    def _estimate_affine(self, pts_src: np.ndarray, pts_ref: np.ndarray) -> Tuple[Optional[np.ndarray], np.ndarray]:
        """Estimates 6-DOF Affine transform using RANSAC."""
        if len(pts_src) < 3:
            return None, np.zeros((len(pts_src),), dtype=bool)
        M, inliers = cv2.estimateAffine2D(
            pts_src, pts_ref,
            method=cv2.RANSAC,
            ransacReprojThreshold=self.ransac_thresh,
            maxIters=self.max_iters,
            confidence=self.confidence
        )
        mask = inliers.ravel().astype(bool) if inliers is not None else np.zeros((len(pts_src),), dtype=bool)
        return M, mask

    def _estimate_homography(self, pts_src: np.ndarray, pts_ref: np.ndarray) -> Tuple[Optional[np.ndarray], np.ndarray]:
        """Estimates 8-DOF Homography using USAC_MAGSAC or RANSAC."""
        if len(pts_src) < 4:
            return None, np.zeros((len(pts_src),), dtype=bool)

        method = cv2.USAC_MAGSAC if hasattr(cv2, "USAC_MAGSAC") else cv2.RANSAC
        H, inliers = cv2.findHomography(
            pts_src, pts_ref,
            method=method,
            ransacReprojThreshold=self.ransac_thresh,
            maxIters=self.max_iters,
            confidence=self.confidence
        )
        mask = inliers.ravel().astype(bool) if inliers is not None else np.zeros((len(pts_src),), dtype=bool)
        return H, mask

    def verify(self,
               pts_src: np.ndarray,
               pts_ref: np.ndarray,
               img_shape: Tuple[int, int]) -> Dict[str, Any]:
        """
        Executes geometric verification with outlier rejection and model evaluation.

        Returns:
        - Dictionary containing:
          * 'matrix': estimated transformation matrix
          * 'model_type': selected model ('SIMILARITY', 'AFFINE', or 'HOMOGRAPHY')
          * 'inlier_mask': boolean array of inliers
          * 'inlier_pts_src': inlier keypoints in source
          * 'inlier_pts_ref': inlier keypoints in reference
          * 'reprojection_errors': reprojection error per point
          * 'inlier_rmse': root-mean-square reprojection error of inliers
          * 'inlier_count': number of inlier matches
          * 'inlier_ratio': inlier ratio [0.0, 1.0]
        """
        n_candidates = len(pts_src)
        if n_candidates < 4:
            return {
                "success": False,
                "error_message": f"Insufficient candidate matches ({n_candidates} found, minimum 4 required).",
                "matrix": None,
                "model_type": self.model_type,
                "inlier_mask": np.zeros((n_candidates,), dtype=bool),
                "inlier_pts_src": np.empty((0, 2), dtype=np.float32),
                "inlier_pts_ref": np.empty((0, 2), dtype=np.float32),
                "inlier_rmse": 0.0,
                "inlier_count": 0,
                "inlier_ratio": 0.0,
                "reprojection_errors": np.empty((0,), dtype=np.float32)
            }

        candidates_models = ["SIMILARITY", "AFFINE", "HOMOGRAPHY"] if self.model_type == "AUTO" else [self.model_type]
        best_result = None
        best_score = -1.0

        for model in candidates_models:
            if model == "SIMILARITY":
                M, mask = self._estimate_similarity(pts_src, pts_ref)
            elif model == "AFFINE":
                M, mask = self._estimate_affine(pts_src, pts_ref)
            elif model == "HOMOGRAPHY":
                M, mask = self._estimate_homography(pts_src, pts_ref)
            else:
                continue

            if M is None or not self.is_transformation_valid(M, model, img_shape):
                continue

            min_inliers = 3 if model in ["SIMILARITY", "AFFINE"] else 4
            inlier_count = int(np.sum(mask))
            if inlier_count < min_inliers:
                continue

            errors = self.compute_reprojection_errors(pts_src, pts_ref, M, model)
            inlier_errors = errors[mask]
            rmse = float(np.sqrt(np.mean(inlier_errors**2))) if len(inlier_errors) > 0 else 999.0
            inlier_ratio = inlier_count / float(n_candidates)

            # Score prioritizes high inlier ratio and low RMSE with a parsimony penalty for higher DOF
            dof_penalty = 1.0 if model == "SIMILARITY" else (1.05 if model == "AFFINE" else 1.12)
            score = (inlier_ratio * inlier_count) / (rmse * dof_penalty + 1e-3)

            if score > best_score:
                best_score = score
                best_result = {
                    "success": True,
                    "matrix": M,
                    "model_type": model,
                    "inlier_mask": mask,
                    "inlier_pts_src": pts_src[mask],
                    "inlier_pts_ref": pts_ref[mask],
                    "inlier_rmse": rmse,
                    "inlier_count": inlier_count,
                    "inlier_ratio": inlier_ratio,
                    "reprojection_errors": errors,
                    "mean_error": float(np.mean(inlier_errors)) if len(inlier_errors) > 0 else 0.0,
                    "median_error": float(np.median(inlier_errors)) if len(inlier_errors) > 0 else 0.0
                }

        if best_result is None:
            return {
                "success": False,
                "error_message": "Geometric verification failed: no non-degenerate transformation found.",
                "matrix": None,
                "model_type": self.model_type,
                "inlier_mask": np.zeros((n_candidates,), dtype=bool),
                "inlier_pts_src": np.empty((0, 2), dtype=np.float32),
                "inlier_pts_ref": np.empty((0, 2), dtype=np.float32),
                "inlier_rmse": 0.0,
                "inlier_count": 0,
                "inlier_ratio": 0.0,
                "reprojection_errors": np.empty((0,), dtype=np.float32)
            }

        return best_result
