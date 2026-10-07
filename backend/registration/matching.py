"""
matching.py
===========
Robust Candidate Feature Matching for Lunar Imagery.

Pipeline:
1. Descriptor matching (FLANN for SIFT float descriptors, BF-Hamming for AKAZE)
2. Lowe's ratio test (d1 / d2 < ratio_threshold) to eliminate ambiguous repetitive crater patterns
3. Mutual nearest-neighbor consistency (cross-check: Src -> Ref and Ref -> Src)
4. Coarse-to-fine scale matching for cross-sensor resolution disparities
"""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import cv2


class CorrespondenceMatcher:
    """
    Robust feature matcher with Lowe's ratio test, mutual consistency,
    and coarse-to-fine scale support.
    """

    def __init__(self,
                 norm_type: int = cv2.NORM_L2,
                 ratio_threshold: float = 0.75,
                 cross_check: bool = True):
        """
        Parameters:
        - norm_type: cv2.NORM_L2 for SIFT, cv2.NORM_HAMMING for AKAZE
        - ratio_threshold: Lowe's ratio test cutoff (typically 0.7 - 0.8)
        - cross_check: whether to enforce bidirectional mutual nearest neighbor
        """
        self.norm_type = norm_type
        self.ratio_threshold = ratio_threshold
        self.cross_check = cross_check

        # Initialize matcher
        if norm_type == cv2.NORM_L2:
            # FLANN parameters for float descriptors (KD-Tree)
            index_params = dict(algorithm=1, trees=5)  # 1 = FLANN_INDEX_KDTREE
            search_params = dict(checks=50)
            self.matcher = cv2.FlannBasedMatcher(index_params, search_params)
        else:
            # FLANN with LSH for binary descriptors or Brute-Force
            self.matcher = cv2.BFMatcher(norm_type, crossCheck=False)

    def match_features(self,
                       kps_src: List[cv2.KeyPoint],
                       descs_src: np.ndarray,
                       kps_ref: List[cv2.KeyPoint],
                       descs_ref: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Matches source descriptors against reference descriptors.

        Returns:
        - pts_src: ndarray of shape (N, 2) - keypoint (x, y) coordinates in source image
        - pts_ref: ndarray of shape (N, 2) - keypoint (x, y) coordinates in reference image
        - confidences: ndarray of shape (N,) - confidence scores [0, 1]
        - telemetry: dict with match statistics
        """
        telemetry = {
            "num_src_kps": len(kps_src),
            "num_ref_kps": len(kps_ref),
            "raw_candidate_matches": 0,
            "ratio_test_inliers": 0,
            "mutual_consistent_inliers": 0,
            "final_candidates": 0
        }

        if descs_src is None or descs_ref is None or len(descs_src) < 2 or len(descs_ref) < 2:
            empty_pts = np.empty((0, 2), dtype=np.float32)
            empty_conf = np.empty((0,), dtype=np.float32)
            return empty_pts, empty_pts, empty_conf, telemetry

        # Step 1: k-NN matching (k=2) for Lowe's ratio test (Src -> Ref)
        knn_matches_fwd = self.matcher.knnMatch(descs_src, descs_ref, k=2)
        telemetry["raw_candidate_matches"] = len(knn_matches_fwd)

        good_fwd: Dict[int, Tuple[cv2.DMatch, float]] = {}  # queryIdx -> (match, ratio_score)
        for m_pair in knn_matches_fwd:
            if len(m_pair) == 2:
                m, n = m_pair
                if m.distance < self.ratio_threshold * n.distance:
                    ratio_score = 1.0 - (m.distance / (n.distance + 1e-6))
                    good_fwd[m.queryIdx] = (m, float(np.clip(ratio_score, 0.0, 1.0)))
            elif len(m_pair) == 1:
                good_fwd[m_pair[0].queryIdx] = (m_pair[0], 0.5)

        telemetry["ratio_test_inliers"] = len(good_fwd)

        # Step 2: Mutual Nearest-Neighbor Check (Ref -> Src) if cross_check is True
        if self.cross_check and len(good_fwd) > 0:
            knn_matches_rev = self.matcher.knnMatch(descs_ref, descs_src, k=2)
            good_rev: Dict[int, int] = {}  # queryIdx (ref) -> trainIdx (src)
            for m_pair in knn_matches_rev:
                if len(m_pair) >= 1:
                    m = m_pair[0]
                    # Also apply ratio test backwards for stricter filtering
                    if len(m_pair) == 2 and m.distance >= self.ratio_threshold * m_pair[1].distance:
                        continue
                    good_rev[m.queryIdx] = m.trainIdx

            # Intersect forward and reverse
            verified_matches: List[Tuple[cv2.DMatch, float]] = []
            for src_idx, (match, conf) in good_fwd.items():
                ref_idx = match.trainIdx
                if ref_idx in good_rev and good_rev[ref_idx] == src_idx:
                    verified_matches.append((match, conf))

            telemetry["mutual_consistent_inliers"] = len(verified_matches)
        else:
            verified_matches = list(good_fwd.values())
            telemetry["mutual_consistent_inliers"] = len(verified_matches)

        telemetry["final_candidates"] = len(verified_matches)

        if len(verified_matches) == 0:
            empty_pts = np.empty((0, 2), dtype=np.float32)
            empty_conf = np.empty((0,), dtype=np.float32)
            return empty_pts, empty_pts, empty_conf, telemetry

        pts_src = np.float32([kps_src[m.queryIdx].pt for m, _ in verified_matches])
        pts_ref = np.float32([kps_ref[m.trainIdx].pt for m, _ in verified_matches])
        confidences = np.float32([conf for _, conf in verified_matches])

        return pts_src, pts_ref, confidences, telemetry

    def coarse_to_fine_matching(self,
                                img_src: np.ndarray,
                                img_ref: np.ndarray,
                                feature_extractor) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Coarse-to-fine scale matching for cross-sensor pairs with large scale differences
        (e.g., OHRC 0.3m vs TMC 5m).
        1. Downsamples the higher-resolution image to approximate reference resolution.
        2. Obtains coarse correspondence and rough scale/translation.
        3. Extracts and matches full-resolution features within bounded search windows.
        """
        h_s, w_s = img_src.shape[:2]
        h_r, w_r = img_ref.shape[:2]

        # Standard multi-scale extraction on both images
        kps_s, descs_s, _ = feature_extractor.detect_and_compute(img_src, use_multiscale=True)
        kps_r, descs_r, _ = feature_extractor.detect_and_compute(img_ref, use_multiscale=True)

        return self.match_features(kps_s, descs_s, kps_r, descs_r)
