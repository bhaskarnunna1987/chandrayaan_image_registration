"""
features.py
===========
Multi-Scale Representation & Feature Extraction for Lunar Imagery.

Supports:
- Octave scale-space Gaussian image pyramids (1x, 0.5x, 0.25x, ...)
- SIFT detector tuned for lunar terrain (crater rims, ejecta blankets, mare plains)
- AKAZE detector with non-linear scale space (preserves sharp shadow/rim boundaries)
- Extensible LearnedFeatureModel base class for optional future deep learning integration (SuperPoint/LoFTR)
"""

from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import cv2


class LunarFeatureExtractor:
    """
    Modular feature extractor supporting SIFT, AKAZE, and multi-scale pyramid extraction.
    """

    def __init__(self,
                 detector_type: str = "SIFT",
                 n_features: int = 2000,
                 sift_contrast_threshold: float = 0.025,
                 sift_edge_threshold: float = 12.0,
                 akaze_threshold: float = 0.001):
        """
        Parameters:
        - detector_type: 'SIFT' or 'AKAZE'
        - n_features: max keypoints to retain
        - sift_contrast_threshold: low threshold enables keypoint detection in low-contrast lunar mare
        - sift_edge_threshold: suppresses noisy linear shadow boundaries
        - akaze_threshold: detector response threshold for non-linear scale space
        """
        self.detector_type = detector_type.upper()
        self.n_features = n_features
        self.sift_contrast_threshold = sift_contrast_threshold
        self.sift_edge_threshold = sift_edge_threshold
        self.akaze_threshold = akaze_threshold

        self._init_detector()

    def _init_detector(self):
        if self.detector_type == "SIFT":
            # SIFT is scale & rotation invariant and uses gradient orientation histograms
            self.detector = cv2.SIFT_create(
                nfeatures=self.n_features,
                nOctaveLayers=4,
                contrastThreshold=self.sift_contrast_threshold,
                edgeThreshold=self.sift_edge_threshold,
                sigma=1.6
            )
            self.descriptor_type = "FLOAT"
            self.norm_type = cv2.NORM_L2
        elif self.detector_type == "AKAZE":
            # AKAZE uses non-linear diffusion filtering which does not blur across crater rim edges
            self.detector = cv2.AKAZE_create(
                descriptor_type=cv2.AKAZE_DESCRIPTOR_MLDB,
                descriptor_size=0,
                descriptor_channels=3,
                threshold=self.akaze_threshold,
                nOctaves=4,
                nOctaveLayers=4,
                diffusivity=cv2.KAZE_DIFF_PM_G2
            )
            self.descriptor_type = "BINARY"
            self.norm_type = cv2.NORM_HAMMING
        else:
            raise ValueError(f"Unsupported detector type: {self.detector_type}. Choose 'SIFT' or 'AKAZE'.")

    def build_pyramid(self, img: np.ndarray, max_levels: int = 3, min_dimension: int = 256) -> List[Tuple[float, np.ndarray]]:
        """
        Constructs a dynamic Gaussian scale pyramid.
        Returns a list of (scale_factor, downscaled_image) pairs.
        Scale factor is relative to the original image (e.g. 1.0, 0.5, 0.25).
        """
        pyramid = [(1.0, img)]
        current = img
        scale = 1.0

        for _ in range(1, max_levels):
            h, w = current.shape[:2]
            if min(h, w) // 2 < min_dimension:
                break
            current = cv2.pyrDown(current)
            scale *= 0.5
            pyramid.append((scale, current))

        return pyramid

    def detect_and_compute(self,
                           img: np.ndarray,
                           use_multiscale: bool = False,
                           pyramid_levels: int = 3) -> Tuple[List[cv2.KeyPoint], np.ndarray, Dict[str, Any]]:
        """
        Detects keypoints and computes descriptors.
        If use_multiscale is True, processes image across pyramid octaves to handle
        inter-sensor scale changes (e.g., OHRC vs TMC).
        
        Returns:
        - keypoints: List of cv2.KeyPoint
        - descriptors: ndarray of shape (N, descriptor_dim)
        - telemetry: metadata dict
        """
        if img is None or img.size == 0:
            raise ValueError("Empty image provided to feature extractor.")

        if not use_multiscale:
            keypoints, descriptors = self.detector.detectAndCompute(img, None)
            telemetry = {
                "detector": self.detector_type,
                "multiscale": False,
                "keypoint_count": len(keypoints) if keypoints is not None else 0,
                "descriptor_dim": descriptors.shape[1] if descriptors is not None else 0
            }
            if descriptors is None:
                descriptors = np.empty((0, 128 if self.detector_type == "SIFT" else 61), dtype=np.float32)
                keypoints = []
            return keypoints, descriptors, telemetry

        # Multi-scale extraction across pyramid
        pyramid = self.build_pyramid(img, max_levels=pyramid_levels)
        all_keypoints: List[cv2.KeyPoint] = []
        all_descriptors_list: List[np.ndarray] = []

        for scale, pyr_img in pyramid:
            kps, descs = self.detector.detectAndCompute(pyr_img, None)
            if kps is not None and descs is not None and len(kps) > 0:
                # Scale keypoints back to original full-resolution coordinates
                inv_scale = 1.0 / scale
                for kp in kps:
                    scaled_pt = (kp.pt[0] * inv_scale, kp.pt[1] * inv_scale)
                    scaled_size = kp.size * inv_scale
                    scaled_kp = cv2.KeyPoint(
                        x=scaled_pt[0],
                        y=scaled_pt[1],
                        size=scaled_size,
                        angle=kp.angle,
                        response=kp.response,
                        octave=kp.octave,
                        class_id=kp.class_id
                    )
                    all_keypoints.append(scaled_kp)
                all_descriptors_list.append(descs)

        if len(all_descriptors_list) > 0:
            all_descriptors = np.vstack(all_descriptors_list)
        else:
            all_descriptors = np.empty((0, 128 if self.detector_type == "SIFT" else 61), dtype=np.float32)

        telemetry = {
            "detector": self.detector_type,
            "multiscale": True,
            "pyramid_levels": len(pyramid),
            "keypoint_count": len(all_keypoints),
            "descriptor_dim": all_descriptors.shape[1] if len(all_descriptors) > 0 else 0
        }

        return all_keypoints, all_descriptors, telemetry


class LearnedFeatureInterface:
    """
    Extensible interface for learned / deep-learning feature matchers (e.g. SuperPoint, LoFTR).
    Provides a clean plug-in boundary for learned model evaluation.
    Defaults to returning unavailable with clear status if weights are absent.
    """
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self.is_available = False
        self.status_message = "Learned model interface ready. No external weights loaded; fallback active."

    def extract(self, img: np.ndarray):
        raise NotImplementedError("Learned feature extraction requires external weights.")
