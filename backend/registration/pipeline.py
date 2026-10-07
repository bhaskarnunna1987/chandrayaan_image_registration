"""
pipeline.py
===========
End-to-End Orchestrator for Chandrayaan-2 Multi-Modal Lunar Image Registration.

Coordinates:
1. Sensor-aware Preprocessing (CLAHE, Homomorphic Filter, Phase/Structural representation)
2. Multi-scale Feature Extraction (SIFT / AKAZE / Scale-space pyramids)
3. Candidate Correspondence Matching (Lowe's ratio test, Mutual consistency)
4. Robust Geometric Verification (RANSAC / MAGSAC with model selection)
5. Coverage-Aware Spatial Tie-Point Selection (8x8 adaptive grid bucketing & entropy)
6. Sub-Pixel Correspondence Refinement (local patch NCC + paraboloid peak fitting)
7. Final Transformation Re-estimation & Image Warping (Bicubic interpolation)
8. Scientific Evaluation Metrics (RMSE, Inlier Ratio, SSIM, NCC, Spatial Coverage)
"""

from typing import Dict, Any, Optional, Tuple
import numpy as np
import cv2

from .preprocessing import sensor_aware_preprocess
from .features import LunarFeatureExtractor
from .matching import CorrespondenceMatcher
from .geometric_verification import GeometricVerifier
from .spatial_selection import SpatialTiePointSelector
from .subpixel import SubPixelRefiner
from .registration import ImageRegistrar
from .metrics import BenchmarkTimer, evaluate_registration_metrics


class LunarRegistrationPipeline:
    """
    Complete end-to-end scientific pipeline for lunar image registration.
    """

    def __init__(self,
                 detector_type: str = "SIFT",
                 sensor_src: str = "OHRC",
                 sensor_ref: str = "TMC",
                 model_type: str = "AUTO",
                 use_homomorphic: bool = True,
                 use_structural: bool = False,
                 use_multiscale: bool = True,
                 ratio_threshold: float = 0.75,
                 cross_check: bool = True,
                 ransac_thresh: float = 3.0,
                 grid_size: int = 8,
                 subpixel_patch_size: int = 21):
        """
        Initializes pipeline with user or preset configuration.
        """
        self.detector_type = detector_type
        self.sensor_src = sensor_src
        self.sensor_ref = sensor_ref
        self.model_type = model_type
        self.use_homomorphic = use_homomorphic
        self.use_structural = use_structural
        self.use_multiscale = use_multiscale
        self.ratio_threshold = ratio_threshold
        self.cross_check = cross_check
        self.ransac_thresh = ransac_thresh
        self.grid_size = grid_size
        self.subpixel_patch_size = subpixel_patch_size

        # Sub-modules
        self.extractor = LunarFeatureExtractor(detector_type=detector_type)
        self.matcher = CorrespondenceMatcher(
            norm_type=self.extractor.norm_type,
            ratio_threshold=ratio_threshold,
            cross_check=cross_check
        )
        self.verifier = GeometricVerifier(model_type=model_type, ransac_thresh=ransac_thresh)
        self.spatial_selector = SpatialTiePointSelector(
            grid_rows=grid_size, grid_cols=grid_size, max_points_per_cell=3, target_total_points=60
        )
        self.subpixel_refiner = SubPixelRefiner(patch_size=subpixel_patch_size)
        self.registrar = ImageRegistrar(interpolation=cv2.INTER_CUBIC)

    def run(self,
            img_src: np.ndarray,
            img_ref: np.ndarray,
            ground_truth: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Executes the full registration pipeline.

        Returns:
        - Extensive dictionary containing images, tie points, visual composites,
          and scientific evaluation metrics.
        """
        timer = BenchmarkTimer()

        # Step 1: Preprocessing
        timer.start("preprocessing")
        prep_src = sensor_aware_preprocess(
            img_src, sensor_type=self.sensor_src,
            apply_homomorphic=self.use_homomorphic,
            apply_structural=self.use_structural
        )
        prep_ref = sensor_aware_preprocess(
            img_ref, sensor_type=self.sensor_ref,
            apply_homomorphic=self.use_homomorphic,
            apply_structural=self.use_structural
        )
        timer.stop("preprocessing")

        # Step 2: Feature Detection & Description
        timer.start("feature_extraction")
        kps_src, descs_src, tel_src = self.extractor.detect_and_compute(
            prep_src["processed"], use_multiscale=self.use_multiscale
        )
        kps_ref, descs_ref, tel_ref = self.extractor.detect_and_compute(
            prep_ref["processed"], use_multiscale=self.use_multiscale
        )
        timer.stop("feature_extraction")

        # Step 3: Candidate Correspondence Matching
        timer.start("matching")
        pts_src_cand, pts_ref_cand, match_confidences, tel_match = self.matcher.match_features(
            kps_src, descs_src, kps_ref, descs_ref
        )
        timer.stop("matching")

        # Step 4: Geometric Verification & Outlier Rejection
        timer.start("geometric_verification")
        h_r, w_r = prep_ref["processed"].shape[:2]
        verification = self.verifier.verify(pts_src_cand, pts_ref_cand, (h_r, w_r))
        timer.stop("geometric_verification")

        if not verification["success"]:
            # Pipeline early exit on verification failure with graceful message
            timings = timer.get_summary()
            return {
                "success": False,
                "error_message": verification.get("error_message", "Geometric verification failed."),
                "stage_reached": "geometric_verification",
                "timings": timings,
                "telemetry": {
                    "src_keypoints": len(kps_src),
                    "ref_keypoints": len(kps_ref),
                    "candidate_matches": len(pts_src_cand),
                    "inliers": 0
                }
            }

        inlier_mask = verification["inlier_mask"]
        inlier_src = verification["inlier_pts_src"]
        inlier_ref = verification["inlier_pts_ref"]
        inlier_errors = verification["reprojection_errors"][inlier_mask]
        inlier_conf = match_confidences[inlier_mask] if len(match_confidences) == len(inlier_mask) else None

        # Step 5: Spatially Uniform Tie-Point Selection (ANMS 8x8 Grid Bucketing)
        timer.start("spatial_selection")
        spatial_result = self.spatial_selector.select_uniform_tie_points(
            pts_src=inlier_src,
            pts_ref=inlier_ref,
            reproj_errors=inlier_errors,
            confidences=inlier_conf,
            img_shape_ref=(h_r, w_r)
        )
        timer.stop("spatial_selection")

        selected_src = spatial_result["selected_pts_src"]
        selected_ref = spatial_result["selected_pts_ref"]

        # Step 6: Sub-Pixel Refinement
        timer.start("subpixel_refinement")
        subpixel_result = self.subpixel_refiner.refine_all(
            pts_src=selected_src,
            pts_ref=selected_ref,
            img_src=prep_src["processed"],
            img_ref=prep_ref["processed"],
            matrix=verification["matrix"],
            model_type=verification["model_type"]
        )
        timer.stop("subpixel_refinement")

        # Step 7: Transformation Re-estimation & Image Warping
        timer.start("registration_warping")
        refined_src = subpixel_result["pts_src_subpix"]
        refined_ref = subpixel_result["pts_ref_subpix"]

        # Re-estimate transformation using sub-pixel coordinates
        refined_matrix = self.registrar.fit_transformation(
            refined_src, refined_ref, model_type=verification["model_type"]
        )
        if refined_matrix is None:
            # Fall back to RANSAC matrix if least-squares re-fit fails
            refined_matrix = verification["matrix"]

        # Warp source image into reference coordinate system
        img_warped, overlap_mask = self.registrar.warp_image(
            img_src=prep_src["raw_normalized"],
            matrix=refined_matrix,
            model_type=verification["model_type"],
            target_shape=(h_r, w_r)
        )

        # Generate visual composites
        alpha_blend = self.registrar.create_alpha_blend(
            img_ref=prep_ref["raw_normalized"],
            img_warped=img_warped,
            overlap_mask=overlap_mask,
            alpha=0.5
        )
        checkerboard = self.registrar.create_checkerboard(
            img_ref=prep_ref["raw_normalized"],
            img_warped=img_warped,
            overlap_mask=overlap_mask,
            tile_size=48
        )
        diff_map = self.registrar.create_difference_map(
            img_ref=prep_ref["raw_normalized"],
            img_warped=img_warped,
            overlap_mask=overlap_mask
        )
        timer.stop("registration_warping")

        # Compile timings
        timings = timer.get_summary()

        # Step 8: Comprehensive Metrics Calculation
        metrics = evaluate_registration_metrics(
            img_ref=prep_ref["raw_normalized"],
            img_src=prep_src["raw_normalized"],
            img_warped=img_warped,
            overlap_mask=overlap_mask,
            verification_result=verification,
            spatial_result=spatial_result,
            subpixel_result=subpixel_result,
            timing_summary=timings,
            ground_truth_data=ground_truth
        )

        return {
            "success": True,
            "images": {
                "source": prep_src["raw_normalized"],
                "reference": prep_ref["raw_normalized"],
                "source_preprocessed": prep_src["processed"],
                "reference_preprocessed": prep_ref["processed"],
                "warped_source": img_warped,
                "alpha_blend": alpha_blend,
                "checkerboard": checkerboard,
                "difference_map": diff_map,
                "overlap_mask": overlap_mask
            },
            "points": {
                "candidate_src": pts_src_cand,
                "candidate_ref": pts_ref_cand,
                "inlier_src": inlier_src,
                "inlier_ref": inlier_ref,
                "selected_src": selected_src,
                "selected_ref": selected_ref,
                "refined_src": refined_src,
                "refined_ref": refined_ref,
                "displacements": subpixel_result["displacements"],
                "displacement_magnitudes": subpixel_result["displacement_magnitudes"]
            },
            "transformation": {
                "model_type": verification["model_type"],
                "matrix": refined_matrix.tolist() if isinstance(refined_matrix, np.ndarray) else refined_matrix,
                "initial_matrix": verification["matrix"].tolist() if isinstance(verification["matrix"], np.ndarray) else None
            },
            "metrics": metrics,
            "timings": timings,
            "telemetry": {
                "src_keypoints": len(kps_src),
                "ref_keypoints": len(kps_ref),
                "candidate_matches": len(pts_src_cand),
                "inliers": verification["inlier_count"],
                "inlier_ratio": verification["inlier_ratio"],
                "spatial_coverage_pct": spatial_result["spatial_coverage_pct"],
                "subpixel_refinement_rate": subpixel_result["refinement_rate"]
            }
        }
