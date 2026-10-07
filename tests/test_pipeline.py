"""
test_pipeline.py
================
Comprehensive Automated Unit & Integration Tests for Chandrayaan-2 Lunar Image Registration.
Tests:
- Scale variation & pyramid correspondence
- Rotation & viewpoint transformation
- Sun angle & illumination invariance
- Outlier rejection via RANSAC
- Spatially uniform tie-point distribution (ANMS 8x8 grid bucketing vs conventional)
- Sub-pixel correspondence refinement accuracy
- End-to-end registration pipeline execution
"""

import os
import sys
import unittest
import numpy as np
import cv2

# Add backend directory to Python path
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(TEST_DIR)
BACKEND_DIR = os.path.join(PROJECT_ROOT, "backend")
sys.path.insert(0, BACKEND_DIR)

from registration import (
    LunarRegistrationPipeline,
    LunarSceneGenerator,
    LunarFeatureExtractor,
    CorrespondenceMatcher,
    GeometricVerifier,
    SpatialTiePointSelector,
    SubPixelRefiner,
    sensor_aware_preprocess
)


class TestLunarRegistration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Initializes procedural lunar crater DEM generator for testing."""
        cls.generator = LunarSceneGenerator(size=384, seed=42)

    def test_01_preprocessing_illumination(self):
        """
        Verifies that homomorphic filtering and CLAHE successfully normalize
        opposing solar illumination angles (morning vs afternoon sun).
        """
        case = self.generator.create_synthetic_test_case("sun_angle_variation")
        src_raw = case["source_image"]
        ref_raw = case["reference_image"]

        prep_src = sensor_aware_preprocess(src_raw, apply_homomorphic=True)
        prep_ref = sensor_aware_preprocess(ref_raw, apply_homomorphic=True)

        self.assertEqual(prep_src["processed"].shape, src_raw.shape)
        self.assertEqual(prep_ref["processed"].shape, ref_raw.shape)
        self.assertEqual(prep_src["processed"].dtype, np.uint8)

        # Preprocessed images should have well-distributed dynamic range
        std_src = np.std(prep_src["processed"])
        std_ref = np.std(prep_ref["processed"])
        self.assertGreater(std_src, 20.0, "Preprocessed source contrast is too flat")
        self.assertGreater(std_ref, 20.0, "Preprocessed reference contrast is too flat")

    def test_02_scale_variation(self):
        """
        Verifies feature extraction and matching across 1.6x scale disparity
        (simulating OHRC vs TMC-2).
        """
        case = self.generator.create_synthetic_test_case("scale_variation_2x")
        pipeline = LunarRegistrationPipeline(detector_type="SIFT", use_multiscale=True)

        res = pipeline.run(case["source_image"], case["reference_image"], ground_truth=case["ground_truth"])

        self.assertTrue(res["success"], f"Registration failed: {res.get('error_message')}")
        self.assertGreaterEqual(res["telemetry"]["inliers"], 4, "Too few inliers under scale change")
        self.assertLess(res["metrics"]["matching"]["inlier_rmse_px"], 3.0, "RMSE too high under scale change")

    def test_03_rotation_viewpoint(self):
        """
        Verifies registration under 35° rotation and perspective shear.
        """
        case = self.generator.create_synthetic_test_case("rotation_viewpoint")
        pipeline = LunarRegistrationPipeline(detector_type="SIFT", model_type="AFFINE")

        res = pipeline.run(case["source_image"], case["reference_image"], ground_truth=case["ground_truth"])

        self.assertTrue(res["success"], f"Registration failed: {res.get('error_message')}")
        self.assertGreater(res["metrics"]["matching"]["inlier_ratio_pct"], 40.0)
        self.assertLess(res["metrics"]["matching"]["inlier_rmse_px"], 2.0)

        # Image similarity metrics should show dramatic improvement
        ncc_pre = res["metrics"]["image_similarity"]["pre_registration"]["ncc"]
        ncc_post = res["metrics"]["image_similarity"]["post_registration"]["ncc"]
        self.assertGreater(ncc_post, ncc_pre + 0.3, "NCC did not improve post-registration")

    def test_04_outlier_rejection(self):
        """
        Verifies that GeometricVerifier rejects injected false correspondences.
        """
        verifier = GeometricVerifier(model_type="AFFINE", ransac_thresh=3.0)

        # Create 20 true points mapped by Affine matrix: x' = 1.2*x + 10, y' = 1.2*y - 5
        pts_src = np.random.uniform(50, 300, (20, 2)).astype(np.float32)
        M_true = np.array([[1.2, 0.0, 10.0], [0.0, 1.2, -5.0]], dtype=np.float32)
        pts_ref_true = cv2.transform(pts_src.reshape(-1, 1, 2), M_true).reshape(-1, 2)

        # Inject 10 wild outlier correspondences (33% outlier ratio)
        outliers_src = np.random.uniform(50, 300, (10, 2)).astype(np.float32)
        outliers_ref = np.random.uniform(50, 300, (10, 2)).astype(np.float32)

        all_src = np.vstack([pts_src, outliers_src])
        all_ref = np.vstack([pts_ref_true, outliers_ref])

        res = verifier.verify(all_src, all_ref, (400, 400))

        self.assertTrue(res["success"])
        self.assertGreaterEqual(res["inlier_count"], 18, "RANSAC failed to retain true inliers")
        self.assertLess(res["inlier_rmse"], 1.0, "Inlier RMSE too high on clean synthetic points")

    def test_05_spatial_uniform_distribution(self):
        """
        Verifies that SpatialTiePointSelector (ANMS 8x8 Grid) achieves superior
        spatial coverage compared to naive greedy selection.
        """
        selector = SpatialTiePointSelector(grid_rows=8, grid_cols=8, max_points_per_cell=3, target_total_points=40)

        # Simulate clustered points: 60 points in top-left quadrant, 20 points dispersed elsewhere
        clustered_pts = np.random.uniform(10, 80, (60, 2)).astype(np.float32)
        dispersed_pts = np.random.uniform(100, 380, (20, 2)).astype(np.float32)
        all_pts = np.vstack([clustered_pts, dispersed_pts])

        # Clustered points artificially have high confidence scores
        confs = np.array([0.95] * 60 + [0.70] * 20, dtype=np.float32)
        errors = np.zeros(80, dtype=np.float32)

        res = selector.select_uniform_tie_points(
            pts_src=all_pts, pts_ref=all_pts, reproj_errors=errors, confidences=confs, img_shape_ref=(400, 400)
        )

        cov_adaptive = res["spatial_coverage_pct"]
        cov_conventional = res["conventional_comparison"]["conventional_coverage_pct"]

        self.assertGreater(cov_adaptive, cov_conventional,
                           f"Coverage-aware selection ({cov_adaptive}%) should exceed conventional ({cov_conventional}%)")
        self.assertGreater(res["spatial_entropy"], 1.5, "Spatial entropy too low")

    def test_06_subpixel_refinement_precision(self):
        """
        Verifies that SubPixelRefiner recovers fractional displacements with high precision.
        """
        refiner = SubPixelRefiner(patch_size=21, search_radius=6, min_correlation=0.35)

        # Generate smooth lunar patch
        base_img = self.generator.render_hillshade(
            self.generator.generate_lunar_dem(), sun_azimuth_deg=315.0, sun_elevation_deg=45.0
        )

        # Apply exact fractional sub-pixel shift: dx = +0.35, dy = -0.42
        true_dx, true_dy = 0.35, -0.42
        M_shift = np.array([[1.0, 0.0, true_dx], [0.0, 1.0, true_dy]], dtype=np.float32)
        shifted_img = cv2.warpAffine(base_img, M_shift, (base_img.shape[1], base_img.shape[0]), flags=cv2.INTER_CUBIC)

        pt_ref = (180.0, 180.0)
        pt_src = (180.0, 180.0)

        ref_res = refiner.refine_point(pt_src, pt_ref, shifted_img, base_img)

        self.assertTrue(ref_res["refined"], f"Refinement failed: {ref_res.get('reason')}")
        recovered_dx, recovered_dy = ref_res["displacement"]

        # When dst(x, y) = base(x + dx, y + dy), the coordinate in base moved by (-dx, -dy)
        expected_dx = -true_dx
        expected_dy = -true_dy
        error_dx = abs(recovered_dx - expected_dx)
        error_dy = abs(recovered_dy - expected_dy)
        self.assertLess(error_dx, 0.15, f"Subpixel dx error ({error_dx:.3f} px) exceeds 0.15 px tolerance")
        self.assertLess(error_dy, 0.15, f"Subpixel dy error ({error_dy:.3f} px) exceeds 0.15 px tolerance")

    def test_07_end_to_end_pipeline(self):
        """
        Full end-to-end integration test on sun angle benchmark case.
        """
        case = self.generator.create_synthetic_test_case("sun_angle_variation")
        pipeline = LunarRegistrationPipeline(detector_type="SIFT", use_homomorphic=True)

        result = pipeline.run(case["source_image"], case["reference_image"], ground_truth=case["ground_truth"])

        self.assertTrue(result["success"])
        self.assertIn("images", result)
        self.assertIn("warped_source", result["images"])
        self.assertIn("alpha_blend", result["images"])
        self.assertIn("checkerboard", result["images"])
        self.assertIn("difference_map", result["images"])
        self.assertIn("metrics", result)
        self.assertIn("timings", result)

        self.assertGreater(result["metrics"]["spatial"]["spatial_coverage_pct"], 0.0)
        self.assertGreaterEqual(result["metrics"]["subpixel"]["refined_tie_points"], 0)


if __name__ == "__main__":
    unittest.main()
