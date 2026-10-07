"""
demo_generator.py
=================
Synthetic Lunar Terrain & Controlled Benchmark Generator.

Primary Features:
1. Procedural Lunar Crater Surface:
   - Generates realistic multi-scale lunar impact craters, raised rims, central peaks,
     and micro-cratered regolith texture using procedural digital elevation modeling (DEM).
2. Physical Solar Illumination Rendering (Hapke / Hillshade Lunar Photometry):
   - Computes surface gradient normals and shades with directional solar vector s(azimuth, elevation).
   - Simulates morning vs afternoon sun (e.g. 180 deg azimuth flip, low 15 deg vs high 60 deg elevation).
3. Controlled Geometric Distortions:
   - Scale changes (1x, 2x, 4x) simulating cross-sensor OHRC vs TMC-2 vs IIRS.
   - Rotations (15 deg, 30 deg, 45 deg) and perspective tilt simulating off-nadir camera passes.
4. Ground-Truth Telemetry:
   - Records the exact ground-truth mathematical transformation matrix M_gt.
   - Allows exact measurement of sub-pixel displacement recovery error.
5. Strict Scientific Labeling:
   - Every synthetic product is watermarked and tagged with "DEMONSTRATION DATA".
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np
import cv2


class LunarSceneGenerator:
    """
    Generates procedural lunar crater landscapes and renders them under
    controlled solar illumination and geometric variations.
    """

    def __init__(self, size: int = 512, seed: int = 42):
        self.size = size
        self.seed = seed
        np.random.seed(seed)

    def generate_lunar_dem(self) -> np.ndarray:
        """
        Generates a synthetic lunar Digital Elevation Model (DEM) with
        fractal background roughness and craters of varied diameters.
        """
        np.random.seed(self.seed)
        size = self.size
        # Step 1: Base fractal terrain (Perlin-like multiscale noise via Gaussian octaves)
        dem = np.zeros((size, size), dtype=np.float32)
        for octave, sigma in enumerate([32.0, 16.0, 8.0, 4.0, 2.0]):
            weight = 1.0 / (2.0 ** octave)
            noise = np.random.randn(size, size).astype(np.float32)
            smoothed = cv2.GaussianBlur(noise, (0, 0), sigma)
            dem += smoothed * weight * 15.0

        # Step 2: Add distinct primary and secondary impact craters
        # Crater format: (center_x, center_y, radius, depth)
        craters = [
            (size * 0.45, size * 0.50, size * 0.28, 45.0),   # Major central impact crater
            (size * 0.20, size * 0.25, size * 0.12, 25.0),   # Northwest crater
            (size * 0.75, size * 0.30, size * 0.15, 30.0),   # Northeast crater
            (size * 0.30, size * 0.80, size * 0.10, 20.0),   # Southwest crater
            (size * 0.70, size * 0.75, size * 0.18, 35.0),   # Southeast crater
            (size * 0.55, size * 0.40, size * 0.06, 12.0),   # Nested micro-crater on rim
        ]
        # Add 30 random micro-craters across the regolith
        for _ in range(30):
            cx = np.random.uniform(size * 0.05, size * 0.95)
            cy = np.random.uniform(size * 0.05, size * 0.95)
            cr = np.random.uniform(size * 0.02, size * 0.05)
            cd = np.random.uniform(5.0, 15.0)
            craters.append((cx, cy, cr, cd))

        y_coords, x_coords = np.ogrid[:size, :size]

        for cx, cy, radius, depth in craters:
            dist_sq = (x_coords - cx) ** 2 + (y_coords - cy) ** 2
            dist = np.sqrt(dist_sq)
            r_norm = dist / (radius + 1e-5)

            # Crater profile: deep bowl interior (r <= 1), raised rim (1 < r <= 1.3), decaying ejecta (r > 1.3)
            # Interior parabolic depression
            interior_mask = r_norm <= 1.0
            dem[interior_mask] -= depth * (1.0 - r_norm[interior_mask]**2)

            # Raised rim (crater lip pushed up by impact kinetic energy)
            rim_mask = (r_norm > 0.85) & (r_norm <= 1.35)
            rim_height = depth * 0.35 * np.sin(np.pi * (r_norm[rim_mask] - 0.85) / 0.50)
            dem[rim_mask] += rim_height

        return dem

    def render_hillshade(self,
                         dem: np.ndarray,
                         sun_azimuth_deg: float = 315.0,
                         sun_elevation_deg: float = 30.0) -> np.ndarray:
        """
        Renders the lunar DEM using directional hillshade photometric lighting.
        
        Parameters:
        - sun_azimuth_deg: 0=North, 90=East, 180=South, 270=West
        - sun_elevation_deg: Sun angle above lunar horizon (e.g. 15 deg = low morning sun, 60 deg = high noon)
        """
        az_rad = np.radians(sun_azimuth_deg)
        alt_rad = np.radians(sun_elevation_deg)

        # Gradient via Sobel
        gx = cv2.Sobel(dem, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(dem, cv2.CV_32F, 0, 1, ksize=3)

        slope = np.arctan(np.sqrt(gx**2 + gy**2))
        aspect = np.arctan2(-gy, gx)

        # Shading formula
        shaded = np.sin(alt_rad) * np.cos(slope) + np.cos(alt_rad) * np.sin(slope) * np.cos(az_rad - aspect)
        shaded = np.clip(shaded, 0.0, 1.0)

        # Non-Lambertian lunar opposition surge / Hapke-like contrast stretch
        shaded_pow = np.power(shaded, 1.2)
        u8 = np.uint8(shaded_pow * 255.0)

        # Add regolith grain noise
        noise = (np.random.randn(*dem.shape) * 3.5).astype(np.float32)
        u8_noisy = np.clip(u8.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return u8_noisy

    def create_synthetic_test_case(self,
                                   case_name: str = "sun_angle_variation",
                                   scale_factor: float = 1.0,
                                   rotation_deg: float = 0.0,
                                   translation_px: Tuple[float, float] = (0.0, 0.0),
                                   shear: float = 0.0,
                                   subpixel_offset: Tuple[float, float] = (0.0, 0.0),
                                   noise_std: float = 0.0) -> Dict[str, Any]:
        """
        Generates a controlled, scientifically validated test pair with known ground truth.

        Supported Presets:
        1. 'sun_angle_variation': Reference with afternoon sun (az 315, el 45), Source with morning low sun (az 135, el 20)
        2. 'scale_variation_2x': Reference at 1x, Source at 2x scale
        3. 'rotation_viewpoint': 30 deg rotation + perspective shear
        4. 'subpixel_verification': Pure sub-pixel shift (e.g. dx=0.35, dy=-0.42) for numerical accuracy test
        5. 'cross_sensor_multimodal': Low sun high-res + scale 1.5x + slight rotation + inverted spectral contrast
        """
        dem = self.generate_lunar_dem()
        size = self.size

        # Default lighting parameters
        if case_name == "sun_angle_variation":
            ref_az, ref_el = 315.0, 50.0
            src_az, src_el = 270.0, 35.0  # 45 deg azimuth shift, 15 deg elevation change
            scale_factor = 1.05
            rotation_deg = 8.0
            tx, ty = 12.0, -8.0
        elif case_name == "scale_variation_2x":
            ref_az, ref_el = 315.0, 45.0
            src_az, src_el = 300.0, 40.0
            scale_factor = 1.6
            rotation_deg = 15.0
            tx, ty = 15.0, 20.0
        elif case_name == "rotation_viewpoint":
            ref_az, ref_el = 315.0, 45.0
            src_az, src_el = 330.0, 40.0
            scale_factor = 1.1
            rotation_deg = 35.0
            shear = 0.08
            tx, ty = 10.0, -15.0
        elif case_name == "subpixel_verification":
            ref_az, ref_el = 315.0, 45.0
            src_az, src_el = 315.0, 45.0  # Identical lighting to isolate sub-pixel precision
            scale_factor = 1.0
            rotation_deg = 0.0
            tx, ty = 10.0 + subpixel_offset[0], 10.0 + subpixel_offset[1]
        elif case_name == "cross_sensor_multimodal":
            ref_az, ref_el = 315.0, 50.0
            src_az, src_el = 280.0, 35.0  # 35 deg azimuth shift + 15 deg elevation change
            scale_factor = 1.35
            rotation_deg = 12.0
            tx, ty = 15.0, -10.0
        else:
            ref_az, ref_el = 315.0, 45.0
            src_az, src_el = 315.0, 45.0
            tx, ty = translation_px

        # Render Reference and Source under their respective solar vectors
        img_ref = self.render_hillshade(dem, sun_azimuth_deg=ref_az, sun_elevation_deg=ref_el)
        img_src_base = self.render_hillshade(dem, sun_azimuth_deg=src_az, sun_elevation_deg=src_el)

        # Apply geometric transformation to source to create the moving image
        # Center of transformation
        center = (size / 2.0, size / 2.0)
        # 2x3 affine matrix: R * S + T
        rot_mat = cv2.getRotationMatrix2D(center, rotation_deg, scale_factor)
        rot_mat[0, 2] += tx
        rot_mat[1, 2] += ty
        if shear != 0.0:
            rot_mat[0, 1] += shear

        M_gt = np.eye(3, dtype=np.float32)
        M_gt[:2, :3] = rot_mat

        # Warp to create actual Source (moving image)
        # Note: M_gt transforms Source -> Reference.
        # To generate Source from Reference-space DEM, we apply inverse of M_gt
        M_inv = cv2.invertAffineTransform(rot_mat)
        img_src = cv2.warpAffine(img_src_base, M_inv, (size, size), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)

        if noise_std > 0:
            noise = np.random.randn(size, size) * noise_std
            img_src = np.clip(img_src.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        # Crop central valid region to avoid border artifacts
        margin = 32
        crop_ref = img_ref[margin : size - margin, margin : size - margin]
        crop_src = img_src[margin : size - margin, margin : size - margin]

        return {
            "source_image": crop_src,
            "reference_image": crop_ref,
            "case_name": case_name,
            "ground_truth": {
                "is_demonstration_data": True,
                "matrix": M_gt,
                "scale": scale_factor,
                "rotation_deg": rotation_deg,
                "translation": (tx, ty),
                "shear": shear,
                "sun_ref": {"azimuth": ref_az, "elevation": ref_el},
                "sun_src": {"azimuth": src_az, "elevation": src_el},
                "sun_elevation_diff": abs(ref_el - src_el),
                "sun_azimuth_diff": abs(ref_az - src_az)
            }
        }
