"""
chandrayaan_image_registration package
======================================
ISRO Chandrayaan-2 Optical Image Registration Reference Implementation.
"""

from .pipeline import LunarRegistrationPipeline
from .preprocessing import sensor_aware_preprocess, homomorphic_filter, apply_clahe
from .features import LunarFeatureExtractor
from .matching import CorrespondenceMatcher
from .geometric_verification import GeometricVerifier
from .spatial_selection import SpatialTiePointSelector
from .subpixel import SubPixelRefiner
from .registration import ImageRegistrar
from .metrics import evaluate_registration_metrics
from .demo_generator import LunarSceneGenerator

__all__ = [
    "LunarRegistrationPipeline",
    "sensor_aware_preprocess",
    "homomorphic_filter",
    "apply_clahe",
    "LunarFeatureExtractor",
    "CorrespondenceMatcher",
    "GeometricVerifier",
    "SpatialTiePointSelector",
    "SubPixelRefiner",
    "ImageRegistrar",
    "evaluate_registration_metrics",
    "LunarSceneGenerator"
]
