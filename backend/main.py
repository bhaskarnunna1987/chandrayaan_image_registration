"""
main.py
=======
Flask REST API & Web Server for Chandrayaan-2 Multi-Modal Lunar Image Registration.
ISRO Space Technology Reference Prototype.
"""

import os
import sys
import io
import json
import base64
import time
from typing import Dict, Any, Optional

import numpy as np
import cv2
from flask import Flask, request, jsonify, send_file, render_template_string, send_from_directory

# Ensure registration package is importable
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from registration import (
    LunarRegistrationPipeline,
    LunarSceneGenerator,
    sensor_aware_preprocess
)

app = Flask(
    __name__,
    static_folder=os.path.join(os.path.dirname(current_dir), "frontend"),
    static_url_path="/static"
)

# Project root paths
PROJECT_ROOT = os.path.dirname(current_dir)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DEMO_DIR = os.path.join(DATA_DIR, "demo")
UPLOAD_DIR = os.path.join(DATA_DIR, "uploads")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Cache last registration result for live export
LATEST_RESULT: Dict[str, Any] = {}


def image_to_base64(img: np.ndarray, quality: int = 90) -> str:
    """Encodes a numpy image (grayscale or BGR) into JPEG/PNG base64 string."""
    if img is None or img.size == 0:
        return ""
    # If float, convert to uint8
    if img.dtype != np.uint8:
        p_min, p_max = np.percentile(img, (1.0, 99.0))
        if p_max > p_min:
            img = np.clip((img - p_min) / (p_max - p_min) * 255.0, 0, 255).astype(np.uint8)
        else:
            img = np.zeros_like(img, dtype=np.uint8)

    # Encode as PNG for lossless high-frequency detail
    _, buf = cv2.imencode(".png", img)
    return "data:image/png;base64," + base64.b64encode(buf).decode("utf-8")


def decode_image_upload(file_storage) -> Optional[np.ndarray]:
    """Reads uploaded image file into OpenCV ndarray."""
    try:
        in_memory_file = io.BytesIO()
        file_storage.save(in_memory_file)
        data = np.frombuffer(in_memory_file.getvalue(), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
        return img
    except Exception as e:
        print(f"Error decoding image: {e}")
        return None


@app.route("/")
def index():
    """Serves the main Scientific Mission Control Dashboard."""
    index_path = os.path.join(PROJECT_ROOT, "frontend", "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>Frontend index.html not found</h1>", 404


@app.route("/<path:path>")
def static_proxy(path):
    """Serves CSS, JS, and static assets from frontend folder."""
    frontend_dir = os.path.join(PROJECT_ROOT, "frontend")
    return send_from_directory(frontend_dir, path)


@app.route("/api/health", methods=["GET"])
def health():
    """System health check and hardware telemetry."""
    return jsonify({
        "status": "OPERATIONAL",
        "system": "ISRO Chandrayaan-2 Registration Prototype",
        "project": "Chandrayaan-2 Multi-Modal Optical Registration",
        "supported_sensors": ["OHRC (0.25m)", "TMC/TMC-2 (5m)", "IIRS (Hyper-spectral)", "LRO NAC", "Generic Lunar"],
        "algorithms": {
            "features": ["SIFT (Tuned)", "AKAZE (Non-linear FED)"],
            "illumination": ["Homomorphic Filtering", "Multi-scale CLAHE", "Structural Scharr Energy"],
            "spatial": ["8x8 Adaptive ANMS Bucketing", "Spatial Shannon Entropy"],
            "subpixel": ["Local Patch Normalized Cross-Correlation", "2D Quadratic Paraboloid Peak Fitting"],
            "verification": ["USAC_MAGSAC", "RANSAC", "Similarity/Affine/Homography Auto-Model"]
        },
        "python_version": sys.version.split()[0],
        "opencv_version": cv2.__version__
    })


REAL_MISSIONS_DIR = os.path.join(DATA_DIR, "real_missions")


@app.route("/api/demo/cases", methods=["GET"])
def get_demo_cases():
    """Returns metadata for bundled real mission datasets and synthetic benchmark scenarios."""
    manifest_file = os.path.join(DEMO_DIR, "manifest.json")
    if os.path.exists(manifest_file):
        with open(manifest_file, "r") as f:
            manifest = json.load(f)
    else:
        manifest = {}

    descriptions = {
        "sun_angle_variation": {
            "title": "Synthetic: Sun Angle Invariance (45° Azimuth Shift)",
            "challenge": "Opposite illumination: Morning (azimuth 270°, elevation 35°) vs Afternoon (azimuth 315°, elevation 50°). Severe shadow inversion across crater rims.",
            "recommended_settings": {"use_homomorphic": True, "detector": "SIFT"}
        },
        "scale_variation_2x": {
            "title": "Synthetic: Cross-Sensor Scale Disparity (1.6x - 2x)",
            "challenge": "Simulates resolution difference between Chandrayaan-2 OHRC (~0.3m) and TMC-2 (~5m) with 15° rotation.",
            "recommended_settings": {"use_multiscale": True, "detector": "SIFT"}
        },
        "rotation_viewpoint": {
            "title": "Synthetic: Orbital Viewpoint & Rotation (35°)",
            "challenge": "Off-nadir orbiter pass with 35° yaw rotation and perspective shear over central impact crater.",
            "recommended_settings": {"model_type": "AFFINE", "detector": "SIFT"}
        },
        "subpixel_verification": {
            "title": "Synthetic: Sub-Pixel Fractional Shift Benchmark",
            "challenge": "Synthetically controlled sub-pixel shift (Δx = +0.35 px, Δy = -0.42 px) to verify numerical sub-pixel recovery precision.",
            "recommended_settings": {"model_type": "SIMILARITY", "use_homomorphic": False}
        },
        "cross_sensor_multimodal": {
            "title": "Synthetic: Multi-Modal Spectral & Scale Inversion",
            "challenge": "Simulates IIRS near-infrared absorption contrast inversion combined with scale disparity and illumination shift.",
            "recommended_settings": {"use_homomorphic": True, "use_structural": True, "detector": "SIFT"}
        }
    }

    cases = [
        {
            "id": "real_ohrc_vs_lronac",
            "title": "Real Mission: Chandrayaan-2 OHRC vs LRO NAC",
            "challenge": "Real lunar crater surface captured by Chandrayaan-2 OHRC (0.25m) and LRO NAC (0.5m) with orbital yaw angle.",
            "recommended_settings": {"sensor_src": "OHRC", "sensor_ref": "LRO_NAC", "use_homomorphic": True, "use_multiscale": True},
            "is_real_mission": True
        },
        {
            "id": "real_tmc2_vs_lronac",
            "title": "Real Mission: Chandrayaan-2 TMC-2 vs LRO NAC",
            "challenge": "Real cross-scale disparity between Chandrayaan-2 TMC-2 (5m) stereo swathe and high-resolution LRO NAC (0.5m) reference.",
            "recommended_settings": {"sensor_src": "TMC", "sensor_ref": "LRO_NAC", "use_multiscale": True, "model_type": "AFFINE"},
            "is_real_mission": True
        },
        {
            "id": "real_iirs_vs_tmc2",
            "title": "Real Mission: Chandrayaan-2 IIRS vs TMC-2",
            "challenge": "Multi-modal spectral reflectance variation in Mare Tranquillitatis: IIRS near-infrared band vs TMC-2 panchromatic.",
            "recommended_settings": {"sensor_src": "IIRS", "sensor_ref": "TMC", "use_structural": True, "use_homomorphic": True},
            "is_real_mission": True
        }
    ]

    for key, data in manifest.items():
        desc = descriptions.get(key, {
            "title": key.replace("_", " ").title(),
            "challenge": "Synthetic lunar terrain challenge.",
            "recommended_settings": {}
        })
        cases.append({
            "id": key,
            "title": desc["title"],
            "challenge": desc["challenge"],
            "recommended_settings": desc["recommended_settings"],
            "ground_truth": data.get("ground_truth", {}),
            "is_real_mission": False
        })

    return jsonify({"cases": cases})


@app.route("/api/demo/load/<case_id>", methods=["GET"])
def load_demo_case(case_id):
    """Loads a specific demo or real mission case and returns preview base64 images."""
    if case_id.startswith("real_"):
        real_name = case_id[5:]
        s_path = os.path.join(REAL_MISSIONS_DIR, f"{real_name}_source.png")
        r_path = os.path.join(REAL_MISSIONS_DIR, f"{real_name}_reference.png")
        is_real = True
    else:
        s_path = os.path.join(DEMO_DIR, f"{case_id}_source.png")
        r_path = os.path.join(DEMO_DIR, f"{case_id}_reference.png")
        is_real = False

    if not os.path.exists(s_path) or not os.path.exists(r_path):
        return jsonify({"error": f"Dataset case {case_id} not found."}), 404

    img_s = cv2.imread(s_path, cv2.IMREAD_GRAYSCALE)
    img_r = cv2.imread(r_path, cv2.IMREAD_GRAYSCALE)

    manifest_file = os.path.join(DEMO_DIR, "manifest.json")
    gt = {}
    if not is_real and os.path.exists(manifest_file):
        with open(manifest_file, "r") as f:
            m = json.load(f)
            gt = m.get(case_id, {}).get("ground_truth", {})

    return jsonify({
        "case_id": case_id,
        "is_demonstration_data": not is_real,
        "is_real_mission": is_real,
        "watermark": None if is_real else "DEMONSTRATION DATA",
        "source_preview": image_to_base64(img_s),
        "reference_preview": image_to_base64(img_r),
        "source_shape": list(img_s.shape),
        "reference_shape": list(img_r.shape),
        "ground_truth": gt
    })


@app.route("/api/upload", methods=["POST"])
def upload_image():
    """Handles upload of user lunar imagery (PNG, JPG, TIFF)."""
    if "image" not in request.files:
        return jsonify({"error": "No image file provided."}), 400

    file = request.files["image"]
    image_role = request.form.get("role", "source").lower()

    img = decode_image_upload(file)
    if img is None:
        return jsonify({"error": "Invalid or corrupted image format. Please upload PNG, JPG, or TIFF."}), 400

    filename = f"upload_{image_role}_{int(time.time()*1000)}.png"
    save_path = os.path.join(UPLOAD_DIR, filename)
    cv2.imwrite(save_path, img)

    # Generate preview
    preview_b64 = image_to_base64(img)

    return jsonify({
        "status": "SUCCESS",
        "role": image_role,
        "filename": filename,
        "shape": list(img.shape),
        "preview": preview_b64
    })


@app.route("/api/pipeline/run", methods=["POST"])
def run_pipeline():
    """
    Executes the end-to-end registration pipeline.
    Accepts JSON with image references (demo_case_id or upload filenames)
    and algorithmic tuning parameters.
    """
    global LATEST_RESULT

    data = request.json or {}
    demo_case_id = data.get("demo_case_id")
    source_filename = data.get("source_filename")
    reference_filename = data.get("reference_filename")

    # Algorithm parameters
    detector_type = data.get("detector_type", "SIFT")
    sensor_src = data.get("sensor_src", "OHRC")
    sensor_ref = data.get("sensor_ref", "TMC")
    model_type = data.get("model_type", "AUTO")
    use_homomorphic = data.get("use_homomorphic", True)
    use_structural = data.get("use_structural", False)
    use_multiscale = data.get("use_multiscale", True)
    ratio_threshold = float(data.get("ratio_threshold", 0.75))
    ransac_thresh = float(data.get("ransac_thresh", 3.0))
    grid_size = int(data.get("grid_size", 8))

    gt_data = None
    img_src = None
    img_ref = None

    # Step A: Load custom uploaded source image if provided
    if source_filename:
        s_path = os.path.join(UPLOAD_DIR, source_filename)
        if os.path.exists(s_path):
            img_src = cv2.imread(s_path, cv2.IMREAD_UNCHANGED)

    # Step B: Load custom uploaded reference image if provided
    if reference_filename:
        r_path = os.path.join(UPLOAD_DIR, reference_filename)
        if os.path.exists(r_path):
            img_ref = cv2.imread(r_path, cv2.IMREAD_UNCHANGED)

    # Step C: Fallback to selected demo or real mission images for whichever is missing
    if (img_src is None or img_ref is None) and demo_case_id:
        if demo_case_id.startswith("real_"):
            real_name = demo_case_id[5:]
            s_demo = os.path.join(REAL_MISSIONS_DIR, f"{real_name}_source.png")
            r_demo = os.path.join(REAL_MISSIONS_DIR, f"{real_name}_reference.png")
            is_real_demo = True
        else:
            s_demo = os.path.join(DEMO_DIR, f"{demo_case_id}_source.png")
            r_demo = os.path.join(DEMO_DIR, f"{demo_case_id}_reference.png")
            is_real_demo = False

        if img_src is None and os.path.exists(s_demo):
            img_src = cv2.imread(s_demo, cv2.IMREAD_UNCHANGED)
        if img_ref is None and os.path.exists(r_demo):
            img_ref = cv2.imread(r_demo, cv2.IMREAD_UNCHANGED)

        # Retain synthetic ground truth only for synthetic benchmarks
        manifest_file = os.path.join(DEMO_DIR, "manifest.json")
        if not is_real_demo and os.path.exists(manifest_file) and not source_filename and not reference_filename:
            with open(manifest_file, "r") as f:
                gt_data = json.load(f).get(demo_case_id, {}).get("ground_truth")

    # Step D: Ultimate fallback to default demo images
    if img_src is None:
        s_path = os.path.join(DEMO_DIR, "sun_angle_variation_source.png")
        img_src = cv2.imread(s_path, cv2.IMREAD_UNCHANGED)
    if img_ref is None:
        r_path = os.path.join(DEMO_DIR, "sun_angle_variation_reference.png")
        img_ref = cv2.imread(r_path, cv2.IMREAD_UNCHANGED)

    # Initialize and execute pipeline
    pipeline = LunarRegistrationPipeline(
        detector_type=detector_type,
        sensor_src=sensor_src,
        sensor_ref=sensor_ref,
        model_type=model_type,
        use_homomorphic=use_homomorphic,
        use_structural=use_structural,
        use_multiscale=use_multiscale,
        ratio_threshold=ratio_threshold,
        ransac_thresh=ransac_thresh,
        grid_size=grid_size
    )

    result = pipeline.run(img_src, img_ref, ground_truth=gt_data)

    if not result["success"]:
        return jsonify({
            "success": False,
            "error_message": result.get("error_message", "Registration failed."),
            "stage_reached": result.get("stage_reached"),
            "timings": result.get("timings"),
            "telemetry": result.get("telemetry")
        }), 200

    # Cache for exports
    LATEST_RESULT = result

    # Save registered output
    registered_path = os.path.join(OUTPUT_DIR, "registered_output.png")
    cv2.imwrite(registered_path, result["images"]["warped_source"])

    # Prepare response payloads
    # Sample up to 100 tie points for frontend interactive canvas rendering
    pts_src = result["points"]["selected_src"].tolist()
    pts_ref = result["points"]["selected_ref"].tolist()
    ref_src = result["points"]["refined_src"].tolist()
    ref_ref = result["points"]["refined_ref"].tolist()
    disps = result["points"]["displacements"].tolist()

    response_payload = {
        "success": True,
        "is_demonstration_data": gt_data is not None,
        "watermark": "DEMONSTRATION DATA" if gt_data is not None else None,
        "images": {
            "source": image_to_base64(result["images"]["source"]),
            "reference": image_to_base64(result["images"]["reference"]),
            "source_preprocessed": image_to_base64(result["images"]["source_preprocessed"]),
            "reference_preprocessed": image_to_base64(result["images"]["reference_preprocessed"]),
            "warped_source": image_to_base64(result["images"]["warped_source"]),
            "alpha_blend": image_to_base64(result["images"]["alpha_blend"]),
            "checkerboard": image_to_base64(result["images"]["checkerboard"]),
            "difference_map": image_to_base64(result["images"]["difference_map"])
        },
        "points": {
            "selected_src": pts_src[:80],
            "selected_ref": pts_ref[:80],
            "refined_src": ref_src[:80],
            "refined_ref": ref_ref[:80],
            "displacements": disps[:80],
            "total_selected": len(pts_src)
        },
        "transformation": result["transformation"],
        "metrics": result["metrics"],
        "timings": result["timings"],
        "telemetry": result["telemetry"]
    }

    return jsonify(response_payload)


@app.route("/api/export/tiepoints", methods=["GET"])
def export_tiepoints():
    """Exports correspondence tie points table in CSV or JSON format."""
    global LATEST_RESULT
    if not LATEST_RESULT or "points" not in LATEST_RESULT:
        return jsonify({"error": "No registration result available to export."}), 400

    fmt = request.args.get("format", "csv").lower()
    pts = LATEST_RESULT["points"]

    sel_src = pts["selected_src"]
    sel_ref = pts["selected_ref"]
    ref_src = pts["refined_src"]
    ref_ref = pts["refined_ref"]
    disps = pts["displacements"]

    if fmt == "json":
        data = []
        for i in range(len(sel_src)):
            data.append({
                "point_id": i + 1,
                "src_raw_x": float(sel_src[i][0]),
                "src_raw_y": float(sel_src[i][1]),
                "ref_raw_x": float(sel_ref[i][0]),
                "ref_raw_y": float(sel_ref[i][1]),
                "src_subpix_x": float(ref_src[i][0]),
                "src_subpix_y": float(ref_src[i][1]),
                "ref_subpix_x": float(ref_ref[i][0]),
                "ref_subpix_y": float(ref_ref[i][1]),
                "displacement_dx": float(disps[i][0]),
                "displacement_dy": float(disps[i][1])
            })
        return jsonify({"tie_points": data, "count": len(data)})

    # CSV format (ISIS3 / Photogrammetric format compatible)
    csv_io = io.StringIO()
    csv_io.write("# ISRO Chandrayaan-2 Lunar Correspondence Tie Points\n")
    csv_io.write("# Format: PointID,Src_X,Src_Y,Ref_X,Ref_Y,Refined_Ref_X,Refined_Ref_Y,Shift_dx,Shift_dy\n")
    for i in range(len(sel_src)):
        csv_io.write(f"{i+1},{sel_src[i][0]:.3f},{sel_src[i][1]:.3f},{sel_ref[i][0]:.3f},{sel_ref[i][1]:.3f},"
                     f"{ref_ref[i][0]:.4f},{ref_ref[i][1]:.4f},{disps[i][0]:.4f},{disps[i][1]:.4f}\n")

    mem = io.BytesIO()
    mem.write(csv_io.getvalue().encode("utf-8"))
    mem.seek(0)
    return send_file(
        mem,
        mimetype="text/csv",
        as_attachment=True,
        download_name="chandrayaan2_tiepoints.csv"
    )


@app.route("/api/export/registered", methods=["GET"])
def export_registered_image():
    """Downloads the registered (warped) source image."""
    registered_path = os.path.join(OUTPUT_DIR, "registered_output.png")
    if os.path.exists(registered_path):
        return send_file(
            registered_path,
            mimetype="image/png",
            as_attachment=True,
            download_name="registered_lunar_source.png"
        )
    return jsonify({"error": "No registered image available."}), 404


if __name__ == "__main__":
    print("=" * 70)
    print("ISRO Space Technology")
    print("Chandrayaan-2 Multi-Modal Lunar Image Registration Prototype")
    print("Starting server at: http://127.0.0.1:5000")
    print("=" * 70)
    app.run(host="127.0.0.1", port=5000, debug=True)
