"""
check_setup.py — Phase 1 check.

Run it after installing the requirements:
    python check_setup.py

It imports every package the project needs, prints its version, and runs a tiny
test of each one. If something is broken, it tells you which package and why.
"""

import importlib
import sys

OK = "[ OK ]"
FAIL = "[FAIL]"
problems = []


def check(label, test):
    """Run one test function and print a clear pass/fail line."""
    try:
        detail = test()
        print(f"{OK} {label}" + (f" — {detail}" if detail else ""))
    except Exception as err:  # we want to catch and show *any* problem
        print(f"{FAIL} {label} — {type(err).__name__}: {err}")
        problems.append(label)


def version_of(module_name):
    module = importlib.import_module(module_name)
    return getattr(module, "__version__", "installed")


# 1. Python itself
def python_version():
    major, minor = sys.version_info[:2]
    if (major, minor) < (3, 10) or (major, minor) > (3, 12):
        raise RuntimeError(f"Python {major}.{minor} found — use Python 3.10, 3.11 or 3.12")
    return f"Python {sys.version.split()[0]} at {sys.executable}"


check("Python version", python_version)

# 2. Each package: import it and show its version
for label, module in [
    ("NumPy", "numpy"),
    ("OpenCV", "cv2"),
    ("Pillow", "PIL"),
    ("scikit-learn", "sklearn"),
    ("Streamlit", "streamlit"),
    ("fpdf2 (PDF reports)", "fpdf"),
]:
    check(label, lambda m=module: version_of(m))


# 3. MediaPipe: run face detection on a blank image (should find 0 faces, without crashing)
def mediapipe_test():
    import mediapipe as mp
    import numpy as np

    blank = np.zeros((240, 320, 3), dtype=np.uint8)
    with mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as fd:
        result = fd.process(blank)
    faces = len(result.detections) if result.detections else 0
    # Face Mesh finds the cheek areas for the skin-tone measurement
    with mp.solutions.face_mesh.FaceMesh(static_image_mode=True) as mesh:
        mesh.process(blank)
    return f"{mp.__version__}, face detection + Face Mesh ran (faces in blank image: {faces})"


check("MediaPipe", mediapipe_test)


# 4. The project's own modules and rules file
def project_test():
    from utils import face_detect, recommend, report, skin_tone  # noqa: F401

    rules = recommend.load_rules()
    rec = recommend.build_routine({"skin_type": "normal", "concerns": ["acne"]})
    return f"utils/ imports fine; {len(rules['skin_types'])} skin types, {len(rules['concerns'])} concerns; " \
           f"sample routine has {len(rec.morning)} morning steps"


check("Project modules", project_test)


# 5. Webcam (optional — for the "Use the webcam" option)
def webcam_test():
    import cv2

    cam = cv2.VideoCapture(0)
    try:
        ok, frame = cam.read()
    finally:
        cam.release()
    if not ok:
        raise RuntimeError("no webcam frame (closed lid, in use by another app, or blocked in Windows privacy settings)")
    return f"webcam works, frame size {frame.shape[1]}x{frame.shape[0]}"


if "--webcam" in sys.argv:
    check("Webcam (optional)", webcam_test)
else:
    print("[skip] Webcam — run `python check_setup.py --webcam` to test it")

print()
if problems:
    print(f"{len(problems)} problem(s): {', '.join(problems)}. See the 'Common errors' section of the README.")
    sys.exit(1)
print("All good! Your environment is ready for Glowie.")
