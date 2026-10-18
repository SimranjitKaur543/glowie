"""
face_detect.py — find the face in a photo and check that the photo is usable.

Main function:  check_photo(rgb_image) -> PhotoCheck

It rejects photos with no face, more than one face, a face that is too small,
or a face that is too dark, too bright or blurry — and explains how to retake it.
"""

from dataclasses import dataclass, field

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageOps

MAX_SIDE = 1280  # photos are resized to at most this many pixels on the longest side

# Quality thresholds (tuned to be forgiving; change them here if needed).
MIN_FACE_SCORE = 0.6        # how sure MediaPipe must be that something is a face (0-1)
MIN_FACE_FRACTION = 0.15    # face width must be at least 15% of the photo width
TOO_DARK = 70               # average face brightness below this (0-255) = too dark
TOO_BRIGHT = 225            # average face brightness above this = overexposed
MIN_SHARPNESS = 40          # "variance of Laplacian" below this = blurry


@dataclass
class PhotoCheck:
    ok: bool                                   # True if the photo can be analysed
    problems: list = field(default_factory=list)  # [(problem, how to fix)]
    image: np.ndarray = None                   # the (resized) RGB photo
    face_box: tuple = None                     # (x, y, width, height) in pixels
    face_crop: np.ndarray = None               # RGB crop of the face
    face_score: float = 0.0                    # detection confidence 0-1
    face_count: int = 0
    brightness: float = 0.0                    # 0 (black) - 255 (white)
    sharpness: float = 0.0                     # higher = sharper


def load_image(file) -> np.ndarray:
    """Opens an uploaded file / camera snapshot / path as an RGB numpy array.
    Applies the phone's rotation info (EXIF) and shrinks very large photos."""
    with Image.open(file) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        im.thumbnail((MAX_SIDE, MAX_SIDE))
        return np.array(im)


def detect_faces(rgb: np.ndarray):
    """Returns a list of (box, score) for every face found. box = (x, y, w, h) in pixels."""
    h, w = rgb.shape[:2]
    faces = []
    # model 0 = faces close to the camera (selfies); model 1 = faces further away.
    for model in (0, 1):
        with mp.solutions.face_detection.FaceDetection(model_selection=model, min_detection_confidence=0.5) as fd:
            result = fd.process(rgb)
        for det in result.detections or []:
            score = det.score[0]
            if score < MIN_FACE_SCORE:
                continue
            rel = det.location_data.relative_bounding_box
            x, y = max(0, int(rel.xmin * w)), max(0, int(rel.ymin * h))
            bw, bh = int(rel.width * w), int(rel.height * h)
            bw, bh = min(bw, w - x), min(bh, h - y)
            if bw > 0 and bh > 0:
                faces.append(((x, y, bw, bh), float(score)))
        if faces:
            break
    return sorted(faces, key=lambda f: f[1], reverse=True)


def measure_brightness(rgb: np.ndarray) -> float:
    """Average lightness (the L channel of LAB, scaled 0-255)."""
    return float(cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)[:, :, 0].mean())


def measure_sharpness(rgb: np.ndarray) -> float:
    """Variance of the Laplacian: a standard, simple blur measure.
    The face is resized to a fixed width first so the score doesn't depend on photo size."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    scale = 256 / gray.shape[1]
    gray = cv2.resize(gray, (256, max(1, int(gray.shape[0] * scale))), interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def check_photo(rgb: np.ndarray) -> PhotoCheck:
    """Finds the face and decides whether the photo is good enough to analyse."""
    result = PhotoCheck(ok=False, image=rgb)
    faces = detect_faces(rgb)
    result.face_count = len(faces)

    if not faces:
        result.brightness = measure_brightness(rgb)
        if result.brightness < TOO_DARK:
            result.problems.append(("The photo is too dark to find a face.",
                                    "Face a window or a lamp so light falls on your face."))
        result.problems.append(("No face was found.",
                                "Look straight at the camera with your whole face in the frame, and move closer "
                                "if you're far away. Remove sunglasses, masks or anything covering your face."))
        return result

    (x, y, w, h), score = faces[0]
    result.face_box, result.face_score = (x, y, w, h), score
    result.face_crop = rgb[y:y + h, x:x + w]
    result.brightness = measure_brightness(result.face_crop)
    result.sharpness = measure_sharpness(result.face_crop)

    if len(faces) > 1:
        # Only report this: size/light checks on "the first face" would just confuse.
        result.problems.append((f"{len(faces)} faces were found.",
                                "Take the photo with only one person in the frame."))
        return result

    if w < MIN_FACE_FRACTION * rgb.shape[1]:
        result.problems.append(("Your face is too small in the photo.",
                                "Move closer so your face fills most of the frame."))
    too_dark = result.brightness < TOO_DARK
    if too_dark:
        result.problems.append(("The photo is too dark.",
                                "Face a window in daylight or turn on a bright, white light in front of you."))
    elif result.brightness > TOO_BRIGHT:
        result.problems.append(("The photo is too bright (overexposed).",
                                "Avoid direct sun or flash; use soft, even daylight."))
    # In a dark photo the sharpness score is low anyway, so "blurry" would be a false alarm.
    if not too_dark and result.sharpness < MIN_SHARPNESS:
        result.problems.append(("The photo is blurry.",
                                "Hold the camera steady, tap to focus on your face, and clean the lens."))

    result.ok = not result.problems
    return result


def draw_face_box(rgb: np.ndarray, box, color=(104, 118, 75)) -> np.ndarray:
    """Returns a copy of the photo with a rectangle drawn around the face."""
    out = rgb.copy()
    if box:
        x, y, w, h = box
        thickness = max(2, rgb.shape[1] // 300)
        cv2.rectangle(out, (x, y), (x + w, y + h), color, thickness)
    return out
