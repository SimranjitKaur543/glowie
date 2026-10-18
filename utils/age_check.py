"""
age_check.py — a rough check that the face in the photo fits the chosen age group.

No trained model and no downloads: it measures fine lines (wrinkles) in the areas where
they usually appear first, and warns if there are clearly more than is usual for a young
age group. It cannot tell exact age — it only catches big mismatches, e.g. an elderly
face marked "18–24".

How the "line score" works:
  1. Face Mesh finds the forehead, the skin under each eye and the outer eye corners.
  2. Every area is scaled so the distance between the eyes is the same in every photo.
  3. A Hessian "ridge" filter finds thin DARK LINES (creases). Round spots such as acne or
     pores respond in both directions, so they are cancelled out.
  4. The line strength is divided by the area's brightness (so lighting matters less) and
     turned into a 0–100 score. The middle value of the areas is used, so hair or glasses
     in one area don't spoil the result.
"""

from dataclasses import dataclass, field

import cv2
import mediapipe as mp
import numpy as np

EYE_DISTANCE_PX = 160        # every face is rescaled so the outer eye corners are this far apart
RIDGE_SIGMA = 1.6            # blur before the line filter (ignores tiny noise and pores)
SCORE_SCALE = 100            # turns the raw line strength into a readable 0–100 score

# Score bands, set from test images (an early-40s portrait scored ~10–15; drawn-on light / deep
# wrinkles scored ~50 / ~80). Not yet checked on many real faces — see README "Age check".
SOME_LINES = 25              # at or above: some visible lines
MANY_LINES = 40              # at or above: many / deep lines

# Face Mesh landmark numbers
OUTER_EYE = (33, 263)        # outer eye corners (image left, image right)
LOWER_LID = (145, 374)       # middle of the lower eyelids
BROW_TOP = (66, 296)         # upper edge of the eyebrows
FOREHEAD_TOP = 10            # top-centre of the face mesh
FACE_EDGE = (234, 454)       # left and right edge of the face

LOW_RES_EYE_DISTANCE = 90   # eyes closer than this (in the original photo) = low detail, less reliable

# Which line level is "clearly too many" for each young age group.
WARN_AT = {"under_18": ("some", "many"), "18_24": ("many",)}


@dataclass
class LineCheck:
    ok: bool
    score: float = 0.0              # 0–100, higher = more visible lines
    level: str = ""                 # "few" | "some" | "many"
    area_scores: dict = field(default_factory=dict)
    message: str = ""


def _landmarks(rgb):
    with mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1,
                                         refine_landmarks=False, min_detection_confidence=0.5) as mesh:
        result = mesh.process(rgb)
    if not result.multi_face_landmarks:
        return None
    h, w = rgb.shape[:2]
    return np.array([[p.x * w, p.y * h] for p in result.multi_face_landmarks[0].landmark])


def line_strength(gray: np.ndarray) -> float:
    """Average strength of thin dark lines in a grayscale patch (already scaled).

    Uses the Hessian (second derivatives). At a dark line the curvature across the line is
    large and positive while the curvature along it is about zero; at a round spot both are
    large, so `max(0, big) − |small|` keeps lines and cancels spots.
    """
    g = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), RIDGE_SIGMA)
    dxx = cv2.Sobel(g, cv2.CV_32F, 2, 0, ksize=5)
    dyy = cv2.Sobel(g, cv2.CV_32F, 0, 2, ksize=5)
    dxy = cv2.Sobel(g, cv2.CV_32F, 1, 1, ksize=5)
    root = np.sqrt((dxx - dyy) ** 2 + 4 * dxy ** 2)
    big, small = (dxx + dyy + root) / 2, (dxx + dyy - root) / 2
    ridge = np.clip(big, 0, None) - np.abs(small)
    ridge = np.clip(ridge, 0, None)
    return float(ridge.mean() / (g.mean() + 1.0))


def _boxes(pts):
    """Areas to measure, as (name, x1, y1, x2, y2) in the scaled image."""
    (lx, ly), (rx, ry) = pts[OUTER_EYE[0]], pts[OUTER_EYE[1]]
    d = abs(rx - lx)
    brow_y = min(pts[BROW_TOP[0]][1], pts[BROW_TOP[1]][1])
    top_y = pts[FOREHEAD_TOP][1]
    # Forehead: a band just above the eyebrows, in the middle — a fixed distance (not the mesh top,
    # which sits under fringes/bangs), so hair is less likely to be included.
    boxes = [("forehead", lx + 0.3 * d, max(top_y, brow_y - 0.26 * d), rx - 0.3 * d, brow_y - 0.05 * d)]
    for side, (cx, cy), lid in (("left", (lx, ly), LOWER_LID[0]), ("right", (rx, ry), LOWER_LID[1])):
        lid_x, lid_y = pts[lid]
        # Under the eye: start below the eyelashes.
        boxes.append((f"under_eye_{side}", lid_x - 0.11 * d, lid_y + 0.08 * d, lid_x + 0.11 * d, lid_y + 0.19 * d))
        # Outer eye corner ("crow's feet"): start a little outside the eye, stop before the hair/ear.
        edge_x = pts[FACE_EDGE[0] if side == "left" else FACE_EDGE[1]][0]
        out = np.sign(edge_x - cx)                                   # direction away from the nose
        end = cx + out * min(0.22 * d, 0.8 * abs(edge_x - cx))       # stop before the face edge
        x1, x2 = sorted((cx + out * 0.05 * d, end))
        boxes.append((f"eye_corner_{side}", x1, cy - 0.08 * d, x2, cy + 0.12 * d))
    return boxes


def measure_lines(rgb: np.ndarray) -> LineCheck:
    pts = _landmarks(rgb)
    if pts is None:
        return LineCheck(ok=False, message="Face Mesh couldn't find the face for the line check.")
    eye_dist = np.linalg.norm(pts[OUTER_EYE[1]] - pts[OUTER_EYE[0]])
    if eye_dist < 30:
        return LineCheck(ok=False, message="The face is too small in the photo for the line check.")

    scale = EYE_DISTANCE_PX / eye_dist
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, None, fx=scale, fy=scale,
                      interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    pts = pts * scale
    h, w = gray.shape

    scores = {}
    for name, x1, y1, x2, y2 in _boxes(pts):
        x1, y1, x2, y2 = (int(round(v)) for v in (x1, y1, x2, y2))
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        if x2 - x1 < 12 or y2 - y1 < 8:
            continue
        scores[name] = round(min(100.0, line_strength(gray[y1:y2, x1:x2]) * SCORE_SCALE), 1)
    if len(scores) < 3:
        return LineCheck(ok=False, area_scores=scores, message="Not enough of the face was visible for the line check.")

    score = float(np.median(list(scores.values())))
    level = "many" if score >= MANY_LINES else "some" if score >= SOME_LINES else "few"
    message = ""
    if eye_dist < LOW_RES_EYE_DISTANCE:
        message = "The face is small in this photo, so fine lines may be missed — a closer photo gives a better check."
    return LineCheck(ok=True, score=round(score, 1), level=level, area_scores=scores, message=message)


def age_mismatch_warning(lines: LineCheck, age_group: str, age_label: str = ""):
    """Returns a warning message if the face clearly looks older than the chosen age group, else None."""
    if not lines or not lines.ok or lines.level not in WARN_AT.get(age_group, ()):
        return None
    amount = "many fine lines and wrinkles" if lines.level == "many" else "some fine lines"
    return (f"The photo shows {amount} (line score {lines.score:.0f}/100), which is unusual for the age "
            f"group you chose ({age_label or age_group}). Please check that the age group is right — the "
            "routine depends on it. If it is right, you can ignore this: smiling, squinting and side "
            "lighting can create lines too.")
