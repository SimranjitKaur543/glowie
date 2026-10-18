"""
skin_tone.py — estimate skin tone and undertone from the cheeks.

Steps:
  1. MediaPipe Face Mesh finds 468 points on the face; we use the points around each cheek.
  2. Keep only skin-coloured pixels (drop shadows, shine, hair, background).
  3. Convert to LAB colour space: L = lightness, a = green↔red, b = blue↔yellow.
  4. K-means groups the pixels into 3 colours; the biggest group is "your skin colour".
  5. Tone   from the ITA angle  = atan((L − 50) / b)  — a published skin-colour measure
            (Chardon et al., 1991; used in dermatology research).
     Undertone from the hue angle = atan2(b, a): more yellow → warm, more pink/red → cool.
            The undertone thresholds are a simple rule of thumb, not a validated method.

Lighting changes colour a lot, so the result includes a confidence level and notes.
"""

from dataclasses import dataclass, field, replace
import math

import cv2
import mediapipe as mp
import numpy as np
from sklearn.cluster import KMeans

# Face Mesh point numbers around each cheek (from MediaPipe's face-mesh map).
LEFT_CHEEK = [117, 118, 119, 100, 142, 203, 206, 207, 187, 123, 116]
RIGHT_CHEEK = [346, 347, 348, 329, 371, 423, 426, 427, 411, 352, 345]

MIN_PIXELS = 150  # fewer usable skin pixels than this → can't measure reliably

# ITA° ranges (higher ITA = lighter skin). Our 4 app categories group the 6 research classes.
ITA_CLASSES = [(55, "very light"), (41, "light"), (28, "intermediate"), (10, "tan"), (-30, "brown"), (-999, "dark")]
TONE_FROM_ITA = [(41, "light"), (28, "medium"), (10, "tan"), (-999, "deep")]

# Hue angle (degrees): rule-of-thumb thresholds for undertone.
WARM_ABOVE = 58
COOL_BELOW = 48

# Everyday names shown to the user (the internal keys stay light / medium / tan / deep).
TONE_NAMES = {"light": "Fair", "medium": "Wheatish", "tan": "Dusky", "deep": "Deep"}
TONE_ORDER = ["light", "medium", "tan", "deep"]

# Photos that are very bright AND unusually smooth are often beauty-filtered or
# over-exposed by the phone, which makes skin look lighter than it is.
BRIGHT_L = 70          # cheek lightness (L*, 0-100) above this counts as "very bright"
SMOOTH_STD = 3.0       # spread of cheek lightness below this counts as "unusually smooth"

FOUNDATION_GUIDE = {
    "light": "fair to light shades",
    "medium": "light-medium to medium shades",
    "tan": "medium-tan to tan shades",
    "deep": "deep to rich shades",
}
UNDERTONE_GUIDE = {
    "warm": "warm / golden / yellow (often marked W, Y or G)",
    "cool": "cool / pink / rosy (often marked C, P or R)",
    "neutral": "neutral (often marked N) — or try one warm and one cool shade",
}


@dataclass
class SkinToneResult:
    ok: bool
    message: str = ""
    lab: tuple = None             # (L, a, b) of the dominant skin colour
    hex_color: str = ""           # the same colour as #RRGGBB, for a colour swatch
    ita: float = None             # Individual Typology Angle, degrees
    ita_class: str = ""           # research class, e.g. "intermediate"
    hue: float = None             # hue angle, degrees
    tone: str = ""                # light / medium / tan / deep (shown as Fair / Wheatish / Dusky / Deep)
    measured_tone: str = ""       # what the photo gave, kept if the user corrects the tone
    corrected_by_user: bool = False
    undertone: str = ""           # warm / cool / neutral
    undertone_source: str = "photo"   # "photo" or "your answers"
    photo_undertone: str = ""     # what the photo gave, kept if answers override it
    region_hues: dict = field(default_factory=dict)  # hue angle per measured area
    white_balanced: bool = False  # True if the light colour was corrected using the eye whites
    confidence: str = ""          # high / medium / low
    notes: list = field(default_factory=list)
    foundation: dict = field(default_factory=dict)
    regions: list = field(default_factory=list)  # cheek polygons (for drawing)


# ---------------------------------------------------------------- small, testable pieces

def ita_angle(L: float, b: float) -> float:
    """ITA° = atan((L − 50) / b) × 180/π. Uses atan2 so b close to 0 doesn't crash."""
    return math.degrees(math.atan2(L - 50, b))


def hue_angle(a: float, b: float) -> float:
    return math.degrees(math.atan2(b, a))


def classify_ita(ita: float):
    """Returns (research class, app tone) for an ITA value."""
    research = next(name for limit, name in ITA_CLASSES if ita > limit)
    tone = next(name for limit, name in TONE_FROM_ITA if ita > limit)
    return research, tone


def classify_undertone(hue: float) -> str:
    if hue > WARM_ABOVE:
        return "warm"
    if hue < COOL_BELOW:
        return "cool"
    return "neutral"


def rgb_to_lab(pixels_rgb: np.ndarray) -> np.ndarray:
    """N×3 uint8 RGB → N×3 float LAB in standard units (L 0-100, a/b roughly −128…127)."""
    as_image = (pixels_rgb.reshape(-1, 1, 3).astype(np.float32) / 255.0)
    return cv2.cvtColor(as_image, cv2.COLOR_RGB2LAB).reshape(-1, 3)


def lab_to_hex(lab) -> str:
    pixel = np.array(lab, dtype=np.float32).reshape(1, 1, 3)
    rgb = np.clip(cv2.cvtColor(pixel, cv2.COLOR_LAB2RGB)[0, 0] * 255, 0, 255).astype(int)
    return "#{:02X}{:02X}{:02X}".format(*rgb)


def skin_pixel_filter(pixels_rgb: np.ndarray) -> np.ndarray:
    """Keeps pixels that look like skin (YCrCb colour range) and drops the darkest/brightest
    5% (shadows and shine)."""
    if len(pixels_rgb) == 0:
        return pixels_rgb
    ycrcb = cv2.cvtColor(pixels_rgb.reshape(-1, 1, 3), cv2.COLOR_RGB2YCrCb).reshape(-1, 3)
    y, cr, cb = ycrcb[:, 0], ycrcb[:, 1], ycrcb[:, 2]
    skin = (cr >= 133) & (cr <= 180) & (cb >= 77) & (cb <= 135)
    if skin.sum() == 0:
        return pixels_rgb[skin]
    low, high = np.percentile(y[skin], [5, 95])
    return pixels_rgb[skin & (y >= low) & (y <= high)]


SAME_COLOUR_DELTA_E = 12  # pixels closer than this to the main colour count as "the same skin colour"


def dominant_lab(pixels_rgb: np.ndarray, k: int = 3):
    """K-means on LAB pixels. Returns (LAB of the biggest cluster, consistency 0-1).

    Consistency = share of ALL pixels whose colour is close (ΔE < 12) to the main colour.
    (We don't use the cluster's own size: K-means always makes k groups, so even, uniform
    skin gets split into similar-looking groups and would wrongly look "uneven".)
    """
    lab = rgb_to_lab(pixels_rgb)
    k = min(k, len(lab))
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(lab)
    counts = np.bincount(km.labels_, minlength=k)
    centre = km.cluster_centers_[int(np.argmax(counts))]
    delta_e = np.linalg.norm(lab - centre, axis=1)  # CIE76 colour difference
    return tuple(float(v) for v in centre), float((delta_e < SAME_COLOUR_DELTA_E).mean())


# ---------------------------------------------------------------- face regions

def face_points(rgb: np.ndarray):
    """Face Mesh landmarks as an N×2 pixel array (478 points incl. irises), or None."""
    h, w = rgb.shape[:2]
    with mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1,
                                         refine_landmarks=True, min_detection_confidence=0.5) as mesh:
        result = mesh.process(rgb)
    if not result.multi_face_landmarks:
        return None
    return np.array([[p.x * w, p.y * h] for p in result.multi_face_landmarks[0].landmark], dtype=np.float32)


def _polygon(pts, indices, shrink=0.85):
    """Convex polygon through the given landmarks, shrunk towards its centre to avoid edges."""
    poly = pts[indices]
    centre = poly.mean(axis=0)
    return cv2.convexHull((centre + (poly - centre) * shrink).astype(np.int32))


def _box(x1, y1, x2, y2):
    return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32)


def face_regions(pts) -> dict:
    """Skin areas to measure: both cheeks, the forehead (just above the brows) and the chin."""
    d = float(np.linalg.norm(pts[263] - pts[33]))            # distance between outer eye corners
    brow_y = min(pts[66][1], pts[296][1])
    lip_y, chin_y, mid_x = pts[17][1], pts[152][1], pts[152][0]
    return {
        "left cheek": _polygon(pts, LEFT_CHEEK),
        "right cheek": _polygon(pts, RIGHT_CHEEK),
        "forehead": _box(pts[33][0] + 0.3 * d, brow_y - 0.26 * d, pts[263][0] - 0.3 * d, brow_y - 0.05 * d),
        "chin": _box(mid_x - 0.15 * d, lip_y + 0.3 * (chin_y - lip_y), mid_x + 0.15 * d, lip_y + 0.75 * (chin_y - lip_y)),
    }


def pixels_inside(rgb: np.ndarray, polygon) -> np.ndarray:
    mask = np.zeros(rgb.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, polygon, 1)
    return rgb[mask.astype(bool)]


# ---------------------------------------------------------------- white balance from the eye whites

LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE = [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466]
IRIS_CENTRE = {"left": 468, "right": 473}       # only present with refine_landmarks=True
IRIS_EDGE = {"left": 469, "right": 474}
MIN_EYE_WHITE_PIXELS = 15
# Normal eye-white colour in neutral light (a, b in LAB). Real eye whites are slightly
# warm, not pure white. Measured ≈ (7, 12) on a studio portrait; results barely change
# between (5, 10) and (6, 12) — see README "Undertone".
EYE_WHITE_REFERENCE = (5.5, 11.0)
MAX_OFFSET = 25.0                               # ignore absurd corrections (failed eye detection)


def eye_white_offset(rgb: np.ndarray, pts):
    """How much the light/camera tints colours, measured on the whites of the eyes.

    A light that makes skin look yellower/pinker/bluer does the same to the eye whites, so
    (eye-white colour − normal eye-white colour) is the tint to remove from the skin.
    Returns ((delta_a, delta_b) or None if the eye whites can't be measured, pixels used).
    """
    if pts is None or len(pts) < 478:
        return None, 0
    samples = []
    for side, eye in (("left", LEFT_EYE), ("right", RIGHT_EYE)):
        mask = np.zeros(rgb.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [pts[eye].astype(np.int32)], 1)
        radius = np.linalg.norm(pts[IRIS_EDGE[side]] - pts[IRIS_CENTRE[side]]) * 1.25
        cv2.circle(mask, tuple(int(v) for v in pts[IRIS_CENTRE[side]]), int(radius) + 1, 0, -1)
        samples.append(rgb[mask.astype(bool)])
    px = np.concatenate(samples)
    if len(px) < MIN_EYE_WHITE_PIXELS * 2:
        return None, len(px)
    # keep the brightest half: real eye white, not lashes, shadow or the eyelid line
    brightness = px.astype(np.float32).mean(axis=1)
    px = px[brightness >= np.median(brightness)]
    lab = rgb_to_lab(px)
    if len(px) < MIN_EYE_WHITE_PIXELS or np.median(lab[:, 0]) < 35 or (lab[:, 0] > 98).mean() > 0.5:
        return None, len(px)          # too dark, or blown out to pure white (no colour information)
    offset = (float(np.median(lab[:, 1])) - EYE_WHITE_REFERENCE[0],
              float(np.median(lab[:, 2])) - EYE_WHITE_REFERENCE[1])
    if max(abs(offset[0]), abs(offset[1])) > MAX_OFFSET:
        return None, len(px)
    return offset, len(px)


def region_hue(pixels_rgb: np.ndarray, offset=(0.0, 0.0)):
    """Hue angle of an area from the MEDIAN a and b of its skin pixels (robust to spots/shine),
    after removing the light tint `offset`."""
    if len(pixels_rgb) < 40:
        return None
    lab = rgb_to_lab(pixels_rgb)
    return hue_angle(float(np.median(lab[:, 1])) - offset[0], float(np.median(lab[:, 2])) - offset[1])


# ---------------------------------------------------------------- main function

def analyze_skin_tone(rgb: np.ndarray) -> SkinToneResult:
    pts = face_points(rgb)
    if pts is None:
        return SkinToneResult(ok=False, message="Couldn't find the cheek areas. Try a straight-on photo with your whole face visible.")

    # 1. Measure the light tint on the eye whites (if they can be measured).
    offset, _ = eye_white_offset(rgb, pts)

    # 2. Skin pixels from each area.
    regions = face_regions(pts)
    skin = {name: skin_pixel_filter(pixels_inside(rgb, poly)) for name, poly in regions.items()}
    per_cheek = [skin["left cheek"], skin["right cheek"]]
    pixels = np.concatenate(per_cheek)
    if len(pixels) < MIN_PIXELS:
        return SkinToneResult(ok=False, regions=list(regions.values()),
                              message="Not enough clear skin on the cheeks (hair, shadows or makeup may cover them). "
                                      "Try a closer photo in even light.")
    # 3. Undertone: hue of every area with the light tint removed.
    hues = {name: region_hue(px, offset or (0.0, 0.0)) for name, px in skin.items()}
    result = describe(pixels, per_cheek, list(regions.values()),
                      region_hues={k: round(v, 1) for k, v in hues.items() if v is not None},
                      tint=offset)
    result.white_balanced = offset is not None
    if offset is None:
        result.notes.insert(0, "The whites of the eyes couldn't be measured (closed eyes, glasses or shadow), so "
                               "the light colour wasn't corrected — undertone is less reliable.")
        result.confidence = {"high": "medium", "medium": "low"}.get(result.confidence, result.confidence)
    return result


def describe(pixels, per_cheek, regions=(), region_hues=None, tint=None) -> SkinToneResult:
    """Turns skin pixels into tone, undertone, confidence and a foundation guide.

    Tone comes from the main cheek colour. Undertone uses the MEDIAN hue of all measured areas
    (cheeks, forehead, chin) when available — cheeks alone are often flushed/red, which would
    wrongly push everyone towards "cool". `tint` = light tint (delta a, delta b) to remove.
    """
    (L, a, b), share = dominant_lab(pixels)
    if tint is not None:                 # remove the light's colour tint from the skin colour too
        a, b = a - tint[0], b - tint[1]
    ita = ita_angle(L, b)
    usable_hues = [h for h in (region_hues or {}).values() if h is not None]
    hue = float(np.median(usable_hues)) if len(usable_hues) >= 2 else hue_angle(a, b)
    ita_class, tone = classify_ita(ita)
    undertone = classify_undertone(hue)

    # Confidence: start high, lower it for anything that makes the colour less reliable.
    notes, penalty = [], 0
    if len(pixels) < 1000:
        penalty += 1
        notes.append("Only a small area of cheek skin was usable.")
    if share < 0.45:
        penalty += 1
        notes.append("Cheek colour was very uneven (shadows, redness or makeup).")
    usable = [c for c in per_cheek if len(c) >= 50]
    if len(usable) == 2:
        left, right = (rgb_to_lab(c)[:, 0].mean() for c in usable)
        if abs(left - right) > 8:
            penalty += 1
            notes.append("One cheek is much brighter than the other — the light is coming from one side.")
    if abs(hue - (WARM_ABOVE + COOL_BELOW) / 2) < 3:
        notes.append("Undertone is right on the border between categories.")
    lightness_spread = float(rgb_to_lab(pixels)[:, 0].std())
    if L > BRIGHT_L and lightness_spread < SMOOTH_STD:
        penalty += 2
        notes.append("Your skin looks very bright and unusually smooth — beauty mode, a filter or the camera's "
                     "auto-brightening may be on. This makes skin look LIGHTER than it really is. "
                     "Turn beauty mode off and retake, or choose your real tone below.")
    elif L > BRIGHT_L:
        penalty += 1
        notes.append("The photo is very bright, so your skin may look lighter than it really is.")
    notes.append("Camera colour, lighting and filters change how skin looks — treat this as a starting point.")
    confidence = ["high", "medium", "low", "low"][min(penalty, 3)]

    return SkinToneResult(ok=True, region_hues=dict(region_hues or {}),
                          lab=(round(L, 1), round(a, 1), round(b, 1)), hex_color=lab_to_hex((L, a, b)),
                          ita=round(ita, 1), ita_class=ita_class, hue=round(hue, 1), tone=tone,
                          undertone=undertone, confidence=confidence, notes=notes,
                          foundation=foundation_guide(tone, undertone), regions=list(regions))


def foundation_guide(tone: str, undertone: str) -> dict:
    """Shade range text for a tone + undertone (also used when the user corrects the tone)."""
    return {
        "depth": FOUNDATION_GUIDE[tone],
        "undertone": UNDERTONE_GUIDE[undertone],
        "summary": f"Start with {FOUNDATION_GUIDE[tone]} in a {undertone} undertone.",
        "tip": "Swatch 2–3 neighbouring shades on your jawline and check them in daylight; "
               "the one that disappears into your skin is your match.",
    }


def with_user_tone(result: SkinToneResult, tone: str) -> SkinToneResult:
    """Returns a copy of the result using the tone the user says is right.
    The measured values (colour, ITA) are kept so the report stays honest."""
    if tone == result.tone or tone not in TONE_NAMES:
        return result
    return replace(result, tone=tone, measured_tone=result.measured_tone or result.tone,
                   corrected_by_user=True, foundation=foundation_guide(tone, result.undertone))


# The 3 classic undertone checks used by makeup artists. Each answer votes warm (+1),
# cool (−1) or neutral (0).
UNDERTONE_QUESTIONS = {
    "veins": ("Look at the veins on the inside of your wrist in daylight. They look mostly…",
              {"green": ("Green / olive", 1), "blue": ("Blue / purple", -1), "both": ("A mix / can't tell", 0)}),
    "jewellery": ("Which jewellery looks better on your skin?",
                  {"gold": ("Gold", 1), "silver": ("Silver", -1), "both": ("Both look good", 0)}),
    "sun": ("In the sun, your skin usually…",
            {"tans": ("Tans easily, rarely burns", 1), "burns": ("Burns or turns pink easily", -1),
             "both": ("Burns a little, then tans", 0)}),
}


def undertone_from_answers(answers: dict):
    """Undertone from the 3 questions (needs at least 2 answers), or None."""
    votes = [UNDERTONE_QUESTIONS[q][1][a][1] for q, a in answers.items()
             if q in UNDERTONE_QUESTIONS and a in UNDERTONE_QUESTIONS[q][1]]
    if len(votes) < 2:
        return None
    total = sum(votes)
    return "warm" if total >= 2 else "cool" if total <= -2 else "neutral"


def with_user_undertone(result: SkinToneResult, undertone: str, source: str = "your answers") -> SkinToneResult:
    """Copy of the result using an undertone from the questions (keeps the photo's guess)."""
    if not undertone or undertone == result.undertone:
        return replace(result, photo_undertone=result.photo_undertone or result.undertone)
    return replace(result, undertone=undertone, undertone_source=source,
                   photo_undertone=result.photo_undertone or result.undertone,
                   foundation=foundation_guide(result.tone, undertone))


def tone_name(tone: str) -> str:
    """Everyday name for a tone key, e.g. "tan" → "Dusky"."""
    return TONE_NAMES.get(tone, tone.title())


def draw_regions(rgb: np.ndarray, regions, color=(195, 164, 119)) -> np.ndarray:
    """Returns a copy of the photo with the measured cheek areas outlined."""
    out = rgb.copy()
    thickness = max(2, rgb.shape[1] // 400)
    for poly in regions or []:
        cv2.polylines(out, [poly], isClosed=True, color=color, thickness=thickness)
    return out
