"""
Tests for the Glowie modules. Run from the project folder:

    python -m unittest discover tests -v

They use made-up images and colours, so no real photos are needed. (A real-face test
has to be done by hand: run the app and try your own selfie.)
"""

import math
import unittest

import cv2
import numpy as np

from utils import age_check, face_detect, recommend, report, skin_tone


def lab_pixels(L, a, b, n=3000, noise=1.5, seed=0):
    """Makes n RGB pixels whose LAB colour is about (L, a, b)."""
    rng = np.random.default_rng(seed)
    lab = np.array([L, a, b], dtype=np.float32) + rng.normal(0, noise, (n, 3)).astype(np.float32)
    rgb = cv2.cvtColor(lab.reshape(-1, 1, 3), cv2.COLOR_LAB2RGB).reshape(-1, 3)
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)


class TestRules(unittest.TestCase):
    def setUp(self):
        self.rules = recommend.load_rules()

    def test_json_is_complete(self):
        order = self.rules["step_order"]
        for name, st in self.rules["skin_types"].items():
            for part in ("label", "morning", "night", "ingredients", "avoid"):
                self.assertIn(part, st, f"{name} is missing {part}")
            for step in st["morning"] + st["night"]:
                self.assertIn(step["step"], order, f"{name}: unknown step {step['step']}")
        for name, c in self.rules["concerns"].items():
            for step in c["morning"] + c["night"]:
                self.assertIn(step["step"], order, f"{name}: unknown step {step['step']}")
        for tone in ("light", "medium", "tan", "deep"):
            self.assertIn(tone, self.rules["tone_tips"])

    def test_every_routine_has_sunscreen_last_in_the_morning(self):
        for skin_type in self.rules["skin_types"]:
            rec = recommend.build_routine({"skin_type": skin_type, "concerns": list(self.rules["concerns"])})
            self.assertEqual(rec.morning[-1]["step"], "sunscreen", skin_type)
            self.assertEqual(rec.morning[0]["step"], "cleanser", skin_type)

    def test_oily_acne(self):
        rec = recommend.build_routine({"skin_type": "oily", "concerns": ["acne"], "acne_level": "few"})
        self.assertIn("Salicylic acid (BHA)", rec.use)
        self.assertTrue(any("Salicylic" in s["product"] for s in rec.night))
        self.assertTrue(any("oils" in a.lower() for a in rec.avoid))
        self.assertEqual(rec.see_doctor, [])

    def test_pregnancy_removes_retinol(self):
        rec = recommend.build_routine({"skin_type": "normal", "concerns": ["wrinkles"], "pregnant": True})
        text = " ".join(s["product"] for s in rec.morning + rec.night).lower()
        self.assertNotIn("retinol", text)
        self.assertNotIn("Retinol", rec.use)
        self.assertTrue(any("pregnan" in t.lower() for t in rec.tips))
        self.assertTrue(any("Peptide" in s["product"] for s in rec.night))

    def test_cystic_acne_and_changing_spot_mean_see_a_doctor(self):
        rec = recommend.build_routine({"skin_type": "oily", "concerns": ["acne"], "acne_level": "cystic",
                                       "changing_spot": True})
        self.assertEqual(len(rec.see_doctor), 2)

    def test_sensitive_adds_patch_test(self):
        rec = recommend.build_routine({"skin_type": "dry", "concerns": [], "sensitive": True})
        self.assertTrue(any("patch test" in t.lower() for t in rec.tips))
        self.assertIn("Fragrance and essential oils", rec.avoid)

    def test_strong_actives_are_not_stacked(self):
        rec = recommend.build_routine({"skin_type": "normal", "concerns": ["acne", "wrinkles", "dullness"]})
        self.assertTrue(any("different nights" in t for t in rec.tips))

    def test_tone_tips_and_no_duplicates(self):
        rec = recommend.build_routine({"skin_type": "combination", "concerns": ["dark_spots", "dullness"]}, tone="deep")
        self.assertTrue(any("white cast" in t for t in rec.tips))
        self.assertEqual(len(rec.use), len(set(rec.use)))
        steps = [(s["step"], s["product"]) for s in rec.morning]
        self.assertEqual(len(steps), len(set(steps)), "Vitamin C must not appear twice")

    def test_unknown_concerns_are_ignored_and_bad_type_rejected(self):
        rec = recommend.build_routine({"skin_type": "normal", "concerns": ["not-a-concern"]})
        self.assertEqual(rec.concerns, [])
        with self.assertRaises(ValueError):
            recommend.build_routine({"skin_type": "purple"})

    def test_age_groups(self):
        def night_text(**profile):
            rec = recommend.build_routine({"skin_type": "normal", **profile})
            return rec, " ".join(s["product"] for s in rec.night).lower()

        rec, night = night_text(concerns=["wrinkles"], age_group="under_18")
        self.assertNotIn("retinol", night)                       # no retinol for teens, even for wrinkles
        self.assertNotIn("Retinol", rec.use)
        self.assertEqual(rec.age_group_label, "Under 18")
        self.assertTrue(any("Keep it simple" in t for t in rec.tips))

        rec, night = night_text(concerns=[], age_group="35_44")
        self.assertIn("retinol", night)                          # 35+ introduces retinol
        rec, night = night_text(concerns=["wrinkles"], age_group="45_plus")
        self.assertEqual(night.count("retinol serum"), 1)        # not added twice
        rec, night = night_text(concerns=[], age_group="45_plus", pregnant=True)
        self.assertNotIn("retinol", night)                       # pregnancy still wins

        rec, _ = night_text(concerns=[], age_group=None)
        self.assertEqual(rec.age_group_label, "")
        rec, _ = night_text(concerns=[], age_group="not-an-age")
        self.assertEqual(rec.age_group_label, "")

    def test_skin_type_quiz(self):
        self.assertEqual(recommend.guess_skin_type("comfortable", "shiny_tzone"), "combination")
        self.assertEqual(recommend.guess_skin_type("oily", "fine"), "oily")
        self.assertEqual(recommend.guess_skin_type("tight", "flaky"), "dry")
        self.assertEqual(recommend.guess_skin_type("comfortable", "fine"), "normal")


class TestSkinTone(unittest.TestCase):
    def test_ita_formula_and_classes(self):
        self.assertAlmostEqual(skin_tone.ita_angle(65, 18), math.degrees(math.atan(15 / 18)), places=6)
        self.assertEqual(skin_tone.classify_ita(60), ("very light", "light"))
        self.assertEqual(skin_tone.classify_ita(45), ("light", "light"))
        self.assertEqual(skin_tone.classify_ita(35), ("intermediate", "medium"))
        self.assertEqual(skin_tone.classify_ita(20), ("tan", "tan"))
        self.assertEqual(skin_tone.classify_ita(-10), ("brown", "deep"))
        self.assertEqual(skin_tone.classify_ita(-50), ("dark", "deep"))

    def test_undertone_from_hue(self):
        self.assertEqual(skin_tone.classify_undertone(65), "warm")
        self.assertEqual(skin_tone.classify_undertone(40), "cool")
        self.assertEqual(skin_tone.classify_undertone(53), "neutral")

    def test_describe_known_colours(self):
        cases = [((72, 10, 20), "light", "warm"),     # ITA 56 → light; hue 63 → warm
                 ((60, 14, 13), "medium", "cool"),    # ITA 37.6 → medium; hue 43 → cool
                 ((52, 13, 17), "tan", "neutral"),    # ITA 6.7 → deep? check below
                 ((35, 12, 16), "deep", "neutral")]   # ITA −43 → deep; hue 53 → neutral
        for (L, a, b), tone, undertone in cases:
            expected_tone = skin_tone.classify_ita(skin_tone.ita_angle(L, b))[1]
            px = lab_pixels(L, a, b)
            result = skin_tone.describe(px, [px[:1500], px[1500:]])
            self.assertTrue(result.ok)
            self.assertEqual(result.tone, expected_tone, (L, a, b))
            self.assertEqual(result.undertone, undertone, (L, a, b))
            self.assertAlmostEqual(result.lab[0], L, delta=1.5)
            self.assertRegex(result.hex_color, r"^#[0-9A-F]{6}$")

    def test_kmeans_ignores_a_minority_colour(self):
        skin = lab_pixels(65, 12, 18, n=2400)
        shadow = lab_pixels(30, 5, 8, n=600, seed=1)
        lab, share = skin_tone.dominant_lab(np.concatenate([skin, shadow]))
        self.assertAlmostEqual(lab[0], 65, delta=2)
        self.assertGreater(share, 0.6)

    def test_even_skin_is_high_confidence(self):
        px = lab_pixels(65, 12, 18, n=3000, noise=3)
        result = skin_tone.describe(px, [px[:1500], px[1500:]])
        self.assertEqual(result.confidence, "high", result.notes)

    def test_uneven_lighting_lowers_confidence(self):
        bright, dark = lab_pixels(75, 12, 18, n=1500), lab_pixels(55, 12, 18, n=1500, seed=2)
        result = skin_tone.describe(np.concatenate([bright, dark]), [bright, dark])
        self.assertNotEqual(result.confidence, "high")
        self.assertTrue(any("brighter" in n for n in result.notes))

    def test_user_can_correct_the_tone(self):
        px = lab_pixels(72, 10, 20)                      # measures as light ("Fair")
        measured = skin_tone.describe(px, [px[:1500], px[1500:]])
        self.assertEqual(skin_tone.tone_name(measured.tone), "Fair")
        fixed = skin_tone.with_user_tone(measured, "tan")
        self.assertEqual(skin_tone.tone_name(fixed.tone), "Dusky")
        self.assertTrue(fixed.corrected_by_user)
        self.assertEqual(fixed.measured_tone, "light")
        self.assertIn("tan", fixed.foundation["summary"])
        self.assertEqual(fixed.ita, measured.ita)        # measured numbers are kept
        self.assertIs(skin_tone.with_user_tone(measured, "light"), measured)  # no change → same result
        rec = recommend.build_routine({"skin_type": "normal"}, fixed.tone)
        self.assertTrue(any("dark marks after breakouts" in t for t in rec.tips))  # Dusky tips used

    def test_undertone_questions(self):
        f = skin_tone.undertone_from_answers
        self.assertEqual(f({"veins": "green", "jewellery": "gold", "sun": "tans"}), "warm")
        self.assertEqual(f({"veins": "blue", "jewellery": "silver"}), "cool")
        self.assertEqual(f({"veins": "green", "jewellery": "silver", "sun": "both"}), "neutral")
        self.assertEqual(f({"veins": "both", "jewellery": "both"}), "neutral")
        self.assertIsNone(f({"veins": "green"}))                    # 1 answer is not enough
        self.assertIsNone(f({}))
        self.assertIsNone(f({"veins": "purple-ish", "colour": "x"}))  # unknown answers ignored

    def test_answers_override_photo_undertone(self):
        px = lab_pixels(60, 14, 13)                                   # photo says cool
        photo = skin_tone.describe(px, [px[:1500], px[1500:]])
        self.assertEqual(photo.undertone, "cool")
        result = skin_tone.with_user_undertone(photo, "warm")
        self.assertEqual(result.undertone, "warm")
        self.assertEqual(result.photo_undertone, "cool")
        self.assertEqual(result.undertone_source, "your answers")
        self.assertIn("warm", result.foundation["summary"])
        same = skin_tone.with_user_undertone(photo, None)              # no answers → photo is used
        self.assertEqual(same.undertone, "cool")
        self.assertEqual(same.undertone_source, "photo")

    def test_light_tint_is_removed_from_skin_colour(self):
        px = lab_pixels(65, 12, 18)                                   # hue ≈ 56°
        tinted = skin_tone.describe(px, [px[:1500], px[1500:]], tint=(8.0, -6.0))
        self.assertAlmostEqual(tinted.lab[1], 4.0, delta=1.5)          # a − 8
        self.assertAlmostEqual(tinted.lab[2], 24.0, delta=1.5)         # b + 6

    def test_bright_smooth_photo_warns_about_filters(self):
        filtered = lab_pixels(78, 9, 16, noise=0.8)      # very bright, almost no texture
        result = skin_tone.describe(filtered, [filtered[:1500], filtered[1500:]])
        self.assertTrue(any("beauty mode" in n for n in result.notes))
        self.assertEqual(result.confidence, "low")
        normal = lab_pixels(62, 12, 18, noise=4)
        result = skin_tone.describe(normal, [normal[:1500], normal[1500:]])
        self.assertFalse(any("beauty mode" in n or "very bright" in n for n in result.notes))

    def test_skin_filter_drops_non_skin(self):
        blue = np.tile(np.array([[40, 80, 200]], dtype=np.uint8), (500, 1))
        self.assertEqual(len(skin_tone.skin_pixel_filter(blue)), 0)
        skin = lab_pixels(65, 12, 18, n=500)
        self.assertGreater(len(skin_tone.skin_pixel_filter(skin)), 300)

    def test_no_face_returns_message(self):
        result = skin_tone.analyze_skin_tone(np.full((300, 300, 3), 200, np.uint8))
        self.assertFalse(result.ok)
        self.assertIn("cheek", result.message)


class TestFaceDetect(unittest.TestCase):
    def test_blank_photo_has_no_face(self):
        check = face_detect.check_photo(np.full((480, 640, 3), 180, np.uint8))
        self.assertFalse(check.ok)
        self.assertEqual(check.face_count, 0)
        self.assertTrue(any("No face" in p for p, _ in check.problems))

    def test_dark_photo_is_explained(self):
        check = face_detect.check_photo(np.full((480, 640, 3), 10, np.uint8))
        self.assertTrue(any("too dark" in p for p, _ in check.problems))

    def test_blur_lowers_sharpness(self):
        rng = np.random.default_rng(0)
        sharp = rng.integers(0, 255, (300, 300, 3), dtype=np.uint8)
        blurry = cv2.GaussianBlur(sharp, (21, 21), 8)
        self.assertGreater(face_detect.measure_sharpness(sharp), face_detect.MIN_SHARPNESS)
        self.assertLess(face_detect.measure_sharpness(blurry), face_detect.MIN_SHARPNESS)

    def test_load_image_shrinks_and_converts(self):
        import io
        from PIL import Image
        buf = io.BytesIO()
        Image.new("RGBA", (3000, 2000), (200, 150, 120, 255)).save(buf, format="PNG")
        buf.seek(0)
        rgb = face_detect.load_image(buf)
        self.assertEqual(rgb.shape[2], 3)
        self.assertLessEqual(max(rgb.shape[:2]), face_detect.MAX_SIDE)


class TestAgeCheck(unittest.TestCase):
    def patch(self, kind):
        """A skin-coloured grey patch with thin lines, round spots, or nothing."""
        img = np.full((60, 120), 170, np.uint8)
        if kind == "lines":
            for y in range(10, 55, 9):
                cv2.line(img, (5, y), (115, y + 3), 120, 1, cv2.LINE_AA)
        elif kind == "spots":
            for x in range(10, 115, 14):
                for y in (15, 35, 50):
                    cv2.circle(img, (x, y), 2, 120, -1, cv2.LINE_AA)
        return img

    def test_lines_score_high_spots_and_smooth_score_low(self):
        lines = age_check.line_strength(self.patch("lines"))
        spots = age_check.line_strength(self.patch("spots"))
        smooth = age_check.line_strength(self.patch("smooth"))
        self.assertGreater(lines, 3 * spots, "round spots (acne, pores) must not count as wrinkles")
        self.assertAlmostEqual(smooth, 0, places=4)

    def test_warning_rules(self):
        many = age_check.LineCheck(ok=True, score=60, level="many")
        some = age_check.LineCheck(ok=True, score=30, level="some")
        few = age_check.LineCheck(ok=True, score=10, level="few")
        self.assertIsNotNone(age_check.age_mismatch_warning(many, "18_24", "18–24"))
        self.assertIsNotNone(age_check.age_mismatch_warning(some, "under_18", "Under 18"))
        self.assertIsNone(age_check.age_mismatch_warning(some, "18_24"))      # some lines at 18–24: fine
        self.assertIsNone(age_check.age_mismatch_warning(few, "under_18"))
        self.assertIsNone(age_check.age_mismatch_warning(many, "45_plus"))   # older groups: never warn
        self.assertIsNone(age_check.age_mismatch_warning(many, None))         # age not given
        self.assertIsNone(age_check.age_mismatch_warning(None, "18_24"))      # check didn't run
        self.assertIn("18–24", age_check.age_mismatch_warning(many, "18_24", "18–24"))

    def test_no_face_is_handled(self):
        result = age_check.measure_lines(np.full((300, 300, 3), 180, np.uint8))
        self.assertFalse(result.ok)
        self.assertIsNone(age_check.age_mismatch_warning(result, "18_24"))

    def test_pdf_includes_age_warning(self):
        rec = recommend.build_routine({"skin_type": "dry", "age_group": "18_24"})
        lines = age_check.LineCheck(ok=True, score=60, level="many")
        warning = age_check.age_mismatch_warning(lines, "18_24", "18–24")
        pdf = report.build_pdf(None, None, rec, {"skin_type": "dry", "age_group": "18_24"}, lines, warning)
        self.assertTrue(pdf.startswith(b"%PDF"))


try:
    from skimage import data as skimage_data   # optional: pip install -r requirements-dev.txt
except ImportError:
    skimage_data = None


@unittest.skipIf(skimage_data is None, "scikit-image not installed (pip install -r requirements-dev.txt)")
class TestRealPortrait(unittest.TestCase):
    """Uses scikit-image's sample portrait (NASA astronaut Eileen Collins, public domain)."""

    def setUp(self):
        self.face = skimage_data.astronaut()

    def test_good_photo_passes_and_gives_a_tone(self):
        check = face_detect.check_photo(self.face)
        self.assertTrue(check.ok, check.problems)
        self.assertEqual(check.face_count, 1)
        tone = skin_tone.analyze_skin_tone(self.face)
        self.assertTrue(tone.ok, tone.message)
        self.assertIn(tone.tone, ("light", "medium", "tan", "deep"))
        self.assertIn(tone.undertone, ("warm", "cool", "neutral"))
        self.assertEqual(len(tone.regions), 4)             # both cheeks, forehead, chin
        self.assertTrue(tone.white_balanced)               # eye whites were found and used

    def test_undertone_is_stable_under_different_light_colours(self):
        def cast(r, g, b):
            return np.clip(self.face.astype(np.float32) * np.array([r, g, b]), 0, 255).astype(np.uint8)
        base = skin_tone.analyze_skin_tone(self.face)
        lights = {"warm bulb": cast(1.10, 1.00, 0.80), "cool screen": cast(0.90, 1.00, 1.15),
                  "pink LED": cast(1.08, 0.94, 1.02), "green tube": cast(0.95, 1.06, 0.95)}
        for name, img in lights.items():
            result = skin_tone.analyze_skin_tone(img)
            self.assertEqual(result.undertone, base.undertone, name)
            self.assertLess(abs(result.hue - base.hue), 8, name)
            self.assertEqual(result.tone, base.tone, name)

    def test_without_eye_whites_there_is_no_light_correction(self):
        pts = skin_tone.face_points(self.face)
        covered = self.face.copy()
        for eye in (skin_tone.LEFT_EYE, skin_tone.RIGHT_EYE):        # "sunglasses"
            cv2.fillPoly(covered, [pts[eye].astype(np.int32)], (20, 20, 20))
        result = skin_tone.analyze_skin_tone(covered)
        if result.ok:      # Face Mesh may still find the face
            self.assertFalse(result.white_balanced)
            self.assertTrue(any("eyes couldn't be measured" in n for n in result.notes))
            self.assertNotEqual(result.confidence, "high")

    def test_line_check_on_real_and_wrinkled_portrait(self):
        original = age_check.measure_lines(self.face)
        self.assertTrue(original.ok, original.message)
        self.assertEqual(original.level, "few")                      # early-40s portrait, smooth lighting
        self.assertIsNone(age_check.age_mismatch_warning(original, "18_24"))

        # Draw thin dark creases on the forehead, eye corners and under the eyes.
        pts = age_check._landmarks(self.face)
        d = np.linalg.norm(pts[263] - pts[33])
        wrinkled = self.face.copy()
        brow = min(pts[66][1], pts[296][1])
        for i in range(4):
            y = int(brow - 0.06 * d - i * 0.05 * d)
            cv2.line(wrinkled, (int(pts[33][0] + 0.15 * d), y), (int(pts[263][0] - 0.15 * d), y + 2), (90, 60, 45), 1, cv2.LINE_AA)
        for (cx, cy), sgn in ((pts[33], -1), (pts[263], 1)):
            for k in range(4):
                a = np.deg2rad(-35 + 22 * k)
                cv2.line(wrinkled, (int(cx + sgn * 0.04 * d), int(cy)),
                         (int(cx + sgn * 0.22 * d * np.cos(a)), int(cy + 0.2 * d * np.sin(a))), (90, 60, 45), 1, cv2.LINE_AA)
        for lid in (145, 374):
            x, y = pts[lid]
            for k in range(3):
                cv2.ellipse(wrinkled, (int(x), int(y + 0.1 * d + 0.03 * d * k)), (int(0.1 * d), 3), 0, 10, 170, (90, 60, 45), 1, cv2.LINE_AA)
        result = age_check.measure_lines(wrinkled)
        self.assertEqual(result.level, "many", result)
        self.assertIsNotNone(age_check.age_mismatch_warning(result, "18_24"))
        self.assertIsNone(age_check.age_mismatch_warning(result, "45_plus"))

    def test_each_bad_photo_gets_one_clear_reason(self):
        cases = {
            "too dark": (self.face * 0.18).astype(np.uint8),
            "blurry": cv2.GaussianBlur(self.face, (0, 0), 6),
            "2 faces": np.concatenate([self.face, self.face], axis=1),
        }
        for expected, img in cases.items():
            check = face_detect.check_photo(img)
            self.assertFalse(check.ok, expected)
            self.assertEqual(len(check.problems), 1, (expected, check.problems))
            self.assertIn(expected, check.problems[0][0])


class TestReport(unittest.TestCase):
    def test_pdf_is_built(self):
        px = lab_pixels(65, 12, 18)
        tone = skin_tone.describe(px, [px[:1500], px[1500:]])
        rec = recommend.build_routine({"skin_type": "oily", "concerns": ["acne", "dark_spots"], "acne_level": "cystic",
                                       "sensitive": True}, tone.tone)
        face = np.full((120, 100, 3), (200, 160, 130), np.uint8)
        pdf = report.build_pdf(face, tone, rec, {"skin_type": "oily", "concerns": ["acne"], "acne_level": "cystic"})
        self.assertTrue(pdf.startswith(b"%PDF"))
        self.assertGreater(len(pdf), 3000)

    def test_pdf_with_corrected_tone_and_age(self):
        px = lab_pixels(72, 10, 20)
        tone = skin_tone.with_user_tone(skin_tone.describe(px, [px[:1500], px[1500:]]), "tan")
        profile = {"skin_type": "combination", "concerns": ["dark_spots"], "age_group": "25_34"}
        rec = recommend.build_routine(profile, tone.tone)
        pdf = report.build_pdf(np.full((120, 100, 3), 150, np.uint8), tone, rec, profile)
        self.assertTrue(pdf.startswith(b"%PDF"))

    def test_pdf_without_tone(self):
        rec = recommend.build_routine({"skin_type": "dry", "concerns": []})
        self.assertTrue(report.build_pdf(None, None, rec, {"skin_type": "dry"}).startswith(b"%PDF"))

    def test_clean_handles_special_characters(self):
        self.assertEqual(report.clean("“Hi” – it’s 0.5–2% → ok…"), '"Hi" - it\'s 0.5-2% -> ok...')


if __name__ == "__main__":
    unittest.main()
