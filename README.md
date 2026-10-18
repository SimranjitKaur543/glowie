# Glowie 🌿 — your skin bestie

Glowie is a final-year AI skincare project: take or upload a selfie and get your **skin tone and undertone**, a
**foundation shade range**, and a **personalised morning & night skincare routine** — with a
downloadable **PDF report**.

> **Disclaimer:** this is a student project, **not a medical diagnosis**. Skin type and concerns
> come from the user's own answers; only skin tone is measured from the photo. Please see a
> dermatologist for severe acne or any unusual or changing skin condition.

## What it does

| Feature | How |
|---|---|
| **Face detection + photo checks** | MediaPipe Face Detection finds the face. Photos are rejected (with a "how to retake" tip) if there is no face, more than one face, the face is too small, or it is too dark, too bright or blurry. |
| **Skin tone & undertone** | MediaPipe Face Mesh locates both cheeks → skin pixels are kept (YCrCb filter, shadows/shine removed) → converted to **LAB** → **K-means** finds the main skin colour → **ITA°** (Individual Typology Angle) gives the tone (light / medium / tan / deep), the **hue angle** gives the undertone (warm / cool / neutral). A confidence level explains how reliable the photo was. |
| **Undertone** | (1) **Light correction from the eye whites**: Face Mesh (with iris points) finds the whites of the eyes; their colour tint compared with a normal eye-white colour is the tint of the light, which is removed from the skin colour. (2) The hue angle is measured on **both cheeks, the forehead and the chin**, and the middle value is used, so flushed red cheeks can't make everyone "cool". (3) Optional **3 classic questions** (vein colour, gold vs silver jewellery, sun reaction) — if at least 2 are answered they decide the undertone, and the app shows what the photo suggested. |
| **Tone names + correction** | Tones are shown as **Fair / Wheatish / Dusky / Deep**. Phone photos often look lighter than real skin (beauty mode, auto-brightening), so the app warns about very bright / unusually smooth photos and lets the user choose their real tone; the routine, foundation range and PDF then use that choice (the PDF also states what the photo measured). |
| **Foundation shade range** | Tone + undertone → a shade range and what undertone labels to look for (e.g. "W / Y / G"). |
| **Skin type** | Chosen by the user, or a 2-question quiz (oily / dry / normal / combination). |
| **Age group** | Chosen by the user (Under 18, 18–24, 25–34, 35–44, 45+). Adjusts the routine: no retinol for under-18s, antioxidants from 25, retinol introduced slowly from 35, richer moisturisers from 45. Pregnancy still removes retinol. |
| **Age check (fine lines)** | `utils/age_check.py` measures fine lines on the forehead, under the eyes and at the outer eye corners (Face Mesh areas, rescaled to the same size, Hessian "line" filter that ignores round spots like acne) → a 0–100 **line score**. If the score is clearly too high for the chosen group (many lines for 18–24, some or many for under-18) the app shows a warning asking the user to check their age group. It never changes the answer itself. |
| **Recommendation engine** | Rules in `data/recommendations.json`: base routine per skin type + extra steps for each concern + safety flags (sensitive skin, pregnancy) + skin-tone tips + "see a dermatologist" messages. |
| **Report** | On-screen results and a PDF (fpdf2). |

Not included (decided during the project): trained skin-type / acne / YOLOv8 models — they
require third-party datasets that we chose not to use. See *Future scope*.

## Folder structure

```
skincare-assistant/
├── app.py                    ← Streamlit web app
├── check_setup.py            ← checks your environment
├── requirements.txt          ← packages to run the app
├── requirements-lock.txt     ← exact tested versions
├── requirements-dev.txt      ← extra package for the tests (scikit-image)
├── .streamlit/config.toml    ← olive/cream theme, 15 MB upload limit, localhost only
├── data/recommendations.json ← all the skincare rules (edit freely)
├── utils/
│   ├── face_detect.py        ← load photo, find face, quality checks
│   ├── skin_tone.py          ← cheeks → LAB → K-means → ITA / hue → tone, undertone
│   ├── age_check.py          ← fine-line score → "check your age group" warning
│   ├── recommend.py          ← rule engine + skin-type quiz
│   └── report.py             ← PDF report
├── tests/test_modules.py     ← 41 automated tests
└── assets/                   ← logo / screenshots for your report
```

## Setup (Windows)

1. Install **Python 3.11** from python.org. If "Install Now" fails with error `0x80070003`,
   choose *Customize installation* → tick **Install for all users**.
2. In this folder:
   ```
   py -3.11 -m venv .venv
   .venv\Scripts\activate
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   python check_setup.py
   ```
   Expected: `[ OK ]` for every line and **"All good!"**.

## Run the app

```
.venv\Scripts\streamlit.exe run app.py --server.address localhost
```
Your browser opens http://localhost:8501. `--server.address localhost` means only your own
computer can open it. Keep the terminal open while using the app; press **Ctrl + C** to stop it.

## Deploy (Streamlit Community Cloud — free public link)

Vercel can't host this app (Streamlit needs a server that stays running; Vercel only runs
short-lived functions, and MediaPipe is too large for them). Use Streamlit Community Cloud:

1. Put this folder in its own GitHub repository (`.venv` and secrets are excluded by `.gitignore`).
2. Go to https://share.streamlit.io → sign in with GitHub → **Create app** → choose the repo,
   branch `main`, main file `app.py`.
3. **Advanced settings → Python version: 3.11**. Click **Deploy**.
4. First build takes ~5–10 minutes. `packages.txt` installs the Linux libraries OpenCV needs.
5. Your link: `https://<your-app-name>.streamlit.app`.

## Run the tests

```
pip install -r requirements-dev.txt     # once — adds a public-domain sample portrait
python -m unittest discover tests -v
```
Expected: `Ran 41 tests ... OK`. Without scikit-image, the 5 real-portrait tests are skipped.

## How the skin-tone maths works

- **LAB colour space** separates lightness (L) from colour (a = green↔red, b = blue↔yellow), so
  it's better than RGB for comparing skin colours.
- **K-means (k = 3)** groups the cheek pixels into 3 colours; the biggest group is the skin
  colour (the others are usually shadow, redness or shine).
- **ITA° = atan((L − 50) / b) × 180/π** — a published skin-colour measure (Chardon et al., 1991).
  Research classes: very light > 55°, light 41–55°, intermediate 28–41°, tan 10–28°,
  brown −30–10°, dark < −30°. The app groups them into light / medium / tan / deep.
- **Hue angle = atan2(b, a)**: more yellow → warm (> 58°), more pink/red → cool (< 48°),
  in between → neutral. *These undertone thresholds are a rule of thumb, not a validated method.*
- **Confidence** goes down when only a small cheek area is usable, the colour is uneven
  (fewer than 45% of pixels within ΔE 12 of the main colour), or one cheek is much brighter.

## Limitations (mention these in your report)

- One photo can't measure skin colour exactly: lighting colour, camera white balance, filters
  and makeup all change it. The app reports a confidence level and asks for daylight photos.
- The ITA ranges come from research with calibrated colour meters; phone photos tend to read
  lighter, so darker skin can be classified one level too light (seen in testing: a dusky
  skin tone shown as Fair). The user-correction option and the bright/filter warning address
  this, but the measurement itself is not calibrated for phone cameras.
- Age group is self-reported. The fine-line check only catches big mismatches (e.g. an elderly
  face marked 18–24); it cannot estimate age. Its thresholds (some ≥ 25, many ≥ 40) were set
  from one public-domain portrait (scored ~15) and drawn-on wrinkles (scored ~50–80), not from
  a real age dataset. Smiling, squinting, side lighting, glasses or a fringe can raise the
  score; beauty filters and small/low-resolution photos can hide lines.

## Undertone: what was tested

Before the fix, the same portrait under different light colours gave hue angles from **25° to
95°** (cool under a phone screen or pink LED, warm under a bulb) — the light, not the skin,
decided the undertone, and flushed cheeks pushed real photos towards "cool". After the
eye-white light correction and the 4-area median, the same face gives **about 58–70°** under
all five test lights and the same undertone each time (see `tests/test_modules.py`,
`test_undertone_is_stable_under_different_light_colours`). Limits: very strong coloured light
can still move a result close to a category border; if the eyes are closed, covered or in
shadow the correction is skipped (the app says so and lowers confidence); the warm/cool cut-offs
(58° / 48°) are a rule of thumb and have not been validated against professional shade
matching — the 3 questions are the more reliable way to decide undertone.

## Age check calibration (to improve it)

Run the app on a few real photos of different ages (with permission), and note the
**Fine lines** score shown under the photo. If young faces often score above 25, or older
faces below 40, change `SOME_LINES` / `MANY_LINES` at the top of `utils/age_check.py`.
- The undertone rule (hue angle thresholds) has not been validated against professional
  shade-matching.
- Skin type and concerns are self-reported, not detected.
- Accuracy hasn't been tested across many real users and skin tones — only on synthetic
  colours and one public-domain portrait.
- Advice is general, conservative and cosmetic-level; it is not medical advice.

## Privacy

Photos are processed in memory and never saved or sent to another service. Streamlit usage
statistics are switched off (`.streamlit/config.toml`).

## Future scope

Trained models for skin type (e.g. EfficientNetB0), acne grading (ACNE04) and skin-issue
detection (YOLOv8); camera colour calibration with a reference card; a user study across skin
tones; live virtual makeup try-on with Face Mesh.

## Common errors

| Error | Fix |
|---|---|
| `Python was not found` | Install Python 3.11 (see Setup) and open a **new** terminal. |
| `running scripts is disabled on this system` (PowerShell) | Use Command Prompt with `.venv\Scripts\activate.bat`, or skip activating and run `.venv\Scripts\streamlit.exe run app.py`. |
| `AttributeError: module 'cv2' has no attribute 'cvtColor'` | OpenCV files were damaged by an uninstall: `pip install --force-reinstall --no-deps opencv-contrib-python==4.10.0.84` |
| `A module compiled using NumPy 1.x cannot be run in NumPy 2.x` | `pip install "numpy==1.26.4"` (another package upgraded NumPy). |
| `module 'mediapipe' has no attribute 'solutions'` | `pip install mediapipe==0.10.14` |
| Photos don't show in the app (broken image) | You're on Streamlit 1.60+; run `pip install streamlit==1.50.0`. |
| "No face was found" for a real selfie | Face the camera, move closer, good light, remove sunglasses/mask. |
| Webcam option shows nothing | Allow camera access in the browser; close other apps using the camera. |
