"""
app.py — Glowie, your skin bestie: the web app (Streamlit).

Run it with:   streamlit run app.py

Flow: photo (upload or webcam) → face detection + quality checks → skin tone & undertone
from the cheeks → rule-based routine from your answers → on-screen report + PDF download.
"""

import hashlib
import html
from pathlib import Path

import streamlit as st

from utils.age_check import age_mismatch_warning, measure_lines
from utils.face_detect import check_photo, draw_face_box, load_image
from utils.recommend import build_routine, guess_skin_type, load_rules
from utils.report import DISCLAIMER, build_pdf
from utils.skin_tone import (TONE_ORDER, UNDERTONE_QUESTIONS, analyze_skin_tone, draw_regions, tone_name,
                             undertone_from_answers, with_user_tone, with_user_undertone)

APP_NAME = "Glowie"
TAGLINE = "your skin bestie"

st.set_page_config(page_title=f"{APP_NAME} — {TAGLINE}", page_icon="🌿", layout="wide",
                   initial_sidebar_state="collapsed")
rules = load_rules()

# ---------------------------------------------------------------- look & feel (UI only)
# All styling lives in assets/style.css so it can be changed without touching this file.
st.markdown(f"<style>{(Path(__file__).parent / 'assets' / 'style.css').read_text(encoding='utf-8')}</style>",
            unsafe_allow_html=True)


def section_title(number, title):
    """A small numbered heading used at the top of each card."""
    st.markdown(f"<div class='section-title'><span class='num'>{number}</span><h3>{title}</h3></div>",
                unsafe_allow_html=True)


def footer():
    """About, privacy and disclaimer — shown at the bottom of every page state."""
    st.divider()
    with st.expander(f"About {APP_NAME} · Privacy · Disclaimer"):
        st.markdown(
            f"**About.** {APP_NAME} is a final-year student project that looks at a selfie and suggests a simple "
            "skincare routine made just for you.\n\n"
            "**How it works.** 1) *Face detection* (MediaPipe) finds your face and checks the photo quality. "
            "2) *Skin tone* is measured from your cheeks using LAB colour and K-means clustering. "
            "3) *Your answers* about skin type and concerns choose a routine from a set of rules.\n\n"
            "**Privacy.** Your photo is processed in memory on the server running this app and is **not saved**. "
            "Nothing is sent to any other service.")
        st.caption(DISCLAIMER)
    st.markdown(f"<div class='glowie-footer'><div class='brand'>{APP_NAME.lower()} ✨</div>"
                f"{TAGLINE} · a student project · not a medical device</div>", unsafe_allow_html=True)


def stop():
    """End the page early, but still show the footer."""
    footer()
    st.stop()


# ---------------------------------------------------------------- hero + how it works
st.markdown(
    f"""<div class="glowie-hero">
          <h1>{APP_NAME.lower()} ✨</h1>
          <p class="tagline">{TAGLINE}</p>
          <p class="intro">Take or upload a selfie, tell me a little about your skin, and I'll find your skin tone,
          a foundation shade range and a morning &amp; night routine just for you.</p>
        </div>
        <div class="glowie-steps">
          <div class="glowie-step"><span class="num">1</span>Snap a selfie<span class="arrow">→</span></div>
          <div class="glowie-step"><span class="num">2</span>Tell me about your skin<span class="arrow">→</span></div>
          <div class="glowie-step"><span class="num">3</span>Get your routine</div>
        </div>""",
    unsafe_allow_html=True)
st.info("**Not a medical diagnosis.** For severe acne or any unusual or changing skin problem, "
        "please see a dermatologist.", icon="ℹ️")

left, right = st.columns(2, gap="large")

# ---------------------------------------------------------------- 1. photo
with left, st.container(border=True):
    section_title(1, "Your photo")
    source = st.radio("Photo source", ["Upload a selfie", "Use the webcam"], horizontal=True,
                      label_visibility="collapsed")
    if source == "Upload a selfie":
        photo = st.file_uploader("Upload a clear, front-facing selfie", type=["jpg", "jpeg", "png", "webp"])
    else:
        photo = st.camera_input("Look straight at the camera in good light")
    with st.expander("Tips for an accurate result"):
        st.markdown("- Face a window in **daylight**; avoid coloured lights and flash.\n"
                    "- **No makeup, filters or beauty mode.**\n"
                    "- Only **your** face in the frame, looking straight ahead.\n"
                    "- Move your hair away from your cheeks.")

# ---------------------------------------------------------------- 2. questions
with right, st.container(border=True):
    section_title(2, "About your skin")
    type_labels = {"Not sure — help me decide": None, "Oily": "oily", "Dry": "dry",
                   "Normal": "normal", "Combination (oily T-zone, normal/dry cheeks)": "combination"}
    choice = st.selectbox("What is your skin type?", list(type_labels))
    skin_type = type_labels[choice]
    if skin_type is None:
        after = st.radio("About an hour after washing your face, it feels…",
                         ["tight", "comfortable", "oily"], horizontal=True,
                         format_func={"tight": "Tight / dry", "comfortable": "Comfortable", "oily": "Already oily"}.get)
        midday = st.radio("By the middle of the day, your face is…",
                          ["shiny_all", "shiny_tzone", "flaky", "fine"],
                          format_func={"shiny_all": "Shiny all over", "shiny_tzone": "Shiny on forehead, nose & chin only",
                                       "flaky": "Dry or flaky in places", "fine": "Comfortable, not shiny"}.get)
        skin_type = guess_skin_type(after, midday)
        st.caption(f"Based on your answers, your skin type is probably **{rules['skin_types'][skin_type]['label']}**.")

    age_groups = rules["age_groups"]
    age_group = st.selectbox("Your age group", [None, *age_groups],
                             format_func=lambda k: "Prefer not to say" if k is None else age_groups[k]["label"],
                             help="Skin changes with age, so the routine is adjusted for your age group.")

    concern_keys = list(rules["concerns"])
    concerns = st.multiselect("Your main concerns (optional)", concern_keys,
                              format_func=lambda k: rules["concerns"][k]["label"])
    acne_level = None
    if "acne" in concerns:
        acne_level = st.radio("How would you describe your acne?", ["few", "many", "cystic"],
                              format_func={"few": "A few spots now and then",
                                           "many": "Many spots most of the time",
                                           "cystic": "Deep, painful lumps (cysts)"}.get)
    sensitive = st.checkbox("My skin is sensitive (stings, burns or reacts to products easily)")
    pregnant = st.checkbox("I am pregnant or breastfeeding")
    changing_spot = st.checkbox("I have a spot or mole that is changing, bleeding or itching")

    with st.expander("🎨 Know your undertone better (3 quick questions, optional)"):
        st.caption("Undertone is hard to judge from a photo. These classic checks are more reliable — "
                   "if you answer at least 2, they decide your undertone.")
        undertone_answers = {}
        for key, (question, options) in UNDERTONE_QUESTIONS.items():
            answer = st.radio(question, list(options), index=None, key=f"ut_{key}",
                              format_func=lambda k, o=options: o[k][0])
            if answer:
                undertone_answers[key] = answer
    answered_undertone = undertone_from_answers(undertone_answers)

profile = {"skin_type": skin_type, "concerns": concerns, "acne_level": acne_level,
           "sensitive": sensitive, "pregnant": pregnant, "changing_spot": changing_spot,
           "age_group": age_group}

# ---------------------------------------------------------------- 3. analyse
st.write("")
photo_id = hashlib.md5(photo.getvalue()).hexdigest() if photo is not None else None
if st.button("Analyse my skin ✨", type="primary", disabled=photo is None, width="stretch"):
    with st.spinner("Finding your face and measuring your skin tone…"):
        try:
            image = load_image(photo)
        except Exception:
            st.warning("That file couldn't be opened as an image. Please try a JPG or PNG photo.")
            stop()
        check = check_photo(image)
        tone = analyze_skin_tone(image) if check.ok else None
        lines = measure_lines(image) if check.ok else None
    st.session_state["analysis"] = {"id": photo_id, "check": check, "tone": tone, "lines": lines}

analysis = st.session_state.get("analysis")
if photo is None:
    st.caption("Add a photo above to start.")
    stop()
if not analysis or analysis["id"] != photo_id:
    st.caption("Press **Analyse my skin** when you're ready.")
    stop()

check, tone, lines = analysis["check"], analysis["tone"], analysis.get("lines")

# Photo problems → explain how to retake it, and stop.
if not check.ok:
    st.warning("We couldn't analyse this photo. Please retake it:")
    for problem, fix in check.problems:
        st.markdown(f"- **{problem}** {fix}")
    st.image(draw_face_box(check.image, check.face_box), caption="Your photo", width=360)
    stop()

# ---------------------------------------------------------------- 4. results
st.write("")
st.markdown("<h2>Your results</h2>", unsafe_allow_html=True)
col_img, col_tone = st.columns([1, 1.4], gap="large")
with col_img, st.container(border=True):
    section_title("📷", "Your photo")
    annotated = draw_face_box(check.image, check.face_box)
    if tone and tone.regions:
        annotated = draw_regions(annotated, tone.regions)
    st.image(annotated, caption=f"Face found ({check.face_score:.0%} sure). Outlined: cheek areas measured.",
             width="stretch")
    q1, q2, q3 = st.columns(3)
    q1.metric("Brightness (0–255)", f"{check.brightness:.0f}")
    q2.metric("Sharpness", f"{check.sharpness:.0f}")
    if lines and lines.ok:
        q3.metric("Fine lines", f"{lines.score:.0f}/100", lines.level, delta_color="off",
                  help="How many fine lines / wrinkles show on the forehead, under the eyes and at the eye "
                       "corners. Used only to double-check the age group you chose.")
    if lines and lines.message:
        st.caption("• " + lines.message)

with col_tone, st.container(border=True):
    section_title("🎨", "Skin tone & undertone")
    if tone and tone.ok:
        st.markdown(
            f"<div class='tone-swatch-row'><div class='tone-swatch' style='background:{tone.hex_color}'></div>"
            f"<div><div class='label'>Your measured cheek colour</div><code>{tone.hex_color}</code></div></div>",
            unsafe_allow_html=True)

        # Warnings that affect the result go first and stand out.
        for note in tone.notes:
            (st.warning if "LIGHTER" in note else st.caption)(("⚠️ " if "LIGHTER" in note else "• ") + note)

        # The photo can be wrong (filters, lighting), so let the user correct the tone.
        measured = tone.tone
        choice = st.radio(
            f"The photo suggests **{tone_name(measured)}**. Does this match your skin in daylight?",
            TONE_ORDER, index=TONE_ORDER.index(measured), horizontal=True, key=f"tone_{photo_id}",
            format_func=lambda t: tone_name(t) + (" (measured)" if t == measured else ""))
        tone = with_user_tone(tone, choice)
        if tone.corrected_by_user:
            st.caption(f"Using **{tone_name(tone.tone)}** as you chose — the routine, foundation range and PDF "
                       "use your choice.")
        # Undertone: the 3 questions (if answered) are more reliable than the photo.
        tone = with_user_undertone(tone, answered_undertone)

        m1, m2, m3 = st.columns(3)
        m1.metric("Skin tone", tone_name(tone.tone))
        m2.metric("Undertone", tone.undertone.title(),
                  "from your answers" if answered_undertone else "from the photo", delta_color="off")
        m3.metric("Photo confidence", tone.confidence.title())
        if answered_undertone and answered_undertone != tone.photo_undertone:
            st.caption(f"Your answers say **{answered_undertone}**; the photo suggested "
                       f"**{tone.photo_undertone}**. Your answers are used — they're more reliable than a photo.")
        elif not answered_undertone:
            st.caption("For a more reliable undertone, answer the 3 quick questions in *Know your undertone better* above.")
        st.markdown(f"<div class='foundation-box'><b>Foundation shade range:</b> "
                    f"{html.escape(tone.foundation['summary'])}<br>"
                    f"Look for shades marked <b>{html.escape(tone.foundation['undertone'])}</b>.<br>"
                    f"<i>{html.escape(tone.foundation['tip'])}</i></div>", unsafe_allow_html=True)
        with st.expander("How was this measured?"):
            st.markdown(
                f"- Main cheek colour (K-means, LAB): L = {tone.lab[0]}, a = {tone.lab[1]}, b = {tone.lab[2]}\n"
                f"- **ITA angle** = atan((L − 50) / b) = **{tone.ita}°** → *{tone.ita_class}* "
                f"(app groups: Fair > 41°, Wheatish 28–41°, Dusky 10–28°, Deep ≤ 10°)\n"
                f"- **Light correction:** "
                + ("the colour tint of the light was measured on the whites of your eyes and removed."
                   if tone.white_balanced else "not possible (eye whites not visible) — undertone is less reliable.")
                + "\n"
                f"- **Hue angle** = atan2(b, a), the middle value of "
                + ", ".join(f"{k} {v}°" for k, v in tone.region_hues.items())
                + f" = **{tone.hue}°** → photo says *{tone.photo_undertone}* (warm > 58°, cool < 48°)\n"
                f"- These ranges come from skin-colour research; phone photos can still differ from real "
                f"skin, which is why you can correct the tone and answer the undertone questions.")
    else:
        st.warning(tone.message if tone else "Skin tone could not be measured.")

# ---------------------------------------------------------------- 5. routine
rec = build_routine(profile, tone.tone if tone and tone.ok else None)
age_warning = age_mismatch_warning(lines, age_group, rec.age_group_label)


def show_steps(steps):
    """Each routine step as a numbered mini-card (text escaped, so it can't break the page)."""
    for n, s in enumerate(steps, 1):
        how = f"<div class='how'>How: {html.escape(s['start'])}</div>" if s.get("start") else ""
        st.markdown(f"<div class='step-card'><span class='n'>{n}</span><div>"
                    f"<div class='what'>{html.escape(s['product'])}</div>"
                    f"<div class='why'>{html.escape(s['why'])}</div>{how}</div></div>", unsafe_allow_html=True)


st.write("")
with st.container(border=True):
    section_title("🌸", "Your skincare routine")
    st.caption(f"Skin type: **{rec.skin_type_label}** · Age group: **{rec.age_group_label or 'not given'}** · "
               f"Concerns: **{', '.join(rec.concerns) if rec.concerns else 'none selected'}** — "
               "based on your answers (skin type, age and concerns are not detected from the photo).")
    if age_warning:
        st.warning("🎂 " + age_warning)
    for message in rec.see_doctor:
        st.error("🩺 " + message)        # medical advice → soft rose box

    morning, night, ingredients, tips = st.tabs(["☀️ Morning", "🌙 Night", "🧪 Ingredients", "💡 Tips"])
    with morning:
        show_steps(rec.morning)
    with night:
        show_steps(rec.night)
    with ingredients:
        c1, c2 = st.columns(2)
        c1.markdown("**✅ Look for**\n" + "\n".join(f"- {i}" for i in rec.use))
        c2.markdown("**🚫 Avoid**\n" + "\n".join(f"- {i}" for i in rec.avoid))
    with tips:
        st.markdown("\n".join(f"- {t}" for t in rec.tips))

# ---------------------------------------------------------------- 6. PDF
st.write("")
pdf_bytes = build_pdf(check.face_crop, tone, rec, profile, lines, age_warning)
st.download_button("📄 Download my report (PDF)", data=pdf_bytes, file_name="skincare_report.pdf",
                   mime="application/pdf", type="primary", width="stretch")
footer()
