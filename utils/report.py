"""
report.py — builds the downloadable PDF report with fpdf2.

    build_pdf(face_crop, tone_result, recommendation, profile, lines=None, age_warning=None) -> bytes

The built-in PDF fonts only support basic Latin characters, so `clean()` swaps
curly quotes, dashes and arrows for plain ones before writing text.
"""

from datetime import date

from fpdf import FPDF
from fpdf.enums import XPos, YPos
from PIL import Image

from utils.skin_tone import tone_name

OLIVE = (104, 118, 75)
DARK = (48, 58, 44)
SOFT = (91, 100, 82)
SAGE = (232, 235, 221)
CAUTION = (255, 243, 217)

DISCLAIMER = ("This report is for general information only and is NOT a medical diagnosis. Skin tone is "
              "estimated from one photo and can be affected by lighting and camera settings. Skin type and "
              "concerns are based on your own answers. Please see a dermatologist for severe acne, painful "
              "or spreading skin problems, or any spot or mole that changes.")

_REPLACE = {"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-",
            "…": "...", "→": "->", "↔": "<->", "•": "-", " ": " ", "−": "-",
            "°": " deg"}


def clean(text) -> str:
    text = str(text)
    for bad, good in _REPLACE.items():
        text = text.replace(bad, good)
    return text.encode("latin-1", "replace").decode("latin-1")


class Report(FPDF):
    def header(self):
        self.set_fill_color(*OLIVE)
        self.rect(0, 0, self.w, 4, "F")

    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(*SOFT)
        self.cell(0, 8, clean(f"Glowie - your skin bestie - not a medical diagnosis - page {self.page_no()}"), align="C")

    # --- small building blocks
    def heading(self, text):
        if self.get_y() > self.h - 45:   # don't leave a heading alone at the bottom of a page
            self.add_page()
        self.ln(3)
        self.set_font("Helvetica", "B", 13)
        self.set_text_color(*OLIVE)
        self.cell(0, 8, clean(text), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_draw_color(*SAGE)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(2)

    def para(self, text, size=10, style=""):
        self.set_font("Helvetica", style, size)
        self.set_text_color(*DARK)
        self.multi_cell(0, 5.5, clean(text), new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    def bullets(self, items, size=10):
        self.set_font("Helvetica", "", size)
        self.set_text_color(*DARK)
        for item in items:
            self.set_x(self.l_margin + 3)
            self.multi_cell(0, 5.5, clean(f"- {item}"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    def box(self, text, fill):
        self.set_fill_color(*fill)
        self.set_font("Helvetica", "", 9)
        self.set_text_color(*DARK)
        self.multi_cell(0, 5, clean(text), fill=True, padding=3, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(2)

    def routine(self, title, steps):
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(*DARK)
        self.cell(0, 7, clean(title), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        for n, s in enumerate(steps, 1):
            self.set_font("Helvetica", "B", 10)
            self.set_x(self.l_margin + 3)
            self.multi_cell(0, 5.5, clean(f"{n}. {s['product']}"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            self.set_font("Helvetica", "", 9)
            self.set_text_color(*SOFT)
            detail = s.get("why", "") + (f" How: {s['start']}" if s.get("start") else "")
            self.set_x(self.l_margin + 8)
            self.multi_cell(0, 5, clean(detail), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            self.set_text_color(*DARK)
        self.ln(2)


def build_pdf(face_crop, tone, rec, profile, lines=None, age_warning=None) -> bytes:
    pdf = Report(format="A4")
    pdf.set_auto_page_break(auto=True, margin=16)
    pdf.set_margins(16, 14, 16)
    pdf.add_page()

    # Title
    pdf.set_font("Helvetica", "B", 20)
    pdf.set_text_color(*DARK)
    pdf.cell(0, 10, "Your Glowie Skin Report", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(*SOFT)
    pdf.cell(0, 6, clean(f"Glowie - your skin bestie - {date.today().strftime('%d %B %Y')}"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(3)
    pdf.box(DISCLAIMER, CAUTION)

    # Photo + colour swatch + results
    top = pdf.get_y()
    if face_crop is not None:
        pdf.image(Image.fromarray(face_crop), x=pdf.l_margin, y=top, w=38)
    x = pdf.l_margin + 44
    pdf.set_xy(x, top)
    if tone is not None and tone.ok:
        r, g, b = (int(tone.hex_color[i:i + 2], 16) for i in (1, 3, 5))
        pdf.set_fill_color(r, g, b)
        pdf.rect(x, top, 14, 14, "F")
        pdf.set_xy(x + 18, top)
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 6, clean(f"Skin tone: {tone_name(tone.tone)}   |   Undertone: {tone.undertone.title()}"),
                 new_x=XPos.LEFT, new_y=YPos.NEXT)
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(*SOFT)
        if tone.corrected_by_user:
            pdf.cell(0, 4.5, clean(f"Skin tone chosen by you (the photo measured {tone_name(tone.measured_tone)})."),
                     new_x=XPos.LEFT, new_y=YPos.NEXT)
        if tone.undertone_source != "photo":
            pdf.cell(0, 4.5, clean(f"Undertone from your answers (the photo suggested {tone.photo_undertone})."),
                     new_x=XPos.LEFT, new_y=YPos.NEXT)
        pdf.multi_cell(pdf.w - pdf.r_margin - pdf.get_x(), 4.5,
                       clean(f"Measured colour {tone.hex_color} - ITA {tone.ita} deg ({tone.ita_class}) - "
                             f"hue {tone.hue} deg - confidence: {tone.confidence}"), new_x=XPos.LEFT, new_y=YPos.NEXT)
        pdf.set_xy(x, max(pdf.get_y(), top + 15) + 2)
        pdf.set_text_color(*DARK)
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(0, 5.5, "Foundation shade range", new_x=XPos.LEFT, new_y=YPos.NEXT)
        pdf.set_font("Helvetica", "", 9.5)
        pdf.multi_cell(pdf.w - pdf.r_margin - x, 5, clean(f"{tone.foundation['summary']} Look for: {tone.foundation['undertone']}. "
                                                           f"{tone.foundation['tip']}"), new_x=XPos.LEFT, new_y=YPos.NEXT)
    else:
        pdf.para("Skin tone could not be measured from this photo.")
    pdf.set_y(max(pdf.get_y(), top + 40) + 2)

    if tone is not None and tone.ok and tone.notes:
        pdf.heading("About this colour estimate")
        pdf.bullets(tone.notes, size=9)

    # Profile
    pdf.heading("Your answers")
    acne = f" (acne: {profile.get('acne_level')})" if profile.get("acne_level") and "acne" in profile.get("concerns", []) else ""
    pdf.bullets([
        f"Skin type: {rec.skin_type_label}",
        f"Age group: {rec.age_group_label or 'not given'}",
        f"Concerns: {', '.join(rec.concerns) if rec.concerns else 'none selected'}{acne}",
        f"Sensitive skin: {'yes' if profile.get('sensitive') else 'no'}",
        f"Pregnant or breastfeeding: {'yes' if profile.get('pregnant') else 'no'}",
    ])
    if lines is not None and lines.ok:
        pdf.bullets([f"Fine lines measured in the photo: {lines.score:.0f}/100 ({lines.level}) - "
                     "used only to double-check the age group."], size=9)
    if age_warning:
        pdf.box(age_warning, CAUTION)

    # Routine
    pdf.heading("Your routine")
    pdf.routine("Morning", rec.morning)
    pdf.routine("Night", rec.night)

    pdf.heading("Ingredients")
    pdf.para("Look for:", style="B")
    pdf.bullets(rec.use)
    pdf.para("Avoid:", style="B")
    pdf.bullets(rec.avoid)

    pdf.heading("Tips")
    pdf.bullets(rec.tips)

    pdf.heading("When to see a dermatologist")
    if rec.see_doctor:
        for message in rec.see_doctor:
            pdf.box(message, CAUTION)
    pdf.bullets(["Severe, painful or scarring acne.",
                 "A rash or redness that spreads, burns, blisters or doesn't settle in 2 weeks.",
                 "Any spot or mole that changes size, shape or colour, itches or bleeds."])

    pdf.ln(3)
    pdf.set_font("Helvetica", "I", 8.5)
    pdf.set_text_color(*SOFT)
    pdf.multi_cell(0, 4.5, clean("Privacy: your photo was processed in memory to make this report and was not saved by the app."))
    return bytes(pdf.output())
