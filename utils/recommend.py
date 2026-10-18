"""
recommend.py — the rule-based recommendation engine.

All advice lives in data/recommendations.json; this file only combines the rules:

    build_routine(profile, tone) -> Recommendation

profile = {
    "skin_type": "oily" | "dry" | "normal" | "combination",
    "concerns": ["acne", "dark_spots", ...],           # keys from the JSON "concerns"
    "acne_level": "few" | "many" | "cystic" | None,
    "sensitive": bool,
    "pregnant": bool,                                   # pregnant or breastfeeding
    "changing_spot": bool,                              # a spot/mole that is changing
    "age_group": "under_18" | "18_24" | "25_34" | "35_44" | "45_plus" | None,
}
tone = "light" | "medium" | "tan" | "deep" | None       # from skin_tone.py
"""

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

RULES_FILE = Path(__file__).resolve().parents[1] / "data" / "recommendations.json"


@lru_cache(maxsize=1)
def load_rules(path: str = str(RULES_FILE)) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@dataclass
class Recommendation:
    skin_type: str
    skin_type_label: str
    concerns: list                                   # labels, e.g. ["Acne / breakouts"]
    age_group_label: str = ""                        # e.g. "25–34" ("" if not given)
    morning: list = field(default_factory=list)      # [{"step", "product", "why", "start"?}]
    night: list = field(default_factory=list)
    use: list = field(default_factory=list)          # key ingredients to look for
    avoid: list = field(default_factory=list)        # ingredients / habits to avoid
    tips: list = field(default_factory=list)
    see_doctor: list = field(default_factory=list)   # reasons to see a dermatologist / doctor


def _add_unique(target: list, items):
    for item in items:
        if item not in target:
            target.append(item)


def _merge_steps(target: list, steps):
    """Adds routine steps, skipping exact duplicates (same step + product)."""
    for step in steps:
        if not any(s["step"] == step["step"] and s["product"] == step["product"] for s in target):
            target.append(dict(step))


def _replace_tagged(rec, remove_tags, replacements):
    """Swaps routine steps carrying one of `remove_tags` (e.g. "retinoid") for a safer step."""
    for routine in (rec.morning, rec.night):
        for i, step in enumerate(routine):
            removed = set(step.get("tags", [])) & set(remove_tags)
            if removed:
                routine[i] = dict(replacements[removed.pop()])
        # the replacement might now appear twice — keep the first one only
        seen, unique = set(), []
        for step in routine:
            key = (step["step"], step["product"])
            if key not in seen:
                seen.add(key)
                unique.append(step)
        routine[:] = unique


def guess_skin_type(after_washing: str, midday: str) -> str:
    """Two-question quiz for users who aren't sure of their skin type.
    after_washing: "tight" | "comfortable" | "oily"     (an hour after washing)
    midday:        "shiny_all" | "shiny_tzone" | "flaky" | "fine"
    """
    if midday == "shiny_tzone":
        return "combination"
    if midday == "shiny_all" or after_washing == "oily":
        return "oily"
    if midday == "flaky" or after_washing == "tight":
        return "dry"
    return "normal"


def build_routine(profile: dict, tone: str = None, rules: dict = None) -> Recommendation:
    rules = rules or load_rules()
    skin_type = profile.get("skin_type", "normal")
    if skin_type not in rules["skin_types"]:
        raise ValueError(f"Unknown skin type: {skin_type}")
    base = rules["skin_types"][skin_type]
    concerns = [c for c in profile.get("concerns", []) if c in rules["concerns"]]

    rec = Recommendation(skin_type=skin_type, skin_type_label=base["label"],
                         concerns=[rules["concerns"][c]["label"] for c in concerns])

    # 1. Base routine for the skin type
    _merge_steps(rec.morning, base["morning"])
    _merge_steps(rec.night, base["night"])
    _add_unique(rec.use, base["ingredients"])
    _add_unique(rec.avoid, base["avoid"])

    # 2. Add each concern
    for key in concerns:
        c = rules["concerns"][key]
        _merge_steps(rec.morning, c["morning"])
        _merge_steps(rec.night, c["night"])
        _add_unique(rec.use, c["ingredients"])
        _add_unique(rec.avoid, c["avoid"])
        _add_unique(rec.tips, c.get("tips", []))

    # 3. Age group (self-reported): extra steps, ingredients and tips for that age
    age = rules.get("age_groups", {}).get(profile.get("age_group") or "")
    if age:
        rec.age_group_label = age["label"]
        _merge_steps(rec.morning, age["morning"])
        _merge_steps(rec.night, age["night"])
        _add_unique(rec.use, age["ingredients"])
        _add_unique(rec.tips, age["tips"])
        if age.get("remove_tags"):            # e.g. no retinol for under-18s
            _replace_tagged(rec, age["remove_tags"], age["replace"])
            rec.use = [i for i in rec.use if i != "Retinol"]

    # 4. Safety flags
    if profile.get("pregnant"):
        flag = rules["flags"]["pregnant"]
        _replace_tagged(rec, flag["remove_tags"], flag["replace"])
        rec.use = [i for i in rec.use if i not in flag["remove_ingredients"]]
        _add_unique(rec.tips, flag["tips"])
    if profile.get("sensitive"):
        flag = rules["flags"]["sensitive"]
        _add_unique(rec.avoid, flag["avoid"])
        _add_unique(rec.tips, flag["tips"])

    # 5. Don't stack strong actives on the same night
    night_text = " ".join(s["product"].lower() for s in rec.night)
    strong = [name for name in ("retinol", "salicylic", "lactic") if name in night_text]
    if len(strong) >= 2:
        _add_unique(rec.tips, ["Your night routine has more than one strong active "
                               f"({', '.join(strong)}). Use them on different nights, never together."])

    # 6. Tips for the measured skin tone
    if tone in rules["tone_tips"]:
        _add_unique(rec.tips, rules["tone_tips"][tone])
    _add_unique(rec.tips, rules["general_tips"])

    # 7. When to see a dermatologist / doctor
    doctor = rules["dermatologist"]
    if "acne" in concerns and profile.get("acne_level") == "cystic":
        rec.see_doctor.append(doctor["cystic_acne"])
    elif "acne" in concerns and profile.get("acne_level") == "many":
        rec.see_doctor.append(doctor["many_acne"])
    if "redness" in concerns:
        rec.see_doctor.append(doctor["persistent_redness"])
    if profile.get("changing_spot"):
        rec.see_doctor.append(doctor["changing_spot"])

    # 8. Put the steps in the usual order (cleanser → treatment → moisturiser → sunscreen)
    order = rules["step_order"]
    rec.morning.sort(key=lambda s: order.get(s["step"], 99))
    rec.night.sort(key=lambda s: order.get(s["step"], 99))
    return rec
