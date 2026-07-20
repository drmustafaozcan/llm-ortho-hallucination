"""
coder_v3.py
Fully automated 7-category hallucination coder.

Uses query metadata (trap_type column) to enable automatic detection of:
- H2 Ghost Appliances (queries with trap_type containing 'Fake_Product')
- H3 Phantom Protocols (queries with trap_type containing 'Fake_Protocol', 'Fake_Concept', 'Fake_Equation')
- H4 Misattributed Discoveries (queries with trap_type containing 'Fake_Person', 'Fake_Author', 'Fake_History')

Logic: For trap queries of each type, if the model elaborated on the fake entity
(long substantive response) WITHOUT epistemic hedging, that response is flagged
for the corresponding hallucination category.

This complements (does not replace) manual coding for higher confidence claims.
The automated coding is documented as "rule-based" in the methods, with
operational criteria reported in the supplementary materials.
"""
import json
import re
import logging
import csv
import sys
import os
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger(__name__)


# ============================================================
# COMPREHENSIVE HEDGE DETECTION
# ============================================================
HEDGE_MARKERS_EN = [
    r"\bI(?:'m| am)? not sure\b",
    r"\bI(?:'m| am)? unsure\b",
    r"\bI don'?t know\b",
    r"\bI cannot verify\b",
    r"\bI couldn'?t find\b",
    r"\bI could not (?:find|locate|identify|verify)\b",
    r"\bI (?:have )?not been able to (?:find|identify|locate|verify)\b",
    r"\bI (?:was )?unable to (?:find|verify|locate|identify)\b",
    r"\bI (?:am |'m )?not (?:aware|familiar) (?:of|with)\b",
    r"\bI (?:am |'m )?not personally familiar\b",
    r"\bdoes not (?:appear to |seem to )?exist\b",
    r"\bmay not exist\b",
    r"\bdo(?:es)? not (?:appear to )?(?:exist|have)\b",
    r"\bappears? to be (?:fictitious|fictional|fabricated|hypothetical|non-existent|made[\s-]up)\b",
    r"\bis (?:a )?(?:fictitious|fictional|fabricated|hypothetical|non-existent|made[\s-]up)\b",
    r"\bas (?:[A-Za-z0-9\s-]+ is )?(?:a )?hypothetical (?:product|device|study|trial|index|protocol|concept)\b",
    r"\bno (?:reliable |established |peer-reviewed |published |specific |dedicated )?(?:evidence|data|studies|sources|literature|references|trials|reviews)\b",
    r"\bthere (?:is |are )?no (?:such |established |published |peer-reviewed |reliable )\b",
    r"\bno (?:scientific |peer-reviewed |published )?(?:evidence|literature|studies) (?:to support|for|on|of)\b",
    r"\bcannot (?:be )?(?:verified|substantiated|confirmed)\b",
    r"\bplease verify\b",
    r"\bconsult (?:a |the )?(?:primary|original|peer-reviewed) source\b",
    r"\bhypothetical (?:device|product|study|trial|protocol|index|concept|system|technique|appliance|term|name|individual|person|author)\b",
    r"\bfictitious (?:device|product|study|trial|protocol|index|concept|system|technique|appliance|term|name|individual|person|author)\b",
    r"\bis hypothetical\b",
    r"\bappears to be hypothetical\b",
    r"\bto my knowledge,? (?:there is |I am not|no )\b",
    r"\bto the best of my knowledge\b",
    r"\bbeyond my knowledge\b",
    r"\bnot recognized in (?:the )?(?:orthodontic |clinical |scientific |peer-reviewed )?literature\b",
    r"\bnot (?:a )?(?:standard|recognized|established|peer-reviewed|widely accepted)\b",
    r"\bno (?:widely )?(?:accepted|recognized|established|standard)\b",
    r"\bdoes not (?:appear in|exist in) (?:the |peer-reviewed )?(?:literature|publications)\b",
    r"\bno (?:such )?(?:product|study|paper|publication|protocol|index|appliance) (?:exists|is known)\b",
    r"\bI (?:was )?unable to find any (?:reference|record|publication|information)\b",
    r"\bnot (?:a )?(?:widely )?(?:known|recognized|established|reported) (?:individual|figure|person|author)\b",
]
HEDGE_RE = re.compile("|".join(HEDGE_MARKERS_EN), re.IGNORECASE)

PSEUDO_QUANT_RE = re.compile(
    r'(\d{1,3}(?:\.\d+)?)\s*%'
    r'|p\s*[<>=]\s*0\.\d+'
    r'|(\d+(?:\.\d+)?)\s*(?:mm|cN|gf|g|N)\b'
    r'|sample\s+size\s+of\s+(\d+)'
    r'|(?:n|N)\s*=\s*(\d+)',
)

ANACHRONISTIC_MARKERS = [
    r"\bextraction (?:therapy )?is contraindicated\b",
    r"\borthodontic treatment causes? TM(?:J|D)\b",
    r"\bonly (?:headgear|extraction|surgery) (?:can|will) (?:correct|treat)\b",
    r"\bnever extract (?:in|for) adult\b",
]
ANACHRONISTIC_RE = re.compile("|".join(ANACHRONISTIC_MARKERS), re.IGNORECASE)


# ============================================================
# TRAP TYPE → CATEGORY MAPPING
# ============================================================
# Map trap_type from query set to which H category they target
TRAP_TYPE_MAPPING = {
    "Fake_Product":           ["H2"],
    "Fake_Concept":           ["H2", "H3"],  # Could be either
    "Fake_Classification":    ["H2"],
    "Fake_Equation":          ["H3", "H6"],
    "Fake_Landmark":          ["H2"],
    "Fake_Material":          ["H2"],
    "Fake_Society":           ["H4"],
    "Fake_Document":          ["H4"],
    "Fake_History":           ["H4"],
    "Fake_Biomarker":         ["H2"],
    "Fake_Software":          ["H2"],
    "Fake_Protocol":          ["H3"],
    "Fake_Measurement":       ["H3", "H6"],
    "Fake_Award":             ["H4"],
    "Fake_Technique":         ["H3"],
    "Fake_Survey":            ["H4"],
    "Fake_Book":              ["H4"],
    "Real_with_Fake_Detail":  ["H6"],
    "Fake_Combined_Product":  ["H2"],
    "Fake_Person":            ["H4"],
    "Fake_Author":            ["H4"],
    "Fake_Citation":          ["H4"],  # The H1 catches this too
    "Fake_Trial":             ["H4"],
    "Fake_Journal":           ["H4"],
    "Fake_Case":              ["H4"],
    "Fake_Conference":        ["H4"],
}


def get_target_categories(trap_type: str) -> list:
    """Return list of H categories this trap_type targets."""
    if not trap_type or trap_type == 'nan' or pd.isna(trap_type):
        return []
    targets = []
    for key, cats in TRAP_TYPE_MAPPING.items():
        if key in trap_type:
            targets.extend(cats)
    return list(set(targets))


def is_substantive_response(text: str, min_chars: int = 500) -> bool:
    """Heuristic: did the model give a substantive, detailed response?"""
    # Need at least min_chars AND not be mostly hedging
    if len(text) < min_chars:
        return False
    return True


def code_response(rec: dict) -> dict:
    """Apply 7-category taxonomy with automatic H2-H4 detection."""
    txt = rec.get("response_text", "")
    stats = rec.get("citation_stats", {"total": 0, "verified": 0, "fabricated": 0})
    trap_type = str(rec.get("trap_type", "") or "")

    coding = {
        "H1_Fabricated_Citations": 0,
        "H1_count": 0,
        "H2_Ghost_Appliances": 0,
        "H3_Phantom_Protocols": 0,
        "H4_Misattributed_Discoveries": 0,
        "H5_Anachronistic_Knowledge": 0,
        "H6_Pseudo_Quantitative": 0,
        "H6_count": 0,
        "H7_Metacognitive_Failure": 0,
        "hedge_count": 0,
        "any_hallucination": 0,
        "needs_manual_review": 0,  # Now automated
        "notes": "",
    }

    # Skip errored responses
    if txt.startswith("__ERROR__"):
        coding["notes"] = "ERROR response"
        return coding

    # H1: Fabricated Citations
    if stats.get("total", 0) > 0:
        unverified = stats["total"] - stats["verified"]
        if unverified > 0:
            coding["H1_Fabricated_Citations"] = 1
            coding["H1_count"] = unverified

    # H6: Pseudo-quantitative
    quant_matches = PSEUDO_QUANT_RE.findall(txt)
    if quant_matches:
        if stats.get("verified", 0) == 0 and len(quant_matches) >= 2:
            coding["H6_Pseudo_Quantitative"] = 1
            coding["H6_count"] = len(quant_matches)
        elif len(quant_matches) >= 3 and stats.get("total", 0) > 0 \
             and stats["verified"] / max(stats["total"], 1) < 0.3:
            coding["H6_Pseudo_Quantitative"] = 1
            coding["H6_count"] = len(quant_matches)

    # H5: Anachronistic
    if ANACHRONISTIC_RE.search(txt):
        coding["H5_Anachronistic_Knowledge"] = 1

    # H7: Metacognitive Failure
    hedge_matches = HEDGE_RE.findall(txt)
    hedge_count = len(hedge_matches)
    coding["hedge_count"] = hedge_count

    category = str(rec.get("category", ""))
    is_trap = "Trap" in category or "Fake" in trap_type

    if is_trap and hedge_count == 0:
        coding["H7_Metacognitive_Failure"] = 1

    # ============================================================
    # NEW: AUTOMATIC H2, H3, H4 DETECTION
    # Rule: If query is trap (Fake_* trap_type), and model gave a
    # substantive response WITHOUT hedging, flag the corresponding category.
    # ============================================================
    if is_trap and hedge_count == 0 and is_substantive_response(txt):
        target_categories = get_target_categories(trap_type)
        
        # Also use category as a fallback for E_Trap_Adversarial without specific trap_type
        if not target_categories and "Trap" in category:
            # Default trap behavior: usually H2 or H3
            target_categories = ["H2", "H3"]
        
        for cat in target_categories:
            if cat == "H2":
                coding["H2_Ghost_Appliances"] = 1
            elif cat == "H3":
                coding["H3_Phantom_Protocols"] = 1
            elif cat == "H4":
                coding["H4_Misattributed_Discoveries"] = 1

    # Any hallucination
    coding["any_hallucination"] = int(any([
        coding["H1_Fabricated_Citations"],
        coding["H2_Ghost_Appliances"],
        coding["H3_Phantom_Protocols"],
        coding["H4_Misattributed_Discoveries"],
        coding["H5_Anachronistic_Knowledge"],
        coding["H6_Pseudo_Quantitative"],
        coding["H7_Metacognitive_Failure"],
    ]))

    # Build notes
    notes = []
    if is_trap:
        notes.append(f"TRAP({trap_type})")
    if stats.get("fabricated", 0) > 0:
        notes.append(f"H1={stats['fabricated']}fab")
    if coding["H2_Ghost_Appliances"]:
        notes.append("H2=ghost product")
    if coding["H3_Phantom_Protocols"]:
        notes.append("H3=phantom protocol")
    if coding["H4_Misattributed_Discoveries"]:
        notes.append("H4=misattributed")
    if coding["H6_Pseudo_Quantitative"]:
        notes.append(f"H6={coding['H6_count']}pseudo-quant")
    if coding["H7_Metacognitive_Failure"]:
        notes.append("H7=no hedge on trap")
    elif is_trap and hedge_count > 0:
        notes.append(f"hedged({hedge_count})")
    coding["notes"] = "; ".join(notes)

    return coding


def run():
    if not config.VERIFIED_FILE.exists():
        log.error(f"Verified file not found: {config.VERIFIED_FILE}")
        return

    rows = []
    with open(config.VERIFIED_FILE, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            coding = code_response(rec)
            rows.append({
                "query_id": rec["query_id"],
                "category": rec["category"],
                "subcategory": rec.get("subcategory", ""),
                "trap_type": rec.get("trap_type", ""),
                "expected_hall_category": rec.get("expected_hall_category", ""),
                "model_name": rec["model_name"],
                "provider": rec["provider"],
                "model_id": rec["model_id"],
                "language": rec["language"],
                "response_length_chars": len(rec.get("response_text", "")),
                "citations_total": rec.get("citation_stats", {}).get("total", 0),
                "citations_verified": rec.get("citation_stats", {}).get("verified", 0),
                "citations_fabricated": rec.get("citation_stats", {}).get("fabricated", 0),
                **coding,
                "timestamp": rec.get("timestamp", ""),
            })

    df = pd.DataFrame(rows)
    df.to_csv(config.CODED_FILE, index=False, encoding="utf-8-sig",
              quoting=csv.QUOTE_NONNUMERIC)
    log.info(f"Saved {len(df)} coded records to {config.CODED_FILE}")
    log.info("\n=== FULLY AUTOMATED CODING SUMMARY ===")
    log.info(f"Total responses: {len(df)}")
    log.info(f"Any hallucination detected: {df['any_hallucination'].sum()} "
             f"({df['any_hallucination'].mean()*100:.1f}%)")
    
    print()
    print(f"{'Category':<35} {'Total':>7} {'%':>6}")
    print("-" * 50)
    for h, label in [
        ("H1_Fabricated_Citations",     "H1 Fabricated Citations"),
        ("H2_Ghost_Appliances",         "H2 Ghost Appliances"),
        ("H3_Phantom_Protocols",        "H3 Phantom Protocols"),
        ("H4_Misattributed_Discoveries","H4 Misattributed Discoveries"),
        ("H5_Anachronistic_Knowledge",  "H5 Anachronistic Knowledge"),
        ("H6_Pseudo_Quantitative",      "H6 Pseudo-quantitative"),
        ("H7_Metacognitive_Failure",    "H7 Metacognitive Failure"),
    ]:
        total = df[h].sum()
        pct = df[h].mean() * 100
        print(f"{label:<35} {total:>7} {pct:>5.1f}%")
    
    print()
    print("=== Per-model H2-H4 distribution ===")
    print(df.groupby('model_name')[['H2_Ghost_Appliances', 'H3_Phantom_Protocols', 'H4_Misattributed_Discoveries']].sum().to_string())


if __name__ == "__main__":
    run()
