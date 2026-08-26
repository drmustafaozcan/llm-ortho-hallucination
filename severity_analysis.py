"""
severity_analysis.py

Severity stratification of LLM hallucinations in orthodontics.

Takes the response-level coded output produced by coder_v3.py and assigns each
response to a severity tier defined by the action a clinician acting on the
content would take. Computes tier prevalences with Wilson intervals, tests
between-model and between-domain differences, evaluates epistemic hedging as a
predictor of clinical consequence, and recomputes overall prevalence under a
strict citation criterion.

Severity tiers
--------------
Tier 1  clinical      H3 (phantom protocols), H5 (anachronistic knowledge)
                      Could alter a treatment decision.
Tier 2  operational   H2 (ghost appliances)
                      Could cause futile procurement or workflow cost.
Tier 3  epistemic     H1 (fabricated citations), H4 (misattributed discoveries)
                      Impedes verification without altering the recommendation.

H6 (pseudo-quantitative) is not a tier. Automated coding cannot distinguish an
incorrect numeric value from a correct but unsourced one, so H6 responses are
adjudicated by two orthodontists; those judged incorrect are promoted to Tier 1.
H7 (absence of epistemic hedging) is a modifier, analysed as a predictor.

Usage
-----
    python3 severity_analysis.py
    python3 severity_analysis.py --adjudication results/h6_adjudication.csv

The adjudication file, when supplied, must contain columns:
    query_id, model_name, verdict     with verdict in {A, B, C}
    A = numeric value incorrect
    B = value acceptable but unsourced
    C = indeterminate

Only verdict A promotes a response to Tier 1.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.proportion import proportion_confint

try:
    import config
    DEFAULT_CODED = config.CODED_FILE
    RESULTS_DIR = config.RESULTS_DIR
except Exception:
    DEFAULT_CODED = Path("results/coded_hallucinations.csv")
    RESULTS_DIR = Path("results")

MODEL_ORDER = ["Claude Opus", "Gemini Pro", "DeepSeek-V3", "GPT-4o", "Llama 3.3"]

TIERS = {
    "T1_clinical":    ["H3_Phantom_Protocols", "H5_Anachronistic_Knowledge"],
    "T2_operational": ["H2_Ghost_Appliances"],
    "T3_epistemic":   ["H1_Fabricated_Citations", "H4_Misattributed_Discoveries"],
}


def rule(title, char="="):
    print("\n" + char * 74)
    print(title)
    print(char * 74)


def wilson(k, n):
    """Wilson interval; Jeffreys when k == n, where Wilson is degenerate."""
    method = "jeffreys" if k == n else "wilson"
    lo, hi = proportion_confint(k, n, method=method)
    return lo * 100, hi * 100


def assign_tiers(df):
    for tier, cats in TIERS.items():
        present = [c for c in cats if c in df.columns]
        if not present:
            raise KeyError(f"none of {cats} present in coded file")
        df[tier] = df[present].max(axis=1).astype(int)
    return df


def apply_adjudication(df, path):
    """Promote H6 responses judged numerically incorrect (verdict A) to Tier 1."""
    adj = pd.read_csv(path)
    need = {"query_id", "model_name", "verdict"}
    if not need.issubset(adj.columns):
        sys.exit(f"adjudication file must contain {sorted(need)}")
    adj["verdict"] = adj["verdict"].astype(str).str.strip().str.upper()

    incorrect = {
        (r.query_id, r.model_name)
        for r in adj.itertuples()
        if r.verdict == "A"
    }
    promoted = [
        (q, m) in incorrect for q, m in zip(df.query_id, df.model_name)
    ]
    df["T1_clinical"] = (df["T1_clinical"].astype(bool) | pd.Series(promoted, index=df.index)).astype(int)

    counts = adj.verdict.value_counts()
    rule("H6 ADJUDICATION")
    labels = {"A": "incorrect value", "B": "acceptable but unsourced", "C": "indeterminate"}
    for v in ["A", "B", "C"]:
        n = int(counts.get(v, 0))
        print(f"  {v} — {labels[v]:<26} {n:>3} / {len(adj)}  ({n / len(adj) * 100:>4.1f}%)")
    n_a, n_b = int(counts.get("A", 0)), int(counts.get("B", 0))
    if n_a:
        print(f"\n  Automated flag over-identified error by roughly {n_b / n_a:.0f}:1.")
    print(f"  Responses promoted to Tier 1: {sum(promoted)}")
    return df


def tier_summary(df):
    rule("SEVERITY DISTRIBUTION")
    n = len(df)
    rows = [
        ("Tier 1 — clinically consequential", "T1_clinical"),
        ("Tier 2 — operationally consequential", "T2_operational"),
        ("Tier 3 — epistemically consequential", "T3_epistemic"),
        ("H6 — pseudo-quantitative", "H6_Pseudo_Quantitative"),
        ("H7 — no hedging on fabricated premise", "H7_Metacognitive_Failure"),
        ("Any hallucination (permissive)", "any_hallucination"),
    ]
    print(f"  {'':<40}{'n':>6}{'%':>8}{'95% CI':>16}")
    for label, col in rows:
        if col not in df.columns:
            continue
        k = int(df[col].sum())
        lo, hi = wilson(k, n)
        print(f"  {label:<40}{k:>6}{k / n * 100:>7.1f}%{f'{lo:.1f}-{hi:.1f}':>16}")

    # mutually exclusive highest tier
    t1 = df.T1_clinical == 1
    t2 = (df.T2_operational == 1) & ~t1
    t3 = (df.T3_epistemic == 1) & ~t1 & ~t2
    none = df.any_hallucination == 0
    other = ~(t1 | t2 | t3 | none)
    print("\n  Highest applicable tier (mutually exclusive):")
    for label, mask in [("Tier 1", t1), ("Tier 2", t2), ("Tier 3", t3),
                        ("H6 only", other), ("no hallucination", none)]:
        print(f"    {label:<20}{int(mask.sum()):>5}{mask.mean() * 100:>7.1f}%")


def strict_citations(df):
    """Recompute prevalence counting only confirmed fabrications for H1."""
    if "citations_fabricated" not in df.columns:
        return
    rule("CITATION CRITERION SENSITIVITY")
    df["H1_strict"] = (df.citations_fabricated > 0).astype(int)
    other = ["H2_Ghost_Appliances", "H3_Phantom_Protocols", "H4_Misattributed_Discoveries",
             "H5_Anachronistic_Knowledge", "H6_Pseudo_Quantitative", "H7_Metacognitive_Failure"]
    df["any_strict"] = (df[["H1_strict"] + other].max(axis=1)).astype(int)

    n = len(df)
    for label, col in [("permissive (total - verified)", "any_hallucination"),
                       ("strict (confirmed fabrications only)", "any_strict")]:
        k = int(df[col].sum())
        lo, hi = wilson(k, n)
        print(f"  {label:<40}{k:>5}/{n}  {k / n * 100:>5.1f}%  ({lo:.1f}-{hi:.1f})")

    inflated = int((df.H1_Fabricated_Citations - df.H1_strict).sum())
    print(f"\n  Responses flagged H1 on indeterminate verifications alone: {inflated}")


def by_model(df):
    rule("BY MODEL")
    n_per = df.groupby("model_name").size()
    order = [m for m in MODEL_ORDER if m in n_per.index] or sorted(n_per.index)

    print(f"  {'Model':<15}{'n':>5}{'Any %':>9}{'Tier1 %':>10}{'Tier1 95% CI':>18}")
    for m in order:
        s = df[df.model_name == m]
        k = int(s.T1_clinical.sum())
        lo, hi = wilson(k, len(s))
        print(f"  {m:<15}{len(s):>5}{s.any_hallucination.mean() * 100:>8.0f}%"
              f"{k / len(s) * 100:>9.0f}%{f'{lo:.1f}-{hi:.1f}':>18}")

    for label, col in [("Any hallucination", "any_hallucination"),
                       ("Tier 1 (clinical)", "T1_clinical")]:
        ct = pd.crosstab(df.model_name, df[col]).reindex(order)
        chi2, p, dof, _ = stats.chi2_contingency(ct)
        v = np.sqrt(chi2 / (len(df) * (min(ct.shape) - 1)))
        spread = df.groupby("model_name")[col].mean()
        verdict = "significant" if p < 0.05 else "NOT significant"
        print(f"\n  {label} x model: chi2 = {chi2:.2f}, df = {dof}, p = {p:.3g}, "
              f"V = {v:.3f} ({verdict})")
        print(f"    range {spread.min() * 100:.0f}-{spread.max() * 100:.0f}% "
              f"({(spread.max() - spread.min()) * 100:.0f} points)")


def by_domain(df):
    rule("BY QUERY DOMAIN")
    g = df.groupby("category").agg(n=("T1_clinical", "size"),
                                   anyh=("any_hallucination", "sum"),
                                   t1=("T1_clinical", "sum"))
    total_t1 = int(df.T1_clinical.sum())
    print(f"  {'Domain':<28}{'n':>5}{'Any %':>9}{'Tier1 %':>10}{'share of Tier1':>16}")
    for c, r in g.iterrows():
        share = r.t1 / total_t1 * 100 if total_t1 else 0
        print(f"  {c:<28}{int(r.n):>5}{r.anyh / r.n * 100:>8.0f}%"
              f"{r.t1 / r.n * 100:>9.0f}%{share:>15.1f}%")

    ct = pd.crosstab(df.category, df.T1_clinical)
    chi2, p, dof, _ = stats.chi2_contingency(ct)
    v = np.sqrt(chi2 / (len(df) * (min(ct.shape) - 1)))
    print(f"\n  Tier 1 x domain: chi2 = {chi2:.2f}, df = {dof}, p = {p:.3g}, V = {v:.3f}")


def hedging(df):
    """Epistemic hedging as a predictor of clinical consequence, on trap items."""
    if "trap_type" not in df.columns or "hedge_count" not in df.columns:
        return
    rule("EPISTEMIC HEDGING")
    tr = df[df.trap_type.fillna("").astype(str).str.strip().ne("")]
    if tr.empty:
        print("  no trap-flagged responses found")
        return

    print(f"  Trap-flagged responses: {len(tr)} "
          f"({tr.query_id.nunique()} unique items)")
    order = [m for m in MODEL_ORDER if m in set(tr.model_name)] or sorted(set(tr.model_name))
    print(f"\n  {'Model':<15}{'hedged':>10}{'%':>7}{'Tier1':>8}")
    for m in order:
        s = tr[tr.model_name == m]
        h = int((s.hedge_count > 0).sum())
        print(f"  {m:<15}{f'{h}/{len(s)}':>10}{h / len(s) * 100:>6.0f}%"
              f"{int(s.T1_clinical.sum()):>8}")

    ct_m = pd.crosstab(tr.model_name, tr.hedge_count > 0).reindex(order)
    chi2, p, dof, _ = stats.chi2_contingency(ct_m)
    print(f"\n  Hedging x model: chi2 = {chi2:.2f}, df = {dof}, p = {p:.3g}")

    tb = (pd.crosstab(tr.hedge_count > 0, tr.T1_clinical)
            .reindex(index=[False, True], columns=[0, 1], fill_value=0)
            .fillna(0).astype(int))
    a, b = int(tb.iat[0, 0]), int(tb.iat[0, 1])   # not hedged: no Tier 1, Tier 1
    c, d = int(tb.iat[1, 0]), int(tb.iat[1, 1])   # hedged:     no Tier 1, Tier 1

    hedged_rate = d / (c + d) * 100 if (c + d) else float("nan")
    unhedged_rate = b / (a + b) * 100 if (a + b) else float("nan")
    print(f"\n  Tier 1 when hedged     : {d}/{c + d} = {hedged_rate:.1f}%")
    print(f"  Tier 1 when not hedged : {b}/{a + b} = {unhedged_rate:.1f}%")

    _, p_fisher = stats.fisher_exact(tb)
    if min(a, b, c, d) == 0:
        # Haldane-Anscombe: OR is otherwise undefined with a zero cell
        or_val = ((b + 0.5) * (c + 0.5)) / ((a + 0.5) * (d + 0.5))
        print(f"  OR = {or_val:.1f} (Haldane-Anscombe corrected; zero cell present)")
        print("  Separation is complete in this sample; interpret with caution.")
    else:
        or_val = (b * c) / (a * d)
        se = np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
        print(f"  OR = {or_val:.1f} (95% CI {np.exp(np.log(or_val) - 1.96 * se):.1f}"
              f"-{np.exp(np.log(or_val) + 1.96 * se):.1f})")
    print(f"  Fisher exact p = {p_fisher:.3g}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--coded", default=str(DEFAULT_CODED),
                    help="response-level coded CSV from coder_v3.py")
    ap.add_argument("--adjudication", default=None,
                    help="optional H6 adjudication CSV (query_id, model_name, verdict)")
    ap.add_argument("--out", default=str(Path(RESULTS_DIR) / "coded_with_severity_tiers.csv"),
                    help="where to write the tiered dataset")
    args = ap.parse_args()

    coded = Path(args.coded)
    if not coded.exists():
        sys.exit(f"coded file not found: {coded}\nRun coder_v3.py first.")

    df = pd.read_csv(coded)
    print(f"Loaded {len(df)} responses from {coded}")

    df = assign_tiers(df)
    if args.adjudication:
        df = apply_adjudication(df, args.adjudication)
    else:
        print("\nNote: no adjudication file supplied. H6 responses are reported "
              "separately and none are promoted to Tier 1.")

    tier_summary(df)
    strict_citations(df)
    by_model(df)
    by_domain(df)
    hedging(df)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nTiered dataset written to {out}")


if __name__ == "__main__":
    main()
