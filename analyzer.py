"""
analyzer.py
Statistical analysis + publication-ready figures from coded_hallucinations.csv.

Produces:
- summary_statistics.csv: aggregate tables ready to drop into the paper
- figures/fig1_hallucination_rate_by_model.png
- figures/fig2_taxonomy_distribution.png
- figures/fig3_category_heatmap.png
- figures/fig4_trap_refusal_rate.png

Stats:
- Per-model hallucination rates with 95% CI (Wilson)
- Chi-square tests for model differences
- Fleiss kappa for cross-model agreement (optional, if model count >= 3)
"""
import logging
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

import config

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger(__name__)

sns.set_style("whitegrid")
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


HALL_COLS = [
    "H1_Fabricated_Citations",
    "H2_Ghost_Appliances",
    "H3_Phantom_Protocols",
    "H4_Misattributed_Discoveries",
    "H5_Anachronistic_Knowledge",
    "H6_Pseudo_Quantitative",
    "H7_Metacognitive_Failure",
]

HALL_LABELS = [
    "H1 Fabricated\nCitations",
    "H2 Ghost\nAppliances",
    "H3 Phantom\nProtocols",
    "H4 Misattributed\nDiscoveries",
    "H5 Anachronistic\nKnowledge",
    "H6 Pseudo-\nquantitative",
    "H7 Metacognitive\nFailure",
]


def wilson_ci(successes: int, n: int, z: float = 1.96):
    """Wilson 95% CI for a proportion."""
    if n == 0:
        return (0, 0)
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2*n)) / denom
    half = (z * np.sqrt(p*(1-p)/n + z**2/(4*n**2))) / denom
    return (max(0, centre - half), min(1, centre + half))


def per_model_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Hallucination rate per model with Wilson CIs."""
    rows = []
    for model in sorted(df["model_name"].unique()):
        sub = df[df["model_name"] == model]
        n = len(sub)
        hall = sub["any_hallucination"].sum()
        rate = hall / n if n else 0
        lo, hi = wilson_ci(hall, n)
        rows.append({
            "Model": model,
            "N": n,
            "Hallucinations": hall,
            "Rate_%": round(rate*100, 1),
            "CI95_lower_%": round(lo*100, 1),
            "CI95_upper_%": round(hi*100, 1),
        })
    return pd.DataFrame(rows)


def per_category_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Hallucination rate per model x category."""
    table = (df.groupby(["model_name", "category"])["any_hallucination"]
               .agg(["sum", "count"])
               .reset_index())
    table["rate_%"] = (table["sum"] / table["count"] * 100).round(1)
    pivot = table.pivot(index="model_name", columns="category", values="rate_%")
    return pivot


def taxonomy_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Count of each hallucination type per model."""
    rows = []
    for model in sorted(df["model_name"].unique()):
        sub = df[df["model_name"] == model]
        row = {"Model": model, "N": len(sub)}
        for col, label in zip(HALL_COLS, HALL_LABELS):
            row[label.replace("\n", " ")] = int(sub[col].sum())
        rows.append(row)
    return pd.DataFrame(rows)


def trap_refusal_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Refusal rate on trap queries per model.
    Refusal = no hallucination on trap query (modelhonestly said it doesn't know)."""
    trap = df[df["category"] == "E_Trap_Adversarial"].copy()
    if trap.empty:
        return pd.DataFrame()
    rows = []
    for model in sorted(trap["model_name"].unique()):
        sub = trap[trap["model_name"] == model]
        n = len(sub)
        # Refusal: hedge_count > 0 AND H1/H7 not flagged (rough definition)
        refused = ((sub["hedge_count"] > 0) &
                   (sub["H7_Metacognitive_Failure"] == 0)).sum()
        rate = refused / n if n else 0
        lo, hi = wilson_ci(refused, n)
        rows.append({
            "Model": model,
            "N_trap_queries": n,
            "Refused (hedged)": refused,
            "Refusal_rate_%": round(rate*100, 1),
            "CI95_lower_%": round(lo*100, 1),
            "CI95_upper_%": round(hi*100, 1),
        })
    return pd.DataFrame(rows)


def chi_square_models(df: pd.DataFrame) -> dict:
    """Chi-square test: do models differ in overall hallucination rate?"""
    contingency = pd.crosstab(df["model_name"], df["any_hallucination"])
    if contingency.shape[0] < 2 or contingency.shape[1] < 2:
        return {"chi2": None, "p": None, "dof": None,
                "note": "Insufficient variation"}
    chi2, p, dof, _ = stats.chi2_contingency(contingency)
    return {"chi2": round(chi2, 3), "p": round(p, 4), "dof": dof}


# ============================================================
# FIGURES
# ============================================================

def fig_hallucination_rate(df: pd.DataFrame):
    """Bar chart of hallucination rate per model with 95% CI."""
    summary = per_model_summary(df)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = range(len(summary))
    ax.bar(x, summary["Rate_%"], yerr=[
        summary["Rate_%"] - summary["CI95_lower_%"],
        summary["CI95_upper_%"] - summary["Rate_%"]
    ], capsize=4, color="#4A7AB7", edgecolor="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(summary["Model"], rotation=20, ha="right")
    ax.set_ylabel("Hallucination rate (%)")
    ax.set_title("Overall hallucination rate per LLM (95% Wilson CI)")
    ax.set_ylim(0, max(100, summary["CI95_upper_%"].max() + 5))
    for i, v in enumerate(summary["Rate_%"]):
        ax.text(i, v + 1.5, f"{v}%", ha="center", fontsize=9)
    plt.savefig(config.FIGURES_DIR / "fig1_hallucination_rate_by_model.png")
    plt.close()


def fig_taxonomy_distribution(df: pd.DataFrame):
    """Stacked bar of taxonomy per model."""
    tax = taxonomy_distribution(df).set_index("Model").drop(columns=["N"])
    fig, ax = plt.subplots(figsize=(8, 5))
    tax.plot(kind="bar", stacked=True, ax=ax, edgecolor="white", linewidth=0.5,
             colormap="tab10")
    ax.set_ylabel("Number of hallucination instances")
    ax.set_xlabel("")
    ax.set_title("Hallucination taxonomy distribution per LLM")
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5),
              frameon=False, fontsize=8)
    plt.xticks(rotation=20, ha="right")
    plt.savefig(config.FIGURES_DIR / "fig2_taxonomy_distribution.png")
    plt.close()


def fig_category_heatmap(df: pd.DataFrame):
    """Heatmap of hallucination rate by model x query category."""
    pivot = per_category_summary(df)
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.heatmap(pivot, annot=True, fmt=".0f", cmap="Reds",
                vmin=0, vmax=100, ax=ax,
                cbar_kws={"label": "Hallucination rate (%)"})
    ax.set_title("Hallucination rate (%) by model and query category")
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.savefig(config.FIGURES_DIR / "fig3_category_heatmap.png")
    plt.close()


def fig_trap_refusal(df: pd.DataFrame):
    """Refusal rate on trap queries per model."""
    summary = trap_refusal_analysis(df)
    if summary.empty:
        log.warning("No trap queries found; skipping fig4")
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = range(len(summary))
    ax.bar(x, summary["Refusal_rate_%"], yerr=[
        summary["Refusal_rate_%"] - summary["CI95_lower_%"],
        summary["CI95_upper_%"] - summary["Refusal_rate_%"]
    ], capsize=4, color="#5B9F5B", edgecolor="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(summary["Model"], rotation=20, ha="right")
    ax.set_ylabel("Refusal rate on trap queries (%)")
    ax.set_title("Epistemic honesty: refusal rate on adversarial trap queries")
    ax.set_ylim(0, 100)
    for i, v in enumerate(summary["Refusal_rate_%"]):
        ax.text(i, v + 1.5, f"{v}%", ha="center", fontsize=9)
    plt.savefig(config.FIGURES_DIR / "fig4_trap_refusal_rate.png")
    plt.close()


# ============================================================
# MAIN
# ============================================================

def run():
    if not config.CODED_FILE.exists():
        log.error(f"Coded file not found: {config.CODED_FILE}")
        log.error("Run coder.py first.")
        return

    df = pd.read_csv(config.CODED_FILE)
    log.info(f"Loaded {len(df)} coded records, "
             f"{df['model_name'].nunique()} models, "
             f"{df['query_id'].nunique()} queries")

    # Summary tables
    log.info("\n=== TABLE 1: Per-model hallucination rates ===")
    t1 = per_model_summary(df)
    print(t1.to_string(index=False))

    log.info("\n=== TABLE 2: Taxonomy distribution per model ===")
    t2 = taxonomy_distribution(df)
    print(t2.to_string(index=False))

    log.info("\n=== TABLE 3: Hallucination rate by model x category (%) ===")
    t3 = per_category_summary(df)
    print(t3.to_string())

    log.info("\n=== TABLE 4: Refusal rate on trap queries ===")
    t4 = trap_refusal_analysis(df)
    print(t4.to_string(index=False))

    log.info("\n=== STATISTICAL TEST: Chi-square (model differences) ===")
    chi = chi_square_models(df)
    print(chi)

    # Save all tables to one xlsx
    with pd.ExcelWriter(config.RESULTS_DIR / "summary_tables.xlsx") as writer:
        t1.to_excel(writer, sheet_name="T1_model_rates", index=False)
        t2.to_excel(writer, sheet_name="T2_taxonomy", index=False)
        t3.to_excel(writer, sheet_name="T3_category_heatmap")
        t4.to_excel(writer, sheet_name="T4_trap_refusal", index=False)
        pd.DataFrame([chi]).to_excel(writer, sheet_name="T5_chi_square", index=False)

    # Figures
    log.info("\nGenerating figures...")
    fig_hallucination_rate(df)
    fig_taxonomy_distribution(df)
    fig_category_heatmap(df)
    fig_trap_refusal(df)
    log.info(f"Figures saved to {config.FIGURES_DIR}")
    log.info("Done.")


if __name__ == "__main__":
    run()
