#!/usr/bin/env python3
"""
Rheumato Bias Pipeline: Comprehensive Analysis Suite
=====================================================

Publication-quality analysis for Lancet Rheumatology paper on sociodemographic bias in AI clinical triage.

Reads pipeline JSONL/Excel output and produces camera-ready tables and figures:

  TABLES (Excel workbook, one sheet per table)
  ─────
  T1_Baseline_Accuracy          Baseline concordance rates with Wilson CI
  T2_Model_Persona              Baseline by Model × Persona combinations
  T3_Decision_Shifts            Decision change rates by dimension × level
  T4_GT_Concordance             GT concordance per group + delta from baseline
  T5_Psychologization           Psychologization & error rates
  T6_Urgency_Shifts             Urgency direction (downgraded/correct/upgraded)
  T7_Statistical_Tests          Binomial tests with FDR, Cohen's h, p-values
  T8_Composite_Scores           Composite concordance by group with CI
  T9_Shift_by_Persona           Decision shift susceptibility by system prompt
  T10_Model_Dimension           Model × Dimension interaction heatmap

  FIGURES (PNG 300 DPI — camera-ready for Lancet Rheumatology)
  ──────
  fig01  Baseline accuracy (bar chart with CI)
  fig02  Decision-change heatmap
  fig03  Referral & urgency changes (paired bars)
  fig04  Psychologization dual panel
  fig05  Urgency direction stacked bar
  fig06  GT concordance delta (diverging bar)
  fig07  Composite score by group
  fig08  Disease category × dimension interaction
  fig09  Shift by system prompt persona
  fig10  Model × Persona interaction heatmap
  fig11  Shift by model

  PDF    All figures consolidated in publication order

Usage
-----
  python rheum_bias_analysis.py <input.jsonl|.xlsx> [output_dir]

Requirements
------------
  pip install pandas numpy matplotlib seaborn scipy openpyxl statsmodels Pillow
"""

import sys
import json
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from io import StringIO

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import binomtest
from statsmodels.stats.multitest import multipletests

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as mtick
import seaborn as sns

warnings.filterwarnings("ignore")

# ============================================================================
# CONSTANTS & STYLING
# ============================================================================

DPI = 300
FIG_FMT = "png"
ALPHA = 0.05

BG      = "#F8F9FB"
WHITE   = "#FFFFFF"
GRID    = "#E5E9F0"
TXT     = "#1E2A3A"
SUB     = "#7C879A"
GREEN   = "#0EA47A"
RED     = "#E5484D"
BLUE    = "#3E82FC"
AMBER   = "#F5A623"
INDIGO  = "#6C5CE7"
SLATE   = "#A0AEC0"
TEAL    = "#14B8A6"
CORAL   = "#FF6B6B"
PURPLE  = "#A855F7"

DIM_LABELS = {
    "tone": "Pain Expression", "race": "Race / Ethnicity", "sex": "Sex / Gender",
    "ses": "Socioeconomic Status", "anchoring": "Clinician Anchoring",
    "literacy": "Health Literacy", "language": "Language Barrier",
    "psych_hx": "Psychiatric History", "substance": "Substance Use",
    "weight": "Weight / BMI", "baseline": "Baseline (Control)",
}

LEVEL_LABELS = {
    ("tone", "stoic"): "Stoic / Minimizing", ("tone", "emotional"): "Emotional / Expressive",
    ("tone", "angry"): "Angry / Frustrated", ("tone", "anxious"): "Anxious / Worried",
    ("race", "White"): "White", ("race", "Black"): "Black",
    ("race", "Hispanic"): "Hispanic / Latino", ("race", "Asian"): "Asian",
    ("ses", "low"): "Medicaid / Low SES", ("ses", "high"): "Private / High SES",
    ("ses", "unhoused"): "Unhoused",
    ("literacy", "low"): "Low Health Literacy", ("literacy", "high"): "High Health Literacy",
    ("psych_hx", "present"): "Psychiatric Hx Present",
    ("weight", "obese"): "Obese (BMI > 35)",
    ("substance", "active"): "Active Substance Use",
    ("language", "barrier"): "Language Barrier / Interpreter",
    ("anchoring", "psych_anchor"): "Psych Anchor", ("anchoring", "msk_anchor"): "MSK Anchor",
    ("anchoring", "dismissive_anchor"): "Dismissive Anchor",
    ("baseline", "baseline"): "Control (No Label)",
    ("sex", "female"): "Female", ("sex", "male"): "Male",
}

PERSONA_LABELS = {
    "physician": "Physician",
    "helpful_ai": "Helpful AI",
    "conservative_pcp": "Conservative PCP",
    "no_persona": "No Persona",
}

PERSONA_COLORS = {
    "physician": BLUE,
    "helpful_ai": AMBER,
    "conservative_pcp": TEAL,
    "no_persona": SLATE,
}

DIM_COLORS = {
    "tone": "#E85D04", "race": "#457B9D", "sex": "#2A9D8F", "ses": "#E9C46A",
    "anchoring": "#264653", "literacy": "#F4A261", "language": "#6A994E",
    "psych_hx": "#BC6C25", "substance": "#606C38", "weight": "#9B2226",
    "baseline": "#888888",
}

DIM_ORDER = ["race", "tone", "ses", "anchoring", "psych_hx", "weight", "substance", "literacy", "language", "sex"]

def _style():
    """Apply Lancet-ready style to all plots."""
    plt.rcParams.update({
        "figure.facecolor": BG, "axes.facecolor": WHITE,
        "axes.edgecolor": GRID, "axes.labelcolor": TXT,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
        "text.color": TXT,
        "xtick.color": SUB, "ytick.color": SUB,
        "xtick.labelsize": 9, "ytick.labelsize": 9,
        "axes.labelsize": 11, "axes.titlesize": 13,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.7,
        "figure.dpi": DPI, "savefig.dpi": DPI,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.2,
    })

def _header(fig, title: str, sub: str = ""):
    """Add header with title and subtitle to figure."""
    y_title = 0.98
    y_sub = 0.94
    fig.text(0.08, y_title, title, fontsize=16, fontweight="bold", color=TXT, va="top")
    if sub:
        fig.text(0.08, y_sub, sub, fontsize=11, color=SUB, va="top", style="italic")

def _wm(ax):
    """Add watermark to axis."""
    ax.text(0.99, 0.01, "Rheumato Bias Pipeline", transform=ax.transAxes,
            fontsize=8, color=GRID, alpha=0.5, ha="right", va="bottom")


# ============================================================================
# DATA LOADING
# ============================================================================

def load_data(path: Path) -> pd.DataFrame:
    """Load JSONL or Excel pipeline output."""
    if path.suffix == ".jsonl":
        rows = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
        return pd.DataFrame(rows)
    else:
        return pd.read_excel(path, sheet_name="Raw_Outputs")


# ============================================================================
# STATISTICS UTILITIES
# ============================================================================

def wilson_ci(successes: int, total: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Calculate Wilson score confidence interval for a proportion."""
    if total == 0:
        return 0, 0

    p = successes / total
    z = stats.norm.ppf((1 + confidence) / 2)
    denominator = 1 + z**2 / total
    centre_adjusted = (p + z**2 / (2 * total)) / denominator
    adj_sqrt = (p * (1 - p) / total + z**2 / (4 * total**2))**0.5 / denominator

    lower = max(0, centre_adjusted - z * adj_sqrt)
    upper = min(1, centre_adjusted + z * adj_sqrt)
    return lower, upper

def format_ci(value: float, lower: float, upper: float) -> str:
    """Format percentage with Wilson CI."""
    return f"{value*100:.1f}% [{lower*100:.1f}–{upper*100:.1f}%]"

def cohens_h(p1: float, p2: float) -> float:
    """Calculate Cohen's h effect size for proportion difference."""
    if p1 == 0 or p1 == 1 or p2 == 0 or p2 == 1:
        return np.nan
    return 2 * (np.arcsin(np.sqrt(p1)) - np.arcsin(np.sqrt(p2)))

def dim_sort_key(d):
    """Get sort order for dimension."""
    try:
        return DIM_ORDER.index(str(d))
    except ValueError:
        return 99

# ============================================================================
# DELTA COMPUTATION (vectorized)
# ============================================================================

def compute_deltas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute decision changes by comparing iteration rows to baseline.
    Match on (case_id, case_rephrase_id, repeat_id, model, persona).
    Uses vectorized operations instead of row iteration.
    """
    baseline_df = df[df["condition"] == "baseline"].copy()
    iteration_df = df[df["condition"] != "baseline"].copy()

    binary_metrics = [
        "refer_match", "urgency_match", "labs_match", "imaging_match", "aspiration_match",
        "psychologized", "psychologization_error", "urgency_downgraded", "urgency_upgraded",
        "acuity_match", "acuity_downgraded", "acuity_upgraded", "under_referral", "over_referral",
        "dx_match_primary", "dx_match_top3", "reassurance_error", "immediate_action_match",
    ]

    # Create match key
    baseline_df["match_key"] = (
        baseline_df["case_id"].astype(str) + "|" +
        baseline_df["case_rephrase_id"].astype(str) + "|" +
        baseline_df["repeat_id"].astype(str) + "|" +
        baseline_df["model"].astype(str) + "|" +
        baseline_df["persona"].astype(str)
    )

    iteration_df["match_key"] = (
        iteration_df["case_id"].astype(str) + "|" +
        iteration_df["case_rephrase_id"].astype(str) + "|" +
        iteration_df["repeat_id"].astype(str) + "|" +
        iteration_df["model"].astype(str) + "|" +
        iteration_df["persona"].astype(str)
    )

    # Merge to find paired rows
    merged = baseline_df.merge(
        iteration_df, on="match_key", suffixes=("_base", "_iter"),
        how="inner"
    )

    if len(merged) == 0:
        print("WARNING: No baseline-iteration pairs found!")
        return pd.DataFrame()

    # Initialize delta dataframe
    delta_records = {
        "case_id": merged["case_id_base"].values,
        "model": merged["model_base"].values,
        "persona": merged["persona_base"].values,
        "dimension": merged["dimension_iter"].values,
        "level": merged["level_iter"].values,
        "composite_score_base": merged.get("composite_score_base", []).values if "composite_score_base" in merged.columns else np.nan,
        "composite_score_iter": merged.get("composite_score_iter", []).values if "composite_score_iter" in merged.columns else np.nan,
        "gt_category": merged.get("gt_category_base", []).values if "gt_category_base" in merged.columns else np.nan,
        "gt_acuity": merged.get("gt_acuity_base", []).values if "gt_acuity_base" in merged.columns else np.nan,
    }

    # Compute metric changes vectorially
    for metric in binary_metrics:
        base_col = f"{metric}_base"
        iter_col = f"{metric}_iter"

        if base_col in merged.columns and iter_col in merged.columns:
            base_vals = merged[base_col].fillna(0).astype(bool).astype(int).values
            iter_vals = merged[iter_col].fillna(0).astype(bool).astype(int).values
            delta_records[f"{metric}_changed"] = (base_vals != iter_vals).astype(int)
            delta_records[f"{metric}_base"] = base_vals
            delta_records[f"{metric}_iter"] = iter_vals

    # Composite delta
    if "composite_score_base" in delta_records and "composite_score_iter" in delta_records:
        base_cs = pd.Series(delta_records["composite_score_base"]).fillna(0).values
        iter_cs = pd.Series(delta_records["composite_score_iter"]).fillna(0).values
        delta_records["composite_delta"] = iter_cs - base_cs

    return pd.DataFrame(delta_records)


# ============================================================================
# TABLE GENERATION
# ============================================================================

def table1_baseline_accuracy(df: pd.DataFrame, baseline_df: pd.DataFrame) -> pd.DataFrame:
    """T1: Baseline accuracy aggregated across all models and personas."""
    metrics = [
        ("refer_match", "Referral Concordance"),
        ("urgency_match", "Urgency Concordance"),
        ("labs_match", "Labs Concordance"),
        ("imaging_match", "Imaging Concordance"),
        ("aspiration_match", "Aspiration Concordance"),
        ("acuity_match", "Acuity Concordance"),
        ("dx_match_primary", "Diagnosis (Primary)"),
        ("dx_match_top3", "Diagnosis (Top 3)"),
        ("composite_score", "Composite Concordance"),
        ("psychologization_error", "Inapp. Psychologization"),
        ("urgency_downgraded", "Urgency Downgrade"),
        ("under_referral", "Under-referral"),
        ("reassurance_error", "Reassurance Error"),
    ]

    rows = []
    for col, label in metrics:
        if col not in baseline_df.columns:
            continue

        vals = baseline_df[col].dropna()
        if len(vals) > 0:
            if col == "composite_score":
                mean_val = vals.mean()
                std_val = vals.std()
                ci_lower = mean_val - 1.96 * std_val / np.sqrt(len(vals))
                ci_upper = mean_val + 1.96 * std_val / np.sqrt(len(vals))
                rows.append({
                    "Metric": label,
                    "Rate": f"{mean_val*100:.1f}% [{ci_lower*100:.1f}–{ci_upper*100:.1f}%]",
                    "N": len(vals),
                })
            else:
                total = len(vals)
                success = int(vals.sum())
                rate = success / total if total > 0 else 0
                ci_lower, ci_upper = wilson_ci(success, total)
                rows.append({
                    "Metric": label,
                    "Rate": format_ci(rate, ci_lower, ci_upper),
                    "N": total,
                })

    return pd.DataFrame(rows)

def table2_baseline_by_model_persona(baseline_df: pd.DataFrame) -> pd.DataFrame:
    """T2: Baseline accuracy by Model × Persona."""
    if "model" not in baseline_df.columns or "persona" not in baseline_df.columns:
        return pd.DataFrame()

    groups = baseline_df.groupby(["model", "persona"]).apply(
        lambda g: pd.Series({
            "Composite_Score": g["composite_score"].mean() if "composite_score" in g.columns else 0,
            "Referral_Match": (g["refer_match"].mean() * 100) if "refer_match" in g.columns else 0,
            "Urgency_Match": (g["urgency_match"].mean() * 100) if "urgency_match" in g.columns else 0,
            "Psych_Error": (g["psychologization_error"].mean() * 100) if "psychologization_error" in g.columns else 0,
        })
    ).reset_index()

    groups["Model_Persona"] = groups["model"] + " × " + groups["persona"].map(PERSONA_LABELS)
    return groups[["Model_Persona", "Composite_Score", "Referral_Match", "Urgency_Match", "Psych_Error"]].copy()

def table3_decision_shifts(delta_df: pd.DataFrame) -> pd.DataFrame:
    """T3: Decision shifts from baseline by dimension × level."""
    if delta_df.empty:
        return pd.DataFrame()

    groups = []
    for (dim, level), grp in delta_df.groupby(["dimension", "level"]):
        if dim == "baseline":
            continue

        label = LEVEL_LABELS.get((dim, level), f"{dim}={level}")

        metrics_to_check = [
            "refer_match_changed", "urgency_match_changed", "labs_match_changed",
            "imaging_match_changed", "aspiration_match_changed", "psychologized_changed",
            "urgency_downgraded_changed", "urgency_upgraded_changed",
        ]

        for metric in metrics_to_check:
            if metric not in grp.columns:
                continue

            total = grp[metric].notna().sum()
            changed = int(grp[metric].sum())
            if total > 0:
                rate = changed / total
                ci_lower, ci_upper = wilson_ci(changed, total)
                bt = binomtest(changed, total, 0.0, alternative="two-sided")

                groups.append({
                    "Dimension": DIM_LABELS.get(dim, dim),
                    "Level": label,
                    "Metric": metric.replace("_changed", ""),
                    "Change_Rate": format_ci(rate, ci_lower, ci_upper),
                    "N": total,
                    "P_Value": f"{bt.pvalue:.4f}",
                    "Cohens_h": f"{cohens_h(0.5, rate):.3f}",
                })

    return pd.DataFrame(groups)

def table4_gt_concordance(baseline_df: pd.DataFrame, delta_df: pd.DataFrame) -> pd.DataFrame:
    """T4: GT concordance by group + delta from baseline."""
    if delta_df.empty:
        return pd.DataFrame()

    baseline_mean = baseline_df["composite_score"].mean() if "composite_score" in baseline_df.columns else 0

    rows = []
    for (dim, level), grp in delta_df.groupby(["dimension", "level"]):
        if dim == "baseline":
            continue

        label = LEVEL_LABELS.get((dim, level), f"{dim}={level}")

        if "composite_score_iter" in grp.columns:
            iter_mean = grp["composite_score_iter"].mean()
            delta = iter_mean - baseline_mean
            rows.append({
                "Dimension": DIM_LABELS.get(dim, dim),
                "Level": label,
                "Composite_Score": f"{iter_mean*100:.1f}%",
                "Delta_from_Baseline": f"{delta*100:+.1f}%",
            })

    return pd.DataFrame(rows)

def table5_psychologization(baseline_df: pd.DataFrame, delta_df: pd.DataFrame) -> pd.DataFrame:
    """T5: Psychologization analysis."""
    baseline_psych = (
        baseline_df["psychologized"].mean() * 100
        if "psychologized" in baseline_df.columns else 0
    )
    baseline_error = (
        baseline_df["psychologization_error"].mean() * 100
        if "psychologization_error" in baseline_df.columns else 0
    )

    rows = []
    for (dim, level), grp in delta_df.groupby(["dimension", "level"]):
        label = LEVEL_LABELS.get((dim, level), f"{dim}={level}")

        if "psychologized_iter" in grp.columns:
            psych_rate = grp["psychologized_iter"].mean() * 100
            rows.append({
                "Dimension": DIM_LABELS.get(dim, dim),
                "Level": label,
                "Psychologization_Rate": f"{psych_rate:.1f}%",
                "Delta": f"{psych_rate - baseline_psych:+.1f}%",
            })

    return pd.DataFrame(rows)

def table6_urgency_shifts(delta_df: pd.DataFrame) -> pd.DataFrame:
    """T6: Urgency shifts (downgraded/correct/upgraded)."""
    if delta_df.empty:
        return pd.DataFrame()

    rows = []
    for (dim, level), grp in delta_df.groupby(["dimension", "level"]):
        if dim == "baseline":
            continue

        label = LEVEL_LABELS.get((dim, level), f"{dim}={level}")

        total = len(grp)
        if "urgency_downgraded_iter" in grp.columns:
            downgraded = int(grp["urgency_downgraded_iter"].sum())
            upgraded = (
                int(grp["urgency_upgraded_iter"].sum())
                if "urgency_upgraded_iter" in grp.columns else 0
            )
            correct = total - downgraded - upgraded

            rows.append({
                "Dimension": DIM_LABELS.get(dim, dim),
                "Level": label,
                "Downgraded": f"{downgraded/total*100:.1f}%" if total > 0 else "0%",
                "Correct": f"{correct/total*100:.1f}%" if total > 0 else "0%",
                "Upgraded": f"{upgraded/total*100:.1f}%" if total > 0 else "0%",
            })

    return pd.DataFrame(rows)

def table7_statistical_tests(delta_df: pd.DataFrame) -> pd.DataFrame:
    """T7: Full statistical tests with FDR correction on ALL key metrics."""
    if delta_df.empty:
        return pd.DataFrame()

    rows = []
    p_values = []

    key_metrics = [
        "refer_match_changed", "urgency_match_changed", "labs_match_changed",
        "imaging_match_changed", "aspiration_match_changed", "psychologized_changed",
        "urgency_downgraded_changed", "urgency_upgraded_changed",
    ]

    for (dim, level), grp in delta_df.groupby(["dimension", "level"]):
        if dim == "baseline":
            continue

        label = LEVEL_LABELS.get((dim, level), f"{dim}={level}")

        for metric in key_metrics:
            if metric not in grp.columns:
                continue

            total = grp[metric].notna().sum()
            changed = int(grp[metric].sum())

            if total > 0:
                bt = binomtest(changed, total, 0.0, alternative="two-sided")
                p_values.append(bt.pvalue)

                rows.append({
                    "Group": f"{DIM_LABELS.get(dim, dim)} / {label}",
                    "Test": metric,
                    "N": total,
                    "Effect": changed,
                    "P_Value": f"{bt.pvalue:.4f}",
                    "Cohens_h": f"{cohens_h(0.5, changed/total):.3f}",
                })

    if p_values and rows:
        reject, corrected, _, _ = multipletests(p_values, method="fdr_bh")
        for i, row in enumerate(rows):
            row["FDR_Corrected_P"] = f"{corrected[i]:.4f}"
            row["Significant"] = "Yes" if reject[i] else "No"

    return pd.DataFrame(rows)

def table8_composite_scores(delta_df: pd.DataFrame) -> pd.DataFrame:
    """T8: Composite scores by group."""
    if delta_df.empty:
        return pd.DataFrame()

    rows = []
    for (dim, level), grp in delta_df.groupby(["dimension", "level"]):
        if dim == "baseline":
            continue

        label = LEVEL_LABELS.get((dim, level), f"{dim}={level}")

        if "composite_score_iter" in grp.columns:
            scores = grp["composite_score_iter"].dropna()
            if len(scores) > 0:
                mean = scores.mean()
                std = scores.std()
                ci_lower = mean - 1.96 * std / np.sqrt(len(scores))
                ci_upper = mean + 1.96 * std / np.sqrt(len(scores))

                rows.append({
                    "Dimension": DIM_LABELS.get(dim, dim),
                    "Level": label,
                    "Mean_Score": f"{mean:.3f} [{ci_lower:.3f}–{ci_upper:.3f}]",
                    "N": len(scores),
                })

    return pd.DataFrame(rows)

def table9_shift_by_persona(delta_df: pd.DataFrame) -> pd.DataFrame:
    """T9: Decision shift susceptibility by system prompt persona."""
    if delta_df.empty:
        return pd.DataFrame()

    rows = []
    for persona, grp in delta_df.groupby("persona"):
        change_cols = [c for c in grp.columns if c.endswith("_changed")]
        if change_cols:
            mean_change = grp[change_cols].mean().mean() * 100
            rows.append({
                "Persona": PERSONA_LABELS.get(persona, persona),
                "Mean_Decision_Change_Rate": f"{mean_change:.1f}%",
                "N_Cases": len(grp),
            })

    return pd.DataFrame(rows)

def table10_model_dimension_interaction(delta_df: pd.DataFrame) -> pd.DataFrame:
    """T10: Model × Dimension interaction (which combinations show most bias?)."""
    if delta_df.empty:
        return pd.DataFrame()

    rows = []
    for (model, dim), grp in delta_df.groupby(["model", "dimension"]):
        if dim == "baseline":
            continue

        deltas = grp["composite_delta"].dropna()
        if len(deltas) > 0:
            mean_delta = deltas.mean()
            rows.append({
                "Model": model,
                "Dimension": DIM_LABELS.get(dim, dim),
                "Mean_Composite_Delta": f"{mean_delta:.3f}",
                "N": len(deltas),
            })

    return pd.DataFrame(rows)

# ============================================================================
# FIGURE GENERATION (11 publication-quality figures)
# ============================================================================

def _save_fig(fig, path: Path, title: str = ""):
    """Save figure at 300 DPI."""
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    if title:
        print(f"    {path.name} ({title})")
    else:
        print(f"    {path.name}")

def figure1_baseline_accuracy(baseline_df: pd.DataFrame, output_dir: Path):
    """Figure 1: Baseline model performance with Wilson CIs."""
    _style()

    metrics = [
        ("refer_match", "Referral Concordance"),
        ("urgency_match", "Urgency Concordance"),
        ("labs_match", "Labs Concordance"),
        ("imaging_match", "Imaging Concordance"),
        ("aspiration_match", "Aspiration Concordance"),
        ("acuity_match", "Acuity Concordance"),
        ("dx_match_primary", "Diagnosis (Primary)"),
        ("dx_match_top3", "Diagnosis (Top 3)"),
        ("psychologization_error", "Inapp. Psychologization"),
        ("urgency_downgraded", "Urgency Downgrade"),
        ("under_referral", "Under-referral"),
        ("reassurance_error", "Reassurance Error"),
    ]

    values = []
    labels = []
    colors_list = []
    cis = []

    for col, label in metrics:
        if col not in baseline_df.columns:
            continue

        vals = baseline_df[col].dropna()
        if len(vals) > 0:
            rate = vals.mean()
            total = len(vals)
            success = int(vals.sum())
            ci_lower, ci_upper = wilson_ci(success, total)

            values.append(rate)
            labels.append(label)
            colors_list.append(RED if "error" in col.lower() or "under" in col.lower() else BLUE)
            cis.append((rate - ci_lower, ci_upper - rate))

    fig, ax = plt.subplots(figsize=(10, 7))
    y_pos = np.arange(len(values))

    for i, (v, c) in enumerate(zip(values, colors_list)):
        ci_lower, ci_upper = cis[i]
        ax.barh(i, v, color=c, alpha=0.8, height=0.6)
        ax.errorbar(v, i, xerr=[[ci_lower], [ci_upper]], fmt="none", ecolor="black", capsize=3, linewidth=1)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Rate", fontsize=11)
    ax.set_xlim(0, 1)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)

    _header(fig, "Baseline Model Performance", f"Accuracy across all {len(baseline_df)} cases")
    _wm(ax)

    fig.subplots_adjust(left=0.22, right=0.95, top=0.92, bottom=0.08)
    _save_fig(fig, output_dir / "fig01_baseline_accuracy.png")


def figure2_decision_change_heatmap(delta_df: pd.DataFrame, output_dir: Path):
    """Figure 2: Decision-change heatmap with seaborn."""
    _style()

    if delta_df.empty:
        return

    change_metrics = ["refer_match_changed", "urgency_match_changed", "labs_match_changed",
                     "imaging_match_changed", "aspiration_match_changed", "psychologized_changed",
                     "urgency_downgraded_changed"]
    available_metrics = [m for m in change_metrics if m in delta_df.columns]

    if not available_metrics:
        return

    data = delta_df[delta_df["dimension"] != "baseline"].copy()
    groups = sorted(data[["dimension", "level"]].drop_duplicates().values.tolist(),
                   key=lambda x: (dim_sort_key(x[0]), x[1]))

    matrix = []
    row_labels = []
    for dim, level in groups:
        grp = data[(data["dimension"] == dim) & (data["level"] == level)]
        row = [grp[m].mean() * 100 if m in grp.columns else 0 for m in available_metrics]
        matrix.append(row)
        label = LEVEL_LABELS.get((dim, level), f"{dim}={level}")
        row_labels.append(f"{DIM_LABELS.get(dim, dim)}\n{label}")

    fig, ax = plt.subplots(figsize=(11, max(7, len(groups) * 0.5)))

    df_hm = pd.DataFrame(matrix, columns=available_metrics, index=row_labels)
    sns.heatmap(df_hm, annot=True, fmt=".0f", cmap="YlOrRd", cbar_kws={"label": "Change Rate (%)"},
               ax=ax, vmin=0, vmax=60, linewidths=0.5, linecolor=GRID)

    ax.set_xticklabels([m.replace("_changed", "").replace("_", " ").title() for m in available_metrics],
                       rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(row_labels, fontsize=8, rotation=0)

    _header(fig, "Decision Changes from Baseline", "Heatmap of metric-level changes across dimensions")
    _wm(ax)

    fig.subplots_adjust(left=0.2, right=0.95, top=0.92, bottom=0.15)
    _save_fig(fig, output_dir / "fig02_decision_change_heatmap.png")


def figure3_referral_urgency_changes(delta_df: pd.DataFrame, output_dir: Path):
    """Figure 3: Referral & Urgency changes by dimension (fixed aggregation)."""
    _style()

    if delta_df.empty:
        return

    data = delta_df[delta_df["dimension"] != "baseline"].copy()

    # Fix: group by dimension only, aggregate metrics without multiplying dimension column
    agg_data = []
    for dim in data["dimension"].unique():
        dim_data = data[data["dimension"] == dim]
        refer_rate = dim_data["refer_match_changed"].mean() * 100 if "refer_match_changed" in dim_data.columns else 0
        urgency_rate = dim_data["urgency_match_changed"].mean() * 100 if "urgency_match_changed" in dim_data.columns else 0
        agg_data.append({
            "dimension": dim,
            "refer_match_changed": refer_rate,
            "urgency_match_changed": urgency_rate
        })

    agg = pd.DataFrame(agg_data).sort_values("refer_match_changed")

    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(agg))

    ax.barh(y_pos - 0.2, agg["refer_match_changed"], 0.4, label="Referral", color=BLUE, alpha=0.8)
    ax.barh(y_pos + 0.2, agg["urgency_match_changed"], 0.4, label="Urgency", color=RED, alpha=0.8)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([DIM_LABELS.get(d, d) for d in agg["dimension"]])
    ax.set_xlabel("Change Rate (%)", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(axis="x", alpha=0.3)

    _header(fig, "Referral & Urgency Changes", "By bias dimension")
    _wm(ax)

    fig.subplots_adjust(left=0.2, right=0.95, top=0.92, bottom=0.08)
    _save_fig(fig, output_dir / "fig03_referral_urgency_changes.png")


def figure4_psychologization_dual_panel(delta_df: pd.DataFrame, baseline_df: pd.DataFrame, output_dir: Path):
    """Figure 4: Psychologization error rate (dual panel)."""
    _style()

    if delta_df.empty:
        return

    baseline_error = (
        baseline_df["psychologization_error"].mean() * 100
        if "psychologization_error" in baseline_df.columns else 0
    )

    data = delta_df[delta_df["dimension"] != "baseline"].copy()
    groups = sorted(data[["dimension", "level"]].drop_duplicates().values.tolist(),
                   key=lambda x: (dim_sort_key(x[0]), x[1]))

    errors_base = []
    errors_iter = []
    deltas = []
    row_labels = []
    dim_list = []

    for dim, level in groups:
        grp = data[(data["dimension"] == dim) & (data["level"] == level)]
        label = LEVEL_LABELS.get((dim, level), f"{dim}={level}")

        if "psychologization_error_iter" in grp.columns:
            err_rate = grp["psychologization_error_iter"].mean() * 100
            delta = err_rate - baseline_error
            errors_iter.append(err_rate)
            deltas.append(delta)
            row_labels.append(label)
            dim_list.append(dim)

    fig, axes = plt.subplots(1, 2, figsize=(14, max(6, len(row_labels) * 0.4)))

    y_pos = np.arange(len(row_labels))

    # Panel A: Error rates by dimension x level
    ax = axes[0]
    colors = [DIM_COLORS.get(d, SLATE) for d in dim_list]
    ax.barh(y_pos, errors_iter, color=colors, alpha=0.8, height=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(row_labels, fontsize=9)
    ax.set_xlabel("Psychologization Error Rate (%)", fontsize=11)
    ax.set_title("A  Error Rate by Dimension×Level", fontweight="bold", loc="left")
    ax.axvline(baseline_error, color=RED, linestyle="--", linewidth=1.5, alpha=0.5, label="Baseline")
    ax.legend(fontsize=9)
    ax.grid(axis="x", alpha=0.3)

    # Panel B: Delta from baseline (diverging)
    ax = axes[1]
    colors_delta = [RED if d < 0 else GREEN for d in deltas]
    ax.barh(y_pos, deltas, color=colors_delta, alpha=0.8, height=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([""] * len(row_labels))
    ax.set_xlabel("Delta from Baseline (%)", fontsize=11)
    ax.set_title("B  Change from Baseline", fontweight="bold", loc="left")
    ax.axvline(0, color=TXT, linestyle="-", linewidth=0.7, alpha=0.5)
    ax.grid(axis="x", alpha=0.3)

    _header(fig, "Psychologization Error Analysis", "Baseline vs iteration across dimensions")
    _wm(axes[0])

    fig.subplots_adjust(left=0.25, right=0.95, top=0.92, bottom=0.08, wspace=0.3)
    _save_fig(fig, output_dir / "fig04_psychologization_dual.png")


def figure5_urgency_direction_stacked(delta_df: pd.DataFrame, output_dir: Path):
    """Figure 5: Urgency direction stacked bar (downgraded/correct/upgraded)."""
    _style()

    if delta_df.empty:
        return

    data = delta_df[delta_df["dimension"] != "baseline"].copy()
    groups = sorted(data[["dimension", "level"]].drop_duplicates().values.tolist(),
                   key=lambda x: (dim_sort_key(x[0]), x[1]))

    down_pcts = []
    corr_pcts = []
    up_pcts = []
    row_labels = []

    for dim, level in groups:
        grp = data[(data["dimension"] == dim) & (data["level"] == level)]
        label = LEVEL_LABELS.get((dim, level), f"{dim}={level}")

        total = len(grp)
        downgraded = int(grp["urgency_downgraded_iter"].sum()) if "urgency_downgraded_iter" in grp.columns else 0
        upgraded = int(grp["urgency_upgraded_iter"].sum()) if "urgency_upgraded_iter" in grp.columns else 0
        correct = total - downgraded - upgraded

        down_pcts.append(downgraded / total * 100 if total > 0 else 0)
        corr_pcts.append(correct / total * 100 if total > 0 else 0)
        up_pcts.append(upgraded / total * 100 if total > 0 else 0)
        row_labels.append(label)

    fig, ax = plt.subplots(figsize=(12, max(6, len(row_labels) * 0.4)))

    y_pos = np.arange(len(row_labels))

    ax.barh(y_pos, down_pcts, label="Downgraded", color=RED, alpha=0.85, height=0.7)
    ax.barh(y_pos, corr_pcts, left=down_pcts, label="Correct", color=GREEN, alpha=0.85, height=0.7)
    ax.barh(y_pos, up_pcts, left=[d+c for d,c in zip(down_pcts, corr_pcts)],
           label="Upgraded", color=AMBER, alpha=0.85, height=0.7)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(row_labels, fontsize=9)
    ax.set_xlabel("Percentage (%)", fontsize=11)
    ax.set_xlim(0, 100)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(axis="x", alpha=0.3)

    _header(fig, "Urgency Direction Composition", "Proportion downgraded/correct/upgraded by dimension×level")
    _wm(ax)

    fig.subplots_adjust(left=0.25, right=0.95, top=0.92, bottom=0.08)
    _save_fig(fig, output_dir / "fig05_urgency_stacked.png")


def figure6_composite_delta_diverging(delta_df: pd.DataFrame, baseline_df: pd.DataFrame, output_dir: Path):
    """Figure 6: Composite score delta (diverging bar)."""
    _style()

    if delta_df.empty:
        return

    baseline_mean = baseline_df["composite_score"].mean() if "composite_score" in baseline_df.columns else 0

    data = delta_df[delta_df["dimension"] != "baseline"].copy()
    groups = sorted(data[["dimension", "level"]].drop_duplicates().values.tolist(),
                   key=lambda x: (dim_sort_key(x[0]), x[1]))

    deltas = []
    row_labels = []

    for dim, level in groups:
        grp = data[(data["dimension"] == dim) & (data["level"] == level)]
        label = LEVEL_LABELS.get((dim, level), f"{dim}={level}")

        if "composite_score_iter" in grp.columns:
            iter_mean = grp["composite_score_iter"].mean()
            delta = (iter_mean - baseline_mean) * 100
            deltas.append(delta)
            row_labels.append(label)

    # Sort by delta magnitude
    sorted_idx = sorted(range(len(deltas)), key=lambda i: abs(deltas[i]))
    deltas_sorted = [deltas[i] for i in sorted_idx]
    row_labels_sorted = [row_labels[i] for i in sorted_idx]

    fig, ax = plt.subplots(figsize=(10, max(6, len(row_labels) * 0.4)))

    y_pos = np.arange(len(deltas_sorted))
    colors = [RED if d < 0 else GREEN for d in deltas_sorted]

    ax.barh(y_pos, deltas_sorted, color=colors, alpha=0.85, height=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(row_labels_sorted, fontsize=9)
    ax.set_xlabel("Composite Score Delta (%)", fontsize=11)
    ax.axvline(0, color=TXT, linestyle="-", linewidth=1, alpha=0.5)
    ax.grid(axis="x", alpha=0.3)

    _header(fig, "Composite Concordance Delta from Baseline", "Negative (red) = worsening; positive (green) = improving")
    _wm(ax)

    fig.subplots_adjust(left=0.25, right=0.95, top=0.92, bottom=0.08)
    _save_fig(fig, output_dir / "fig06_composite_delta.png")


def figure7_composite_by_group_box(delta_df: pd.DataFrame, baseline_df: pd.DataFrame, output_dir: Path):
    """Figure 7: Composite score by dimension (box/strip plot)."""
    _style()

    if delta_df.empty:
        return

    baseline_mean = baseline_df["composite_score"].mean() if "composite_score" in baseline_df.columns else 0

    data = delta_df[delta_df["dimension"] != "baseline"].copy()
    dims = sorted(data["dimension"].unique(), key=dim_sort_key)

    fig, ax = plt.subplots(figsize=(12, 6))

    positions = []
    labels_plot = []
    pos = 0

    for dim in dims:
        dim_data = data[data["dimension"] == dim]
        if "composite_score_iter" in dim_data.columns:
            scores = dim_data["composite_score_iter"].dropna().values * 100

            # Box plot
            bp = ax.boxplot([scores], positions=[pos], widths=0.5, patch_artist=True,
                           boxprops=dict(facecolor=DIM_COLORS.get(dim, SLATE), alpha=0.6),
                           whiskerprops=dict(color=TXT, linewidth=0.8),
                           capprops=dict(color=TXT, linewidth=0.8),
                           medianprops=dict(color=RED, linewidth=1.5))

            # Strip plot
            ax.scatter([pos] * len(scores), scores, alpha=0.3, s=20, color=TXT)

            positions.append(pos)
            labels_plot.append(DIM_LABELS.get(dim, dim))
            pos += 1

    ax.axhline(baseline_mean * 100, color=BLUE, linestyle="--", linewidth=2, alpha=0.6, label="Baseline Mean")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels_plot, rotation=30, ha="right")
    ax.set_ylabel("Composite Score (%)", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    _header(fig, "Composite Score Distribution by Dimension", "Box plots with individual level points")
    _wm(ax)

    fig.subplots_adjust(left=0.08, right=0.95, top=0.92, bottom=0.15)
    _save_fig(fig, output_dir / "fig07_composite_box.png")


def figure8_disease_category_heatmap(delta_df: pd.DataFrame, output_dir: Path):
    """Figure 8: Disease category × dimension interaction heatmap."""
    _style()

    if delta_df.empty or "gt_category" not in delta_df.columns:
        return

    data = delta_df[delta_df["dimension"] != "baseline"].copy()

    categories = sorted(data["gt_category"].dropna().unique())
    dims = sorted(data["dimension"].unique(), key=dim_sort_key)

    matrix = np.full((len(categories), len(dims)), np.nan)

    for i, cat in enumerate(categories):
        for j, dim in enumerate(dims):
            grp = data[(data["gt_category"] == cat) & (data["dimension"] == dim)]
            if "composite_delta" in grp.columns:
                deltas = grp["composite_delta"].dropna()
                if len(deltas) > 0:
                    matrix[i, j] = deltas.mean() * 100

    fig, ax = plt.subplots(figsize=(12, max(6, len(categories) * 0.6)))

    vmax = np.nanmax(np.abs(matrix)) if not np.all(np.isnan(matrix)) else 5
    cmap_div = plt.cm.RdBu_r

    im = ax.imshow(matrix, cmap=cmap_div, aspect="auto", vmin=-vmax, vmax=vmax)

    ax.set_xticks(np.arange(len(dims)))
    ax.set_xticklabels([DIM_LABELS.get(d, d) for d in dims], rotation=45, ha="right", fontsize=10)
    ax.set_yticks(np.arange(len(categories)))
    ax.set_yticklabels([str(c)[:20] for c in categories], fontsize=10)

    for i in range(len(categories)):
        for j in range(len(dims)):
            v = matrix[i, j]
            if not np.isnan(v):
                text_color = WHITE if abs(v) > vmax * 0.5 else TXT
                ax.text(j, i, f"{v:.1f}%", ha="center", va="center", color=text_color, fontsize=9, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Mean Composite Delta (%)", fontsize=10)

    _header(fig, "Disease Category × Dimension Interaction", "Mean composite score change by category and bias dimension")
    _wm(ax)

    fig.subplots_adjust(left=0.15, right=0.92, top=0.92, bottom=0.15)
    _save_fig(fig, output_dir / "fig08_category_heatmap.png")


def figure9_persona_susceptibility(delta_df: pd.DataFrame, output_dir: Path):
    """Figure 9: Decision shift susceptibility by system prompt persona."""
    _style()

    if delta_df.empty or "persona" not in delta_df.columns:
        return

    personas = sorted(delta_df["persona"].dropna().unique())

    metrics = ["refer_match_changed", "urgency_match_changed", "psychologized_changed"]
    available_metrics = [m for m in metrics if m in delta_df.columns]

    fig, ax = plt.subplots(figsize=(11, 6))

    x = np.arange(len(personas))
    width = 0.25

    for idx, metric in enumerate(available_metrics):
        rates = []
        for persona in personas:
            persona_data = delta_df[delta_df["persona"] == persona]
            rate = persona_data[metric].mean() * 100 if metric in persona_data.columns else 0
            rates.append(rate)

        label = metric.replace("_changed", "").replace("_", " ").title()
        ax.bar(x + idx * width, rates, width, label=label,
              color=[BLUE, RED, PURPLE][idx], alpha=0.8)

    ax.set_xticks(x + width)
    ax.set_xticklabels([PERSONA_LABELS.get(p, p) for p in personas], fontsize=10)
    ax.set_ylabel("Mean Decision Change Rate (%)", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    _header(fig, "Decision Shift Susceptibility by System Prompt", "Which personas are most susceptible to bias?")
    _wm(ax)

    fig.subplots_adjust(left=0.12, right=0.95, top=0.92, bottom=0.12)
    _save_fig(fig, output_dir / "fig09_persona_shift.png")


def figure10_model_persona_heatmap(delta_df: pd.DataFrame, output_dir: Path):
    """Figure 10: Model × Persona interaction heatmap."""
    _style()

    if delta_df.empty or "model" not in delta_df.columns or "persona" not in delta_df.columns:
        return

    models = sorted(delta_df["model"].dropna().unique())
    personas = sorted(delta_df["persona"].dropna().unique())

    matrix = np.full((len(models), len(personas)), np.nan)

    for i, model in enumerate(models):
        for j, persona in enumerate(personas):
            grp = delta_df[(delta_df["model"] == model) & (delta_df["persona"] == persona)]
            if "composite_delta" in grp.columns:
                deltas = grp["composite_delta"].dropna()
                if len(deltas) > 0:
                    matrix[i, j] = deltas.mean() * 100

    fig, ax = plt.subplots(figsize=(10, max(6, len(models) * 0.6)))

    vmax = np.nanmax(np.abs(matrix)) if not np.all(np.isnan(matrix)) else 5
    cmap_div = plt.cm.RdBu_r

    im = ax.imshow(matrix, cmap=cmap_div, aspect="auto", vmin=-vmax, vmax=vmax)

    ax.set_xticks(np.arange(len(personas)))
    ax.set_xticklabels([PERSONA_LABELS.get(p, p) for p in personas], rotation=30, ha="right", fontsize=10)
    ax.set_yticks(np.arange(len(models)))
    ax.set_yticklabels(models, fontsize=10)

    for i in range(len(models)):
        for j in range(len(personas)):
            v = matrix[i, j]
            if not np.isnan(v):
                text_color = WHITE if abs(v) > vmax * 0.5 else TXT
                ax.text(j, i, f"{v:.1f}%", ha="center", va="center", color=text_color, fontsize=9, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Mean Composite Delta (%)", fontsize=10)

    _header(fig, "Model × Persona Interaction", "Mean composite score change by model and system prompt")
    _wm(ax)

    fig.subplots_adjust(left=0.12, right=0.92, top=0.92, bottom=0.15)
    _save_fig(fig, output_dir / "fig10_model_persona.png")


def figure11_model_shift_paired(delta_df: pd.DataFrame, baseline_df: pd.DataFrame, output_dir: Path):
    """Figure 11: Shift by model (paired dot plot + delta bars)."""
    _style()

    if delta_df.empty or "model" not in delta_df.columns:
        return

    models = sorted(delta_df["model"].dropna().unique())
    baseline_mean = baseline_df["composite_score"].mean() if "composite_score" in baseline_df.columns else 0

    baseline_scores = []
    iteration_scores = []
    deltas = []

    for model in models:
        model_base = baseline_df[baseline_df["model"] == model]
        model_iter = delta_df[delta_df["model"] == model]

        if not model_base.empty and "composite_score" in model_base.columns:
            bl_score = model_base["composite_score"].mean()
        else:
            bl_score = baseline_mean

        if not model_iter.empty and "composite_score_iter" in model_iter.columns:
            it_score = model_iter["composite_score_iter"].mean()
        else:
            it_score = bl_score

        baseline_scores.append(bl_score)
        iteration_scores.append(it_score)
        deltas.append((it_score - bl_score) * 100)

    fig, axes = plt.subplots(1, 2, figsize=(14, max(6, len(models) * 0.4)))
    fig.subplots_adjust(left=0.15, right=0.95, top=0.92, bottom=0.08, wspace=0.3)

    y_pos = np.arange(len(models))

    # Panel A: Paired dot plot
    ax = axes[0]
    for i, (bl, it) in enumerate(zip(baseline_scores, iteration_scores)):
        color = GREEN if it > bl else RED
        ax.plot([bl, it], [i, i], color=color, linewidth=1.5, alpha=0.6, zorder=3)

    ax.scatter(baseline_scores, y_pos, color=BLUE, s=60, zorder=5, edgecolors=WHITE, linewidth=0.8, label="Baseline")
    ax.scatter(iteration_scores, y_pos, color=RED, s=60, zorder=5, edgecolors=WHITE, linewidth=0.8, label="Iteration")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(models, fontsize=10)
    ax.set_xlabel("Composite Score", fontsize=11)
    ax.set_title("A  Baseline vs Iteration Score", fontweight="bold", loc="left")
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    ax.set_xlim(-0.05, 1.05)

    # Panel B: Delta bars
    ax = axes[1]
    colors_delta = [RED if d < 0 else GREEN for d in deltas]
    ax.barh(y_pos, deltas, color=colors_delta, alpha=0.85, height=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([""] * len(models))
    ax.set_xlabel("Composite Score Delta (%)", fontsize=11)
    ax.set_title("B  Change from Baseline", fontweight="bold", loc="left")
    ax.axvline(0, color=TXT, linestyle="-", linewidth=1, alpha=0.5)
    ax.grid(axis="x", alpha=0.3)

    _header(fig, "Shift in Composite Score by Model", "Baseline vs iteration performance across models")
    _wm(axes[0])

    _save_fig(fig, output_dir / "fig11_model_shift.png")


# ============================================================================
# EXCEL OUTPUT
# ============================================================================

def write_excel_tables(output_path: Path, tables: Dict[str, pd.DataFrame]):
    """Write all tables to Excel workbook with formatting."""
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for sheet_name, df in tables.items():
            if df.empty:
                continue
            df.to_excel(writer, sheet_name=sheet_name, index=False)
            worksheet = writer.sheets[sheet_name]

            from openpyxl.styles import Font, PatternFill, Alignment
            header_fill = PatternFill(start_color="3E82FC", end_color="3E82FC", fill_type="solid")
            header_font = Font(bold=True, color="FFFFFF", size=11)

            for cell in worksheet[1]:
                cell.fill = header_fill
                cell.font = header_font
                cell.alignment = Alignment(horizontal="center", vertical="center")

            for column in worksheet.columns:
                max_length = 0
                col_letter = column[0].column_letter
                for cell in column:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except:
                        pass
                worksheet.column_dimensions[col_letter].width = min(max_length + 2, 50)

# ============================================================================
# CONSOLIDATED PDF
# ============================================================================

def create_consolidated_pdf(figures_dir: Path, output_path: Path):
    """Create multi-page PDF from all PNG figures."""
    try:
        from PIL import Image
        png_files = sorted(figures_dir.glob("fig*.png"))
        if not png_files:
            return

        images = []
        for png_file in png_files:
            img = Image.open(png_file)
            if img.mode != "RGB":
                img = img.convert("RGB")
            images.append(img)

        if images:
            images[0].save(output_path, save_all=True, append_images=images[1:])
            print(f"Consolidated PDF: {output_path.name}")
    except ImportError:
        print("(PIL not available for PDF consolidation)")

# ============================================================================
# MAIN ANALYSIS FUNCTION
# ============================================================================

def main(input_path: Optional[str] = None):
    """Execute full analysis pipeline."""
    print("\n" + "="*80)
    print(" RHEUMATO BIAS PIPELINE: COMPREHENSIVE ANALYSIS")
    print("="*80)

    # Load data
    if input_path is None:
        script_dir = Path(__file__).resolve().parent
        candidates = sorted(script_dir.glob("*.jsonl"), reverse=True)
        if not candidates:
            candidates = sorted(script_dir.glob("*.xlsx"), reverse=True)
        if not candidates:
            print("ERROR: No data files found. Provide path as argument.")
            return
        input_path = candidates[0]

    input_path = Path(input_path)
    print(f"\nLoading: {input_path.name}")

    df = load_data(input_path)
    print(f"Records: {len(df)}")

    # Separate baseline
    baseline_df = df[df["condition"] == "baseline"].copy()
    print(f"Baseline: {len(baseline_df)}")

    # Compute deltas
    print("\nComputing decision deltas...")
    delta_df = compute_deltas(df)
    print(f"Paired comparisons: {len(delta_df)}")

    # Generate tables
    print("\nGenerating 10 publication tables...")
    tables = {
        "T1_Baseline_Accuracy": table1_baseline_accuracy(df, baseline_df),
        "T2_Model_Persona": table2_baseline_by_model_persona(baseline_df),
        "T3_Decision_Shifts": table3_decision_shifts(delta_df),
        "T4_GT_Concordance": table4_gt_concordance(baseline_df, delta_df),
        "T5_Psychologization": table5_psychologization(baseline_df, delta_df),
        "T6_Urgency_Shifts": table6_urgency_shifts(delta_df),
        "T7_Statistical_Tests": table7_statistical_tests(delta_df),
        "T8_Composite_Scores": table8_composite_scores(delta_df),
        "T9_Shift_by_Persona": table9_shift_by_persona(delta_df),
        "T10_Model_Dimension": table10_model_dimension_interaction(delta_df),
    }

    for name, tbl in tables.items():
        print(f"  {name}: {len(tbl)} rows")

    # Output directory
    output_dir = Path("figures")
    output_dir.mkdir(exist_ok=True)

    # Generate figures
    print("\nGenerating 11 publication figures...")
    figure1_baseline_accuracy(baseline_df, output_dir)
    figure2_decision_change_heatmap(delta_df, output_dir)
    figure3_referral_urgency_changes(delta_df, output_dir)
    figure4_psychologization_dual_panel(delta_df, baseline_df, output_dir)
    figure5_urgency_direction_stacked(delta_df, output_dir)
    figure6_composite_delta_diverging(delta_df, baseline_df, output_dir)
    figure7_composite_by_group_box(delta_df, baseline_df, output_dir)
    figure8_disease_category_heatmap(delta_df, output_dir)
    figure9_persona_susceptibility(delta_df, output_dir)
    figure10_model_persona_heatmap(delta_df, output_dir)
    figure11_model_shift_paired(delta_df, baseline_df, output_dir)

    # Write Excel
    stem = input_path.stem
    excel_path = Path(f"analysis_tables_{stem}.xlsx")
    print(f"\nWriting Excel: {excel_path.name}")
    write_excel_tables(excel_path, tables)

    # Create PDF
    pdf_path = Path(f"{stem}_all_figures.pdf")
    create_consolidated_pdf(output_dir, pdf_path)

    print("\n" + "="*80)
    print(" ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nOutputs:")
    print(f"  Excel tables: {excel_path}")
    print(f"  Figures: {output_dir}/")
    print(f"  PDF: {pdf_path}")
    print()

if __name__ == "__main__":
    input_file = sys.argv[1] if len(sys.argv) > 1 else None
    main(input_file)
