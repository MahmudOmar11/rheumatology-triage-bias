#!/usr/bin/env python3
"""
Rheumato Bias Pipeline: Multi-Model Publication-Quality Analysis Suite
========================================================================

Analysis pipeline for sociodemographic bias in LLM clinical triage across multiple models.

Reads checkpoint JSONL (or directory of JSONL files) and produces publication-quality
tables and figures suitable for Lancet Rheumatology.

TABLES (Excel workbook):
  T1  Baseline accuracy (pooled)
  T2  Baseline accuracy by model
  T3  Baseline accuracy by persona
  T4  Baseline accuracy by provider
  T5  Decision shifts by dimension × level (pooled)
  T6  Decision shifts by model × dimension
  T7  Decision shifts by persona × dimension
  T8  Decision shifts by provider × dimension
  T9  Psychologization rates
  T10 Urgency direction (down/correct/up)
  T11 Composite score deltas (paired t-tests)
  T12 Statistical tests master table
  T13 Model ranking
  T14 Dimension ranking
  T15 Pairwise within-dimension comparisons

FIGURES (PNG 300 DPI + consolidated PDF):
  fig01  Baseline accuracy bar chart
  fig02  Baseline accuracy by model
  fig03  Decision-change heatmap (pooled)
  fig04  Decision-change heatmap by model
  fig05  Referral & urgency changes paired bars
  fig06  Psychologization rates by dimension
  fig07  Urgency direction stacked bar
  fig08  Composite delta diverging bar
  fig09  Model susceptibility scatter
  fig10  Model × dimension heatmap
  fig11  Provider comparison
  fig12  Persona susceptibility
  fig13  Dimension group comparison
  fig14  Disease category × dimension interaction
  fig15  Forest plot (model rankings)

Usage
-----
  python analysis.py [input_dir_or_file] [output_dir]
  python analysis.py  # interactive prompt

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
from scipy.stats import binomtest, wilcoxon, chi2_contingency
import scipy.stats as spstats
from statsmodels.stats.multitest import multipletests

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as mtick
import seaborn as sns
from PIL import Image

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

PROVIDER_MAP = {
    "gpt-": "OpenAI", "o4-": "OpenAI",
    "claude-": "Anthropic",
    "gemini-": "Google",
}

DIM_GROUPS = {
    "Demographics": ["race", "ses", "sex"],
    "Clinical History": ["psych_hx", "substance", "weight"],
    "Communication": ["tone", "literacy", "language"],
    "System / Anchoring": ["anchoring"],
}

PROVIDER_COLORS = {
    "OpenAI": BLUE,
    "Anthropic": AMBER,
    "Google": GREEN,
}

# ============================================================================
# STYLING HELPERS
# ============================================================================

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
# DATA LOADING & VALIDATION
# ============================================================================

def load_data(path: Path) -> pd.DataFrame:
    """Load JSONL or Excel pipeline output. Handles single file or directory."""
    frames = []

    if path.is_dir():
        # Scan directory for .jsonl AND .xlsx files
        for jsonl_file in sorted(path.glob("*.jsonl")):
            rows = []
            with open(jsonl_file, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        rows.append(json.loads(line))
            if rows:
                print(f"  Loaded {len(rows)} records from {jsonl_file.name}")
                frames.append(pd.DataFrame(rows))

        for xlsx_file in sorted(path.glob("*.xlsx")):
            try:
                df_xlsx = pd.read_excel(xlsx_file, sheet_name="Raw_Outputs")
                print(f"  Loaded {len(df_xlsx)} records from {xlsx_file.name}")
                frames.append(df_xlsx)
            except Exception as e:
                print(f"  ⚠ Skipped {xlsx_file.name}: {e}")

        if not frames:
            raise ValueError(f"No .jsonl or .xlsx files found in {path}")
        return pd.concat(frames, ignore_index=True)

    elif path.suffix == ".jsonl":
        rows = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
        return pd.DataFrame(rows)

    elif path.suffix in [".xlsx", ".xls"]:
        return pd.read_excel(path, sheet_name="Raw_Outputs")

    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")


def add_provider_column(df: pd.DataFrame) -> pd.DataFrame:
    """Derive provider from model name if not present."""
    if "provider" not in df.columns:
        df["provider"] = df["model"].apply(lambda m: _get_provider(m))
    return df

def _get_provider(model: str) -> str:
    """Get provider name from model string."""
    for prefix, provider in PROVIDER_MAP.items():
        if model.startswith(prefix):
            return provider
    return "Unknown"

def prompt_for_paths() -> Tuple[Path, Path]:
    """Prompt user for input and output paths."""
    while True:
        input_path = input("\nEnter input file/directory path (.jsonl, .xlsx, or directory): ").strip()
        input_path = Path(input_path)
        if input_path.exists():
            break
        print(f"Path not found: {input_path}")

    while True:
        output_dir = input("Enter output directory (default: ./output): ").strip() or "./output"
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        break

    return input_path, output_dir

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

def apply_fdr_correction(pvalues: List[float]) -> Tuple[List[bool], List[float]]:
    """Apply Benjamini-Hochberg FDR correction."""
    pvalues_arr = np.array(pvalues)
    rejected, corrected, _, _ = multipletests(pvalues_arr, alpha=ALPHA, method="fdr_bh")
    return rejected, corrected

# ============================================================================
# DELTA COMPUTATION (multi-model)
# ============================================================================

def compute_deltas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute decision changes by comparing iteration rows to baseline.
    Match on (case_id, case_rephrase_id, repeat_id, model, persona).
    Vectorized across all models.
    """
    baseline_df = df[df["condition"] == "baseline"].copy()
    iteration_df = df[df["condition"] != "baseline"].copy()

    if len(baseline_df) == 0 or len(iteration_df) == 0:
        print("WARNING: No baseline or iteration data found!")
        return pd.DataFrame()

    binary_metrics = [
        "refer_match", "urgency_match", "labs_match", "imaging_match", "aspiration_match",
        "psychologized", "psychologization_error", "urgency_downgraded", "urgency_upgraded",
        "acuity_match", "acuity_downgraded", "acuity_upgraded", "under_referral", "over_referral",
        "dx_match_primary", "dx_match_top3", "reassurance_error", "immediate_action_match",
    ]

    # Create match key including model
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
        iteration_df[["match_key", "dimension", "level", "condition", "composite_score",
                     "gt_category", "gt_acuity"] +
                     [c for c in iteration_df.columns if c in binary_metrics]],
        on="match_key", suffixes=("_base", "_iter"), how="inner"
    )

    if len(merged) == 0:
        print("WARNING: No matched baseline/iteration pairs!")
        return pd.DataFrame()

    # Rename suffixed columns: use iteration's dimension/level as canonical
    if "dimension_iter" in merged.columns:
        merged["dimension"] = merged["dimension_iter"]
        merged.drop(columns=["dimension_base", "dimension_iter"], errors="ignore", inplace=True)
    if "level_iter" in merged.columns:
        merged["level"] = merged["level_iter"]
        merged.drop(columns=["level_base", "level_iter"], errors="ignore", inplace=True)
    if "condition_iter" in merged.columns:
        merged["condition"] = merged["condition_iter"]
        merged.drop(columns=["condition_base", "condition_iter"], errors="ignore", inplace=True)
    if "gt_category_iter" in merged.columns:
        merged["gt_category"] = merged["gt_category_base"]
        merged.drop(columns=["gt_category_base", "gt_category_iter"], errors="ignore", inplace=True)
    if "gt_acuity_iter" in merged.columns:
        merged["gt_acuity"] = merged["gt_acuity_base"]
        merged.drop(columns=["gt_acuity_base", "gt_acuity_iter"], errors="ignore", inplace=True)

    # Compute deltas for binary metrics (cast to int to avoid boolean subtract error)
    for metric in binary_metrics:
        if f"{metric}_base" in merged.columns and f"{metric}_iter" in merged.columns:
            base_int = merged[f"{metric}_base"].fillna(0).astype(int)
            iter_int = merged[f"{metric}_iter"].fillna(0).astype(int)
            merged[f"{metric}_changed"] = (base_int != iter_int).astype(int)
            merged[f"{metric}_delta"] = iter_int - base_int
            merged[f"{metric}_direction"] = iter_int - base_int

    # Composite score delta
    merged["composite_delta"] = pd.to_numeric(merged["composite_score_iter"], errors="coerce") - pd.to_numeric(merged["composite_score_base"], errors="coerce")

    return merged

# ============================================================================
# TABLE GENERATION: BASELINE ACCURACY (POOLED, BY MODEL, BY PERSONA, BY PROVIDER)
# ============================================================================

def table_baseline_accuracy_pooled(df: pd.DataFrame) -> pd.DataFrame:
    """T1: Overall baseline accuracy (all models pooled) with Wilson CI."""
    baseline_df = df[df["condition"] == "baseline"].copy()

    binary_metrics = [
        "refer_match", "urgency_match", "labs_match", "imaging_match", "aspiration_match",
        "acuity_match", "dx_match_primary", "dx_match_top3", "immediate_action_match",
        "psychologized", "psychologization_error", "urgency_downgraded", "urgency_upgraded",
        "acuity_downgraded", "acuity_upgraded", "under_referral", "over_referral", "reassurance_error"
    ]

    results = []
    for metric in binary_metrics:
        if metric in baseline_df.columns:
            total = len(baseline_df)
            successes = int(baseline_df[metric].sum())
            rate = successes / total if total > 0 else 0
            lower, upper = wilson_ci(successes, total)

            results.append({
                "Metric": metric.replace("_", " ").title(),
                "N": total,
                "Count": successes,
                "Rate": rate,
                "Rate %": f"{rate*100:.1f}%",
                "95% CI": f"[{lower*100:.1f}–{upper*100:.1f}%]",
                "Composite Score": baseline_df["composite_score"].mean(),
                "Composite SD": baseline_df["composite_score"].std(),
            })

    return pd.DataFrame(results)

def table_baseline_by_model(df: pd.DataFrame) -> pd.DataFrame:
    """T2: Baseline accuracy by model with composite score and CI."""
    baseline_df = df[df["condition"] == "baseline"].copy()

    results = []
    for model in sorted(baseline_df["model"].unique()):
        model_data = baseline_df[baseline_df["model"] == model]
        provider = model_data["provider"].iloc[0] if len(model_data) > 0 else "Unknown"

        n = len(model_data)
        composite_mean = model_data["composite_score"].mean()
        composite_sd = model_data["composite_score"].std()

        refer_match = model_data["refer_match"].sum()
        urgency_match = model_data["urgency_match"].sum()
        imaging_match = model_data["imaging_match"].sum()
        dx_match = model_data["dx_match_primary"].sum()
        psych_rate = model_data["psychologized"].sum()

        results.append({
            "Model": model,
            "Provider": provider,
            "N Cases": n,
            "Composite Score": f"{composite_mean:.3f} ± {composite_sd:.3f}",
            "Referral Match %": f"{refer_match/n*100:.1f}%",
            "Urgency Match %": f"{urgency_match/n*100:.1f}%",
            "Imaging Match %": f"{imaging_match/n*100:.1f}%",
            "Dx Match %": f"{dx_match/n*100:.1f}%",
            "Psychologized %": f"{psych_rate/n*100:.1f}%",
        })

    return pd.DataFrame(results)

def table_baseline_by_persona(df: pd.DataFrame) -> pd.DataFrame:
    """T3: Baseline accuracy by persona."""
    baseline_df = df[df["condition"] == "baseline"].copy()

    results = []
    for persona in sorted(baseline_df["persona"].unique()):
        persona_data = baseline_df[baseline_df["persona"] == persona]
        n = len(persona_data)

        results.append({
            "Persona": PERSONA_LABELS.get(persona, persona),
            "N Cases": n,
            "Composite Mean": f"{persona_data['composite_score'].mean():.3f}",
            "Composite SD": f"{persona_data['composite_score'].std():.3f}",
            "Referral Match %": f"{persona_data['refer_match'].sum()/n*100:.1f}%",
            "Urgency Match %": f"{persona_data['urgency_match'].sum()/n*100:.1f}%",
            "Psychologized %": f"{persona_data['psychologized'].sum()/n*100:.1f}%",
        })

    return pd.DataFrame(results)

def table_baseline_by_provider(df: pd.DataFrame) -> pd.DataFrame:
    """T4: Baseline accuracy by provider."""
    baseline_df = df[df["condition"] == "baseline"].copy()

    results = []
    for provider in sorted(baseline_df["provider"].unique()):
        provider_data = baseline_df[baseline_df["provider"] == provider]
        n = len(provider_data)
        models = provider_data["model"].nunique()

        results.append({
            "Provider": provider,
            "N Cases": n,
            "N Models": models,
            "Composite Mean": f"{provider_data['composite_score'].mean():.3f}",
            "Composite SD": f"{provider_data['composite_score'].std():.3f}",
            "Referral Match %": f"{provider_data['refer_match'].sum()/n*100:.1f}%",
            "Urgency Match %": f"{provider_data['urgency_match'].sum()/n*100:.1f}%",
            "Imaging Match %": f"{provider_data['imaging_match'].sum()/n*100:.1f}%",
            "Dx Match %": f"{provider_data['dx_match_primary'].sum()/n*100:.1f}%",
        })

    return pd.DataFrame(results)

# ============================================================================
# TABLE GENERATION: DECISION SHIFTS
# ============================================================================

def table_decision_shifts_pooled(deltas: pd.DataFrame) -> pd.DataFrame:
    """T5: Decision shifts by dimension × level (pooled across models)."""
    results = []

    baseline_df = deltas[[col for col in deltas.columns if col.endswith("_base") and
                          col.replace("_base", "") in ["refer_match", "urgency_match", "imaging_match"]]].iloc[:, 0:1]

    for _, row in deltas.iterrows():
        dim = row.get("dimension")
        level = row.get("level")

        if pd.isna(dim) or dim == "baseline":
            continue

        # Track change in key metrics
        refer_delta = row.get("refer_match_delta", 0)
        urgency_delta = row.get("urgency_match_delta", 0)
        imaging_delta = row.get("imaging_match_delta", 0)

        results.append({
            "Dimension": DIM_LABELS.get(dim, dim),
            "Level": LEVEL_LABELS.get((dim, level), level),
            "Refer Change": "↓" if refer_delta < 0 else ("↑" if refer_delta > 0 else "→"),
            "Urgency Change": "↓" if urgency_delta < 0 else ("↑" if urgency_delta > 0 else "→"),
            "Imaging Change": "↓" if imaging_delta < 0 else ("↑" if imaging_delta > 0 else "→"),
            "Composite Δ": f"{row.get('composite_delta', 0):.3f}",
        })

    result_df = pd.DataFrame(results)
    if len(result_df) > 0:
        result_df = result_df.drop_duplicates().sort_values("Dimension")
    return result_df

def table_decision_shifts_by_model(deltas: pd.DataFrame) -> pd.DataFrame:
    """T6: Decision shifts by model × dimension."""
    results = []

    for model in sorted(deltas["model"].unique()):
        model_deltas = deltas[deltas["model"] == model]
        provider = model_deltas["provider"].iloc[0] if len(model_deltas) > 0 else "Unknown"

        for dim in sorted(model_deltas["dimension"].dropna().unique()):
            if dim == "baseline":
                continue

            dim_data = model_deltas[model_deltas["dimension"] == dim]

            refer_changes = dim_data["refer_match_delta"].dropna()
            urgency_changes = dim_data["urgency_match_delta"].dropna()

            results.append({
                "Model": model,
                "Provider": provider,
                "Dimension": DIM_LABELS.get(dim, dim),
                "N": len(dim_data),
                "Avg Refer Δ": f"{refer_changes.mean():.3f}" if len(refer_changes) > 0 else "N/A",
                "Avg Urgency Δ": f"{urgency_changes.mean():.3f}" if len(urgency_changes) > 0 else "N/A",
            })

    return pd.DataFrame(results)

def table_decision_shifts_by_persona(deltas: pd.DataFrame) -> pd.DataFrame:
    """T7: Decision shifts by persona × dimension."""
    results = []

    for persona in sorted(deltas["persona"].unique()):
        persona_deltas = deltas[deltas["persona"] == persona]

        for dim in sorted(persona_deltas["dimension"].dropna().unique()):
            if dim == "baseline":
                continue

            dim_data = persona_deltas[persona_deltas["dimension"] == dim]
            refer_changes = dim_data["refer_match_delta"].dropna()

            results.append({
                "Persona": PERSONA_LABELS.get(persona, persona),
                "Dimension": DIM_LABELS.get(dim, dim),
                "N": len(dim_data),
                "Refer Change %": f"{refer_changes.mean()*100:.1f}%" if len(refer_changes) > 0 else "N/A",
            })

    return pd.DataFrame(results)

def table_decision_shifts_by_provider(deltas: pd.DataFrame) -> pd.DataFrame:
    """T8: Decision shifts by provider × dimension."""
    results = []

    for provider in sorted(deltas["provider"].unique()):
        provider_deltas = deltas[deltas["provider"] == provider]

        for dim in sorted(provider_deltas["dimension"].dropna().unique()):
            if dim == "baseline":
                continue

            dim_data = provider_deltas[provider_deltas["dimension"] == dim]
            refer_changes = dim_data["refer_match_delta"].dropna()

            results.append({
                "Provider": provider,
                "Dimension": DIM_LABELS.get(dim, dim),
                "N": len(dim_data),
                "Avg Refer Change": f"{refer_changes.mean():.3f}" if len(refer_changes) > 0 else "N/A",
            })

    return pd.DataFrame(results)

# ============================================================================
# TABLE GENERATION: PSYCHOLOGIZATION & URGENCY
# ============================================================================

def table_psychologization_rates(df: pd.DataFrame, deltas: pd.DataFrame) -> pd.DataFrame:
    """T9: Psychologization rates by dimension × level."""
    baseline_df = df[df["condition"] == "baseline"].copy()
    baseline_psych_rate = baseline_df["psychologized"].mean()

    results = []

    for dim in sorted(deltas["dimension"].dropna().unique()):
        if dim == "baseline":
            continue

        dim_data = deltas[deltas["dimension"] == dim]

        for level in sorted(dim_data["level"].dropna().unique()):
            level_data = dim_data[dim_data["level"] == level]

            psych_count = (level_data["psychologized_iter"] == 1).sum()
            total = len(level_data)
            psych_rate = psych_count / total if total > 0 else 0
            delta = psych_rate - baseline_psych_rate

            results.append({
                "Dimension": DIM_LABELS.get(dim, dim),
                "Level": LEVEL_LABELS.get((dim, level), level),
                "N": total,
                "Psychologized %": f"{psych_rate*100:.1f}%",
                "Δ from Baseline": f"{delta*100:+.1f}%",
            })

    return pd.DataFrame(results)

def table_urgency_direction(deltas: pd.DataFrame) -> pd.DataFrame:
    """T10: Urgency direction (downgraded/correct/upgraded) by dimension × level."""
    results = []

    for dim in sorted(deltas["dimension"].dropna().unique()):
        if dim == "baseline":
            continue

        dim_data = deltas[deltas["dimension"] == dim]

        for level in sorted(dim_data["level"].dropna().unique()):
            level_data = dim_data[dim_data["level"] == level]

            downgraded = (level_data["urgency_downgraded_iter"] == 1).sum()
            upgraded = (level_data["urgency_upgraded_iter"] == 1).sum()
            correct = len(level_data) - downgraded - upgraded
            total = len(level_data)

            results.append({
                "Dimension": DIM_LABELS.get(dim, dim),
                "Level": LEVEL_LABELS.get((dim, level), level),
                "Downgraded": f"{downgraded/total*100:.1f}%" if total > 0 else "N/A",
                "Correct": f"{correct/total*100:.1f}%" if total > 0 else "N/A",
                "Upgraded": f"{upgraded/total*100:.1f}%" if total > 0 else "N/A",
            })

    return pd.DataFrame(results)

# ============================================================================
# TABLE GENERATION: COMPOSITE DELTAS & STATISTICAL TESTS
# ============================================================================

def table_composite_deltas(deltas: pd.DataFrame) -> pd.DataFrame:
    """T11: Composite score deltas by dimension × level with paired t-tests."""
    results = []

    for dim in sorted(deltas["dimension"].dropna().unique()):
        if dim == "baseline":
            continue

        dim_data = deltas[deltas["dimension"] == dim]

        for level in sorted(dim_data["level"].dropna().unique()):
            level_data = dim_data[dim_data["level"] == level]

            deltas_list = level_data["composite_delta"].dropna()
            if len(deltas_list) == 0:
                continue

            mean_delta = deltas_list.mean()
            sd_delta = deltas_list.std()
            se = sd_delta / np.sqrt(len(deltas_list))
            ci_lower = mean_delta - 1.96 * se
            ci_upper = mean_delta + 1.96 * se

            # Paired t-test against 0
            t_stat, p_val = stats.ttest_1samp(deltas_list, 0)

            results.append({
                "Dimension": DIM_LABELS.get(dim, dim),
                "Level": LEVEL_LABELS.get((dim, level), level),
                "N": len(deltas_list),
                "Mean Δ": f"{mean_delta:.4f}",
                "SD": f"{sd_delta:.4f}",
                "95% CI": f"[{ci_lower:.4f}, {ci_upper:.4f}]",
                "t-statistic": f"{t_stat:.3f}",
                "p-value": f"{p_val:.4f}",
                "Significant": "Yes" if p_val < ALPHA else "No",
            })

    return pd.DataFrame(results)

def table_statistical_tests_master(df: pd.DataFrame, deltas: pd.DataFrame) -> pd.DataFrame:
    """T12: Master statistical tests table with binomtest, FDR correction, Cohen's h."""
    baseline_df = df[df["condition"] == "baseline"].copy()

    all_tests = []

    for dim in sorted(deltas["dimension"].dropna().unique()):
        if dim == "baseline":
            continue

        dim_data = deltas[deltas["dimension"] == dim]

        for level in sorted(dim_data["level"].dropna().unique()):
            level_data = dim_data[dim_data["level"] == level]

            # Test for refer_match change
            refer_baseline = baseline_df["refer_match"].mean()
            refer_count = (level_data["refer_match_iter"] == 1).sum()
            refer_total = len(level_data)
            refer_rate = refer_count / refer_total if refer_total > 0 else 0

            if refer_total > 0:
                try:
                    binom_result = binomtest(refer_count, refer_total, refer_baseline, alternative="two-sided")
                    all_tests.append({
                        "Dimension": DIM_LABELS.get(dim, dim),
                        "Level": LEVEL_LABELS.get((dim, level), level),
                        "Metric": "Referral Match",
                        "N": refer_total,
                        "Count": refer_count,
                        "Rate": f"{refer_rate*100:.1f}%",
                        "Baseline": f"{refer_baseline*100:.1f}%",
                        "p-value": binom_result.pvalue,
                        "Cohen's h": cohens_h(refer_rate, refer_baseline),
                    })
                except:
                    pass

            # Test for urgency_match change
            urgency_baseline = baseline_df["urgency_match"].mean()
            urgency_count = (level_data["urgency_match_iter"] == 1).sum()
            urgency_total = len(level_data)
            urgency_rate = urgency_count / urgency_total if urgency_total > 0 else 0

            if urgency_total > 0:
                try:
                    binom_result = binomtest(urgency_count, urgency_total, urgency_baseline, alternative="two-sided")
                    all_tests.append({
                        "Dimension": DIM_LABELS.get(dim, dim),
                        "Level": LEVEL_LABELS.get((dim, level), level),
                        "Metric": "Urgency Match",
                        "N": urgency_total,
                        "Count": urgency_count,
                        "Rate": f"{urgency_rate*100:.1f}%",
                        "Baseline": f"{urgency_baseline*100:.1f}%",
                        "p-value": binom_result.pvalue,
                        "Cohen's h": cohens_h(urgency_rate, urgency_baseline),
                    })
                except:
                    pass

    result_df = pd.DataFrame(all_tests)

    # Apply FDR correction to p-values
    if len(result_df) > 0 and "p-value" in result_df.columns:
        pvals = result_df["p-value"].dropna().values
        if len(pvals) > 0:
            _, fdr_corrected = apply_fdr_correction(pvals.tolist())
            fdr_col = np.full(len(result_df), np.nan)
            fdr_col[:len(fdr_corrected)] = fdr_corrected
            result_df["FDR-corrected p"] = fdr_col
            result_df["Significant (FDR)"] = result_df["FDR-corrected p"] < ALPHA

    return result_df

# ============================================================================
# TABLE GENERATION: RANKINGS
# ============================================================================

def table_model_ranking(df: pd.DataFrame) -> pd.DataFrame:
    """T13: Model ranking by overall composite, referral, urgency, psychologization, susceptibility."""
    baseline_df = df[df["condition"] == "baseline"].copy()

    results = []
    for model in sorted(baseline_df["model"].unique()):
        model_data = baseline_df[baseline_df["model"] == model]
        provider = model_data["provider"].iloc[0] if len(model_data) > 0 else "Unknown"

        composite = model_data["composite_score"].mean()
        referral = model_data["refer_match"].mean()
        urgency = model_data["urgency_match"].mean()
        psych = model_data["psychologized"].mean()

        results.append({
            "Model": model,
            "Provider": provider,
            "Composite Score": f"{composite:.4f}",
            "Referral Accuracy": f"{referral*100:.1f}%",
            "Urgency Accuracy": f"{urgency*100:.1f}%",
            "Psychologization %": f"{psych*100:.1f}%",
        })

    result_df = pd.DataFrame(results)
    # Sort by composite score descending
    result_df["Composite Numeric"] = pd.to_numeric(result_df["Composite Score"])
    result_df = result_df.sort_values("Composite Numeric", ascending=False).drop("Composite Numeric", axis=1)
    result_df.insert(0, "Rank", range(1, len(result_df) + 1))

    return result_df

def table_dimension_ranking(deltas: pd.DataFrame) -> pd.DataFrame:
    """T14: Dimension ranking by overall shift rates."""
    results = []

    for dim in sorted(deltas["dimension"].dropna().unique()):
        if dim == "baseline":
            continue

        dim_data = deltas[deltas["dimension"] == dim]

        refer_changes = dim_data["refer_match_delta"].dropna()
        urgency_changes = dim_data["urgency_match_delta"].dropna()

        avg_shift = np.mean([abs(refer_changes.mean()) if len(refer_changes) > 0 else 0,
                             abs(urgency_changes.mean()) if len(urgency_changes) > 0 else 0])

        # Assign to dimension group
        dim_group = None
        for group, dims in DIM_GROUPS.items():
            if dim in dims:
                dim_group = group
                break

        results.append({
            "Dimension": DIM_LABELS.get(dim, dim),
            "Group": dim_group,
            "Mean |Shift|": f"{avg_shift:.4f}",
        })

    result_df = pd.DataFrame(results)
    result_df["Mean Numeric"] = pd.to_numeric(result_df["Mean |Shift|"])
    result_df = result_df.sort_values("Mean Numeric", ascending=False)
    result_df.insert(0, "Rank", range(1, len(result_df) + 1))
    result_df = result_df.drop("Mean Numeric", axis=1)

    return result_df

def table_pairwise_comparisons(deltas: pd.DataFrame) -> pd.DataFrame:
    """T15: Pairwise within-dimension comparisons (e.g., Black vs White vs Hispanic)."""
    results = []

    # Focus on race, ses, tone, anchoring dimensions which have multiple levels
    pairwise_dims = ["race", "ses", "tone", "anchoring"]

    for dim in pairwise_dims:
        dim_data = deltas[deltas["dimension"] == dim]
        levels = sorted(dim_data["level"].dropna().unique())

        if len(levels) < 2:
            continue

        for i, level1 in enumerate(levels):
            for level2 in levels[i+1:]:
                l1_data = dim_data[dim_data["level"] == level1]
                l2_data = dim_data[dim_data["level"] == level2]

                # Chi-square or Fisher's exact for refer_match
                try:
                    refer1 = (l1_data["refer_match_iter"] == 1).sum()
                    refer2 = (l2_data["refer_match_iter"] == 1).sum()
                    total1 = len(l1_data)
                    total2 = len(l2_data)

                    if total1 > 0 and total2 > 0:
                        contingency = np.array([[refer1, total1 - refer1],
                                               [refer2, total2 - refer2]])
                        try:
                            chi2, p_val, dof, expected = chi2_contingency(contingency)
                        except:
                            p_val = np.nan

                        results.append({
                            "Dimension": DIM_LABELS.get(dim, dim),
                            "Level 1": LEVEL_LABELS.get((dim, level1), level1),
                            "Level 2": LEVEL_LABELS.get((dim, level2), level2),
                            "Rate 1 %": f"{refer1/total1*100:.1f}%" if total1 > 0 else "N/A",
                            "Rate 2 %": f"{refer2/total2*100:.1f}%" if total2 > 0 else "N/A",
                            "p-value": f"{p_val:.4f}" if not np.isnan(p_val) else "N/A",
                        })
                except:
                    pass

    return pd.DataFrame(results)

# ============================================================================
# FIGURE GENERATION: BASELINE ACCURACY & DECISION CHANGES
# ============================================================================

def fig_baseline_accuracy_pooled(df: pd.DataFrame, output_dir: Path):
    """fig01: Baseline accuracy bar chart with Wilson CI (pooled)."""
    _style()
    baseline_df = df[df["condition"] == "baseline"].copy()

    metrics = ["refer_match", "urgency_match", "imaging_match", "dx_match_primary", "psychologized"]
    metric_labels = ["Referral\nAccuracy", "Urgency\nAccuracy", "Imaging\nAccuracy", "Dx\nAccuracy", "Psychologized"]

    rates = []
    cis_lower = []
    cis_upper = []

    for metric in metrics:
        successes = int(baseline_df[metric].sum())
        total = len(baseline_df)
        rate = successes / total if total > 0 else 0
        lower, upper = wilson_ci(successes, total)
        rates.append(rate)
        cis_lower.append(rate - lower)
        cis_upper.append(upper - rate)

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(metrics))
    bars = ax.bar(x, rates, color=[BLUE, TEAL, AMBER, GREEN, RED], alpha=0.8, width=0.6)
    ax.errorbar(x, rates, yerr=[cis_lower, cis_upper], fmt="none", ecolor=TXT, capsize=5, linewidth=2)

    ax.set_ylabel("Accuracy / Rate", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=11)
    ax.set_ylim([0, 1.0])
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.grid(axis="y", alpha=0.3)

    _wm(ax)
    _header(fig, "Baseline Clinical Accuracy", "All Models Pooled (n={})".format(len(baseline_df)))

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(output_dir / "fig01_baseline_accuracy_pooled.png", dpi=DPI, bbox_inches="tight")
    plt.close()
    print("✓ fig01_baseline_accuracy_pooled.png")

def fig_baseline_by_model(df: pd.DataFrame, output_dir: Path):
    """fig02: Baseline accuracy by model."""
    _style()
    baseline_df = df[df["condition"] == "baseline"].copy()

    models = sorted(baseline_df["model"].unique())
    composites = [baseline_df[baseline_df["model"] == m]["composite_score"].mean() for m in models]
    providers = [baseline_df[baseline_df["model"] == m]["provider"].iloc[0] for m in models]

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = [PROVIDER_COLORS.get(p, SLATE) for p in providers]
    bars = ax.barh(range(len(models)), composites, color=colors, alpha=0.8)

    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=10)
    ax.set_xlabel("Composite Score", fontsize=12, fontweight="bold")
    ax.set_xlim([0, 1.0])
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.grid(axis="x", alpha=0.3)

    # Legend for providers
    legend_patches = [mpatches.Patch(color=PROVIDER_COLORS.get(p, SLATE), label=p)
                     for p in sorted(set(providers))]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=10)

    _wm(ax)
    _header(fig, "Baseline Accuracy by Model", "Sorted by Provider")

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(output_dir / "fig02_baseline_by_model.png", dpi=DPI, bbox_inches="tight")
    plt.close()
    print("✓ fig02_baseline_by_model.png")

def fig_decision_change_heatmap_pooled(deltas: pd.DataFrame, output_dir: Path):
    """fig03: Decision-change heatmap (dimension × level, urgency_match change rate) pooled."""
    _style()

    # Build a tidy table of dim_label × level_label → urgency change rate
    records = []
    for dim in deltas["dimension"].dropna().unique():
        if dim == "baseline":
            continue
        dim_data = deltas[deltas["dimension"] == dim]
        for level in dim_data["level"].dropna().unique():
            level_data = dim_data[dim_data["level"] == level]
            if "urgency_match_changed" in level_data.columns:
                rate = level_data["urgency_match_changed"].mean() * 100
            else:
                rate = level_data["urgency_match_delta"].abs().mean() * 100 if "urgency_match_delta" in level_data.columns else 0
            records.append({
                "Dimension": DIM_LABELS.get(dim, dim),
                "Level": LEVEL_LABELS.get((dim, level), level),
                "Rate": rate,
                "dim_order": dim_sort_key(dim),
            })

    if not records:
        plt.close("all")
        print("⚠ fig03: No data for heatmap")
        return

    rdf = pd.DataFrame(records).sort_values("dim_order")

    # Use a horizontal bar chart instead of heatmap for ragged data
    fig, ax = plt.subplots(figsize=(12, max(6, len(records) * 0.35)))
    labels = [f"{r['Dimension']} — {r['Level']}" for _, r in rdf.iterrows()]
    colors = [DIM_COLORS.get(d, SLATE) for d in deltas["dimension"].dropna().unique() for _ in range(1)]
    bar_colors = []
    for _, r in rdf.iterrows():
        dim_raw = [k for k, v in DIM_LABELS.items() if v == r["Dimension"]]
        bar_colors.append(DIM_COLORS.get(dim_raw[0], SLATE) if dim_raw else SLATE)

    bars = ax.barh(range(len(rdf)), rdf["Rate"].values, color=bar_colors, alpha=0.8, height=0.7)
    ax.set_yticks(range(len(rdf)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Urgency Decision Change Rate (%)", fontsize=12, fontweight="bold")
    ax.invert_yaxis()

    # Add value labels
    for i, v in enumerate(rdf["Rate"].values):
        ax.text(v + 0.2, i, f"{v:.1f}%", va="center", fontsize=8, color=TXT)

    _wm(ax)
    _header(fig, "Decision Change Rates (Urgency)", "Urgency accuracy change rate by dimension × level (all models pooled)")

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(output_dir / "fig03_decision_heatmap_pooled.png", dpi=DPI, bbox_inches="tight")
    plt.close()
    print("✓ fig03_decision_heatmap_pooled.png")

def fig_decision_change_heatmap_by_model(deltas: pd.DataFrame, output_dir: Path):
    """fig04: Decision-change by model — faceted horizontal bar charts."""
    _style()

    models = sorted(deltas["model"].unique())
    n_models = len(models)
    n_cols = min(3, n_models)
    n_rows = max(1, (n_models + n_cols - 1) // n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    if n_models == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes).flatten()

    for idx, model in enumerate(models):
        ax = axes[idx]
        model_deltas = deltas[deltas["model"] == model]

        records = []
        for dim in model_deltas["dimension"].dropna().unique():
            if dim == "baseline":
                continue
            dim_data = model_deltas[model_deltas["dimension"] == dim]
            for level in dim_data["level"].dropna().unique():
                level_data = dim_data[dim_data["level"] == level]
                if "urgency_match_changed" in level_data.columns:
                    rate = level_data["urgency_match_changed"].mean() * 100
                else:
                    rate = level_data["urgency_match_delta"].abs().mean() * 100
                records.append({
                    "label": f"{DIM_LABELS.get(dim, dim)[:12]}—{LEVEL_LABELS.get((dim, level), level)[:15]}",
                    "rate": rate,
                    "dim_order": dim_sort_key(dim),
                })

        if records:
            rdf = pd.DataFrame(records).sort_values("dim_order")
            bar_y = range(len(rdf))
            ax.barh(bar_y, rdf["rate"].values, color=BLUE, alpha=0.7, height=0.7)
            ax.set_yticks(bar_y)
            ax.set_yticklabels(rdf["label"].values, fontsize=7)
            ax.invert_yaxis()
            ax.set_xlabel("Urgency Change %", fontsize=9)

        ax.set_title(model, fontsize=10, fontweight="bold")

    for idx in range(len(models), len(axes)):
        axes[idx].axis("off")

    _header(fig, "Decision Changes by Model", "Urgency accuracy change rate per dimension × level")
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(output_dir / "fig04_decision_heatmap_by_model.png", dpi=DPI, bbox_inches="tight")
    plt.close()
    print("✓ fig04_decision_heatmap_by_model.png")

def fig_referral_urgency_changes(deltas: pd.DataFrame, output_dir: Path):
    """fig05: Referral & urgency change rates paired bars by dimension."""
    _style()

    dims_list = sorted([d for d in deltas["dimension"].dropna().unique() if d != "baseline"],
                       key=dim_sort_key)

    refer_means = []
    urgency_means = []
    dim_labels_plot = []

    for dim in dims_list:
        dim_data = deltas[deltas["dimension"] == dim]
        refer_changes = dim_data["refer_match_delta"].dropna()
        urgency_changes = dim_data["urgency_match_delta"].dropna()

        refer_means.append(refer_changes.mean() if len(refer_changes) > 0 else 0)
        urgency_means.append(urgency_changes.mean() if len(urgency_changes) > 0 else 0)
        dim_labels_plot.append(DIM_LABELS.get(dim, dim))

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(dims_list))
    width = 0.35

    bars1 = ax.bar(x - width/2, refer_means, width, label="Referral", color=BLUE, alpha=0.8)
    bars2 = ax.bar(x + width/2, urgency_means, width, label="Urgency", color=TEAL, alpha=0.8)

    ax.axhline(y=0, color=TXT, linestyle="-", linewidth=0.8)
    ax.set_ylabel("Change in Match Rate", fontsize=12, fontweight="bold")
    ax.set_xlabel("Dimension", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(dim_labels_plot, fontsize=11, rotation=45, ha="right")
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    _wm(ax)
    _header(fig, "Referral & Urgency Decision Changes", "By Dimension (Pooled Models)")

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(output_dir / "fig05_referral_urgency_changes.png", dpi=DPI, bbox_inches="tight")
    plt.close()
    print("✓ fig05_referral_urgency_changes.png")

def fig_psychologization_by_dimension(df: pd.DataFrame, deltas: pd.DataFrame, output_dir: Path):
    """fig06: Psychologization rates by dimension (bar + delta from baseline)."""
    _style()

    baseline_df = df[df["condition"] == "baseline"].copy()
    baseline_psych = baseline_df["psychologized"].mean()

    dims_list = sorted([d for d in deltas["dimension"].dropna().unique() if d != "baseline"],
                       key=dim_sort_key)

    psych_rates = []
    deltas_psych = []
    dim_labels_plot = []

    for dim in dims_list:
        dim_data = deltas[deltas["dimension"] == dim]
        psych_count = (dim_data["psychologized_iter"] == 1).sum()
        total = len(dim_data)
        psych_rate = psych_count / total if total > 0 else 0

        psych_rates.append(psych_rate)
        deltas_psych.append(psych_rate - baseline_psych)
        dim_labels_plot.append(DIM_LABELS.get(dim, dim))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Absolute rates
    colors_psych = [RED if d > baseline_psych else GREEN for d in psych_rates]
    ax1.barh(range(len(dims_list)), psych_rates, color=colors_psych, alpha=0.8)
    ax1.axvline(x=baseline_psych, color=TXT, linestyle="--", linewidth=2, label="Baseline")
    ax1.set_yticks(range(len(dims_list)))
    ax1.set_yticklabels(dim_labels_plot, fontsize=11)
    ax1.set_xlabel("Psychologization Rate", fontsize=12, fontweight="bold")
    ax1.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax1.legend(fontsize=10)
    ax1.grid(axis="x", alpha=0.3)

    # Right: Delta from baseline
    colors_delta = [RED if d > 0 else GREEN for d in deltas_psych]
    ax2.barh(range(len(dims_list)), deltas_psych, color=colors_delta, alpha=0.8)
    ax2.axvline(x=0, color=TXT, linestyle="-", linewidth=0.8)
    ax2.set_yticks(range(len(dims_list)))
    ax2.set_yticklabels(dim_labels_plot, fontsize=11)
    ax2.set_xlabel("Δ from Baseline", fontsize=12, fontweight="bold")
    ax2.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax2.grid(axis="x", alpha=0.3)

    _wm(ax1)
    _wm(ax2)
    _header(fig, "Psychologization Rates by Dimension", "Baseline: {:.1f}%".format(baseline_psych * 100))

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(output_dir / "fig06_psychologization_by_dimension.png", dpi=DPI, bbox_inches="tight")
    plt.close()
    print("✓ fig06_psychologization_by_dimension.png")

def fig_urgency_direction_stacked(deltas: pd.DataFrame, output_dir: Path):
    """fig07: Urgency direction stacked bar (down/correct/up by dimension)."""
    _style()

    dims_list = sorted([d for d in deltas["dimension"].dropna().unique() if d != "baseline"],
                       key=dim_sort_key)

    downgraded_rates = []
    correct_rates = []
    upgraded_rates = []
    dim_labels_plot = []

    for dim in dims_list:
        dim_data = deltas[deltas["dimension"] == dim]

        downgraded = (dim_data["urgency_downgraded_iter"] == 1).sum()
        upgraded = (dim_data["urgency_upgraded_iter"] == 1).sum()
        correct = len(dim_data) - downgraded - upgraded
        total = len(dim_data)

        downgraded_rates.append(downgraded / total if total > 0 else 0)
        correct_rates.append(correct / total if total > 0 else 0)
        upgraded_rates.append(upgraded / total if total > 0 else 0)
        dim_labels_plot.append(DIM_LABELS.get(dim, dim))

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(dims_list))

    ax.bar(x, downgraded_rates, label="Downgraded", color=RED, alpha=0.8)
    ax.bar(x, correct_rates, bottom=downgraded_rates, label="Correct", color=GREEN, alpha=0.8)
    ax.bar(x, upgraded_rates, bottom=np.array(downgraded_rates) + np.array(correct_rates),
           label="Upgraded", color=AMBER, alpha=0.8)

    ax.set_ylabel("Proportion", fontsize=12, fontweight="bold")
    ax.set_xlabel("Dimension", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(dim_labels_plot, fontsize=11, rotation=45, ha="right")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.legend(fontsize=11, loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    _wm(ax)
    _header(fig, "Urgency Direction Changes", "By Dimension (Pooled Models)")

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(output_dir / "fig07_urgency_direction_stacked.png", dpi=DPI, bbox_inches="tight")
    plt.close()
    print("✓ fig07_urgency_direction_stacked.png")

def fig_composite_delta_diverging(deltas: pd.DataFrame, output_dir: Path):
    """fig08: Composite delta diverging bar by dimension × level."""
    _style()

    dims_list = sorted([d for d in deltas["dimension"].dropna().unique() if d != "baseline"],
                       key=dim_sort_key)

    all_deltas_data = []

    for dim in dims_list:
        dim_data = deltas[deltas["dimension"] == dim]
        levels = sorted(dim_data["level"].dropna().unique())

        for level in levels:
            level_data = dim_data[dim_data["level"] == level]
            composite_deltas = level_data["composite_delta"].dropna()
            mean_delta = composite_deltas.mean() if len(composite_deltas) > 0 else 0

            all_deltas_data.append({
                "label": "{} - {}".format(DIM_LABELS.get(dim, dim), LEVEL_LABELS.get((dim, level), level)),
                "delta": mean_delta,
            })

    # Sort by delta
    all_deltas_data = sorted(all_deltas_data, key=lambda x: x["delta"])

    labels_plot = [d["label"] for d in all_deltas_data[-15:]]  # Top 15 to keep fig manageable
    deltas_plot = [d["delta"] for d in all_deltas_data[-15:]]

    colors_div = [RED if d < 0 else GREEN for d in deltas_plot]

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.barh(range(len(deltas_plot)), deltas_plot, color=colors_div, alpha=0.8)
    ax.axvline(x=0, color=TXT, linestyle="-", linewidth=0.8)

    ax.set_yticks(range(len(deltas_plot)))
    ax.set_yticklabels(labels_plot, fontsize=10)
    ax.set_xlabel("Composite Score Change", fontsize=12, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)

    _wm(ax)
    _header(fig, "Composite Score Changes", "Dimension × Level (Top 15 by Magnitude)")

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(output_dir / "fig08_composite_delta_diverging.png", dpi=DPI, bbox_inches="tight")
    plt.close()
    print("✓ fig08_composite_delta_diverging.png")

# ============================================================================
# FIGURE GENERATION: MODEL & PROVIDER COMPARISONS
# ============================================================================

def fig_model_susceptibility_scatter(df: pd.DataFrame, deltas: pd.DataFrame, output_dir: Path):
    """fig09: Model susceptibility scatter (x=baseline composite, y=mean shift rate)."""
    _style()

    baseline_df = df[df["condition"] == "baseline"].copy()

    model_stats = []
    for model in sorted(baseline_df["model"].unique()):
        model_baseline = baseline_df[baseline_df["model"] == model]
        model_deltas = deltas[deltas["model"] == model]

        composite = model_baseline["composite_score"].mean()

        refer_shifts = model_deltas["refer_match_delta"].dropna()
        urgency_shifts = model_deltas["urgency_match_delta"].dropna()

        mean_shift = np.mean([abs(refer_shifts.mean()) if len(refer_shifts) > 0 else 0,
                             abs(urgency_shifts.mean()) if len(urgency_shifts) > 0 else 0])

        provider = model_baseline["provider"].iloc[0] if len(model_baseline) > 0 else "Unknown"

        model_stats.append({
            "model": model,
            "composite": composite,
            "shift": mean_shift,
            "provider": provider,
        })

    fig, ax = plt.subplots(figsize=(10, 7))

    for provider, color in PROVIDER_COLORS.items():
        provider_stats = [m for m in model_stats if m["provider"] == provider]
        if provider_stats:
            composites = [m["composite"] for m in provider_stats]
            shifts = [m["shift"] for m in provider_stats]
            ax.scatter(composites, shifts, s=150, color=color, alpha=0.7, label=provider, edgecolors=TXT, linewidth=1.5)

            for m in provider_stats:
                ax.annotate(m["model"], (m["composite"], m["shift"]), fontsize=8, ha="center", va="bottom")

    ax.set_xlabel("Baseline Composite Score", fontsize=12, fontweight="bold")
    ax.set_ylabel("Mean |Shift Rate| (Refer + Urgency)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=11, loc="best")
    ax.grid(True, alpha=0.3)

    _wm(ax)
    _header(fig, "Model Susceptibility to Bias", "Baseline Accuracy vs. Decision Shift Magnitude")

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(output_dir / "fig09_model_susceptibility_scatter.png", dpi=DPI, bbox_inches="tight")
    plt.close()
    print("✓ fig09_model_susceptibility_scatter.png")

def fig_model_dimension_heatmap(deltas: pd.DataFrame, output_dir: Path):
    """fig10: Model × dimension heatmap (mean composite delta)."""
    _style()

    models = sorted(deltas["model"].unique())
    dims_list = sorted([d for d in deltas["dimension"].dropna().unique() if d != "baseline"],
                       key=dim_sort_key)

    heatmap_data = []
    for model in models:
        row = []
        model_deltas = deltas[deltas["model"] == model]
        for dim in dims_list:
            dim_data = model_deltas[model_deltas["dimension"] == dim]
            comp_deltas = dim_data["composite_delta"].dropna()
            mean_delta = comp_deltas.mean() if len(comp_deltas) > 0 else 0
            row.append(mean_delta)
        heatmap_data.append(row)

    fig, ax = plt.subplots(figsize=(12, 8))
    heatmap_array = np.array(heatmap_data, dtype=float)

    im = ax.imshow(heatmap_array, cmap="RdBu_r", aspect="auto", vmin=-0.2, vmax=0.2)

    ax.set_xticks(range(len(dims_list)))
    ax.set_yticks(range(len(models)))
    ax.set_xticklabels([DIM_LABELS.get(d, d) for d in dims_list], fontsize=11, rotation=45, ha="right")
    ax.set_yticklabels(models, fontsize=11)

    ax.set_xlabel("Dimension", fontsize=12, fontweight="bold")
    ax.set_ylabel("Model", fontsize=12, fontweight="bold")

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Mean Composite Δ", fontsize=11, fontweight="bold")

    _wm(ax)
    _header(fig, "Model × Dimension Composite Score Changes", "Heatmap of Bias Impact")

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(output_dir / "fig10_model_dimension_heatmap.png", dpi=DPI, bbox_inches="tight")
    plt.close()
    print("✓ fig10_model_dimension_heatmap.png")

def fig_provider_comparison(df: pd.DataFrame, deltas: pd.DataFrame, output_dir: Path):
    """fig11: Provider comparison (grouped bars)."""
    _style()

    baseline_df = df[df["condition"] == "baseline"].copy()
    providers = sorted(baseline_df["provider"].unique())

    metrics_data = []
    for provider in providers:
        prov_baseline = baseline_df[baseline_df["provider"] == provider]
        prov_deltas = deltas[deltas["provider"] == provider]

        composite = prov_baseline["composite_score"].mean()
        referral = prov_baseline["refer_match"].mean()
        urgency = prov_baseline["urgency_match"].mean()

        refer_shifts = prov_deltas["refer_match_delta"].dropna()
        urgency_shifts = prov_deltas["urgency_match_delta"].dropna()
        mean_shift = np.mean([abs(refer_shifts.mean()) if len(refer_shifts) > 0 else 0,
                             abs(urgency_shifts.mean()) if len(urgency_shifts) > 0 else 0])

        metrics_data.append({
            "provider": provider,
            "composite": composite,
            "referral": referral,
            "urgency": urgency,
            "shift": mean_shift,
        })

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Composite score
    ax = axes[0, 0]
    composites = [m["composite"] for m in metrics_data]
    ax.bar(range(len(providers)), composites, color=[PROVIDER_COLORS.get(p, SLATE) for p in providers], alpha=0.8)
    ax.set_xticks(range(len(providers)))
    ax.set_xticklabels(providers, fontsize=11)
    ax.set_ylabel("Composite Score", fontsize=11, fontweight="bold")
    ax.set_ylim([0, 1])
    ax.grid(axis="y", alpha=0.3)

    # Referral accuracy
    ax = axes[0, 1]
    referrals = [m["referral"] for m in metrics_data]
    ax.bar(range(len(providers)), referrals, color=[PROVIDER_COLORS.get(p, SLATE) for p in providers], alpha=0.8)
    ax.set_xticks(range(len(providers)))
    ax.set_xticklabels(providers, fontsize=11)
    ax.set_ylabel("Referral Accuracy", fontsize=11, fontweight="bold")
    ax.set_ylim([0, 1])
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.grid(axis="y", alpha=0.3)

    # Urgency accuracy
    ax = axes[1, 0]
    urgencies = [m["urgency"] for m in metrics_data]
    ax.bar(range(len(providers)), urgencies, color=[PROVIDER_COLORS.get(p, SLATE) for p in providers], alpha=0.8)
    ax.set_xticks(range(len(providers)))
    ax.set_xticklabels(providers, fontsize=11)
    ax.set_ylabel("Urgency Accuracy", fontsize=11, fontweight="bold")
    ax.set_ylim([0, 1])
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.grid(axis="y", alpha=0.3)

    # Susceptibility
    ax = axes[1, 1]
    shifts = [m["shift"] for m in metrics_data]
    ax.bar(range(len(providers)), shifts, color=[PROVIDER_COLORS.get(p, SLATE) for p in providers], alpha=0.8)
    ax.set_xticks(range(len(providers)))
    ax.set_xticklabels(providers, fontsize=11)
    ax.set_ylabel("Mean |Shift|", fontsize=11, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    _wm(axes[0, 0])
    _header(fig, "Provider Comparison", "OpenAI vs. Anthropic vs. Google")

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(output_dir / "fig11_provider_comparison.png", dpi=DPI, bbox_inches="tight")
    plt.close()
    print("✓ fig11_provider_comparison.png")

def fig_persona_susceptibility(deltas: pd.DataFrame, output_dir: Path):
    """fig12: Persona susceptibility (mean shift rate per persona with CI)."""
    _style()

    personas = sorted(deltas["persona"].unique())

    persona_shifts = []
    for persona in personas:
        persona_deltas = deltas[deltas["persona"] == persona]

        refer_shifts = persona_deltas["refer_match_delta"].dropna()
        urgency_shifts = persona_deltas["urgency_match_delta"].dropna()

        refer_mean = abs(refer_shifts.mean()) if len(refer_shifts) > 0 else 0
        urgency_mean = abs(urgency_shifts.mean()) if len(urgency_shifts) > 0 else 0

        mean_shift = np.mean([refer_mean, urgency_mean])

        # Compute CI (normal approximation)
        all_shifts = []
        if len(refer_shifts) > 0:
            all_shifts.extend(refer_shifts.values)
        if len(urgency_shifts) > 0:
            all_shifts.extend(urgency_shifts.values)

        if len(all_shifts) > 1:
            se = np.std(all_shifts) / np.sqrt(len(all_shifts))
            ci_lower = mean_shift - 1.96 * se
            ci_upper = mean_shift + 1.96 * se
        else:
            ci_lower = ci_upper = mean_shift

        persona_shifts.append({
            "persona": persona,
            "shift": mean_shift,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
        })

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(personas))
    shifts = [p["shift"] for p in persona_shifts]
    ci_lowers = [max(0, p["ci_lower"]) for p in persona_shifts]
    ci_uppers = [p["ci_upper"] for p in persona_shifts]

    colors_pers = [PERSONA_COLORS.get(p, SLATE) for p in personas]

    ax.bar(x, shifts, color=colors_pers, alpha=0.8)
    errors = [np.array(shifts) - np.array(ci_lowers), np.array(ci_uppers) - np.array(shifts)]
    ax.errorbar(x, shifts, yerr=errors, fmt="none", ecolor=TXT, capsize=5, linewidth=2)

    ax.set_xticks(x)
    ax.set_xticklabels([PERSONA_LABELS.get(p, p) for p in personas], fontsize=11)
    ax.set_ylabel("Mean |Decision Shift|", fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    _wm(ax)
    _header(fig, "Persona Susceptibility to Bias", "Mean Shift Rate with 95% CI")

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(output_dir / "fig12_persona_susceptibility.png", dpi=DPI, bbox_inches="tight")
    plt.close()
    print("✓ fig12_persona_susceptibility.png")

def fig_dimension_group_comparison(deltas: pd.DataFrame, output_dir: Path):
    """fig13: Dimension group comparison."""
    _style()

    group_shifts = {}
    for group, dims in DIM_GROUPS.items():
        group_data = deltas[deltas["dimension"].isin(dims)]

        refer_shifts = group_data["refer_match_delta"].dropna()
        urgency_shifts = group_data["urgency_match_delta"].dropna()

        refer_mean = abs(refer_shifts.mean()) if len(refer_shifts) > 0 else 0
        urgency_mean = abs(urgency_shifts.mean()) if len(urgency_shifts) > 0 else 0

        mean_shift = np.mean([refer_mean, urgency_mean])
        group_shifts[group] = mean_shift

    groups = sorted(group_shifts.keys())
    shifts = [group_shifts[g] for g in groups]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors_group = [BLUE, TEAL, AMBER, INDIGO]
    ax.bar(range(len(groups)), shifts, color=colors_group[:len(groups)], alpha=0.8)

    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups, fontsize=11, rotation=15, ha="right")
    ax.set_ylabel("Mean |Decision Shift|", fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    _wm(ax)
    _header(fig, "Dimension Group Comparison", "Bias Impact by Bias Category")

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(output_dir / "fig13_dimension_group_comparison.png", dpi=DPI, bbox_inches="tight")
    plt.close()
    print("✓ fig13_dimension_group_comparison.png")

def fig_disease_category_dimension_interaction(df: pd.DataFrame, deltas: pd.DataFrame, output_dir: Path):
    """fig14: Disease category × dimension interaction heatmap."""
    _style()

    categories = sorted(deltas["gt_category"].dropna().unique())
    dims_list = sorted([d for d in deltas["dimension"].dropna().unique() if d != "baseline"],
                       key=dim_sort_key)

    if len(categories) == 0 or len(dims_list) == 0:
        print("⊘ fig14: Insufficient data for disease × dimension heatmap")
        return

    heatmap_data = []
    for cat in categories:
        row = []
        for dim in dims_list:
            dim_cat_data = deltas[(deltas["dimension"] == dim) & (deltas["gt_category"] == cat)]
            comp_deltas = dim_cat_data["composite_delta"].dropna()
            mean_delta = comp_deltas.mean() if len(comp_deltas) > 0 else 0
            row.append(mean_delta)
        heatmap_data.append(row)

    fig, ax = plt.subplots(figsize=(12, 8))
    heatmap_array = np.array(heatmap_data, dtype=float)

    im = ax.imshow(heatmap_array, cmap="RdBu_r", aspect="auto", vmin=-0.2, vmax=0.2)

    ax.set_xticks(range(len(dims_list)))
    ax.set_yticks(range(len(categories)))
    ax.set_xticklabels([DIM_LABELS.get(d, d) for d in dims_list], fontsize=11, rotation=45, ha="right")
    ax.set_yticklabels([str(c) for c in categories], fontsize=11)

    ax.set_xlabel("Dimension", fontsize=12, fontweight="bold")
    ax.set_ylabel("Disease Category", fontsize=12, fontweight="bold")

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Mean Composite Δ", fontsize=11, fontweight="bold")

    _wm(ax)
    _header(fig, "Disease Category × Dimension Interaction", "Composite Score Changes")

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(output_dir / "fig14_disease_dimension_interaction.png", dpi=DPI, bbox_inches="tight")
    plt.close()
    print("✓ fig14_disease_dimension_interaction.png")

def fig_model_ranking_forest(df: pd.DataFrame, output_dir: Path):
    """fig15: Forest plot — overall shift rates per model with CI."""
    _style()

    baseline_df = df[df["condition"] == "baseline"].copy()

    models = sorted(baseline_df["model"].unique())
    composites = []

    for model in models:
        model_data = baseline_df[baseline_df["model"] == model]
        comp = model_data["composite_score"].mean()
        composites.append(comp)

    # Sort by composite score
    sorted_indices = np.argsort(composites)[::-1]
    models_sorted = [models[i] for i in sorted_indices]
    composites_sorted = [composites[i] for i in sorted_indices]

    fig, ax = plt.subplots(figsize=(11, 8))

    y_pos = np.arange(len(models_sorted))
    providers_sorted = [_get_provider(m) for m in models_sorted]
    colors_forest = [PROVIDER_COLORS.get(p, SLATE) for p in providers_sorted]

    ax.barh(y_pos, composites_sorted, color=colors_forest, alpha=0.8)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(models_sorted, fontsize=10)
    ax.set_xlabel("Composite Score", fontsize=12, fontweight="bold")
    ax.set_xlim([0, 1])
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.grid(axis="x", alpha=0.3)

    # Legend
    legend_patches = [mpatches.Patch(color=PROVIDER_COLORS.get(p, SLATE), label=p)
                     for p in sorted(set(providers_sorted))]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=10)

    _wm(ax)
    _header(fig, "Model Ranking: Baseline Clinical Accuracy", "Composite Score (Descending)")

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(output_dir / "fig15_model_ranking_forest.png", dpi=DPI, bbox_inches="tight")
    plt.close()
    print("✓ fig15_model_ranking_forest.png")

# ============================================================================
# PDF CONSOLIDATION
# ============================================================================

def consolidate_figures_to_pdf(output_dir: Path):
    """Consolidate all PNG figures into a single PDF."""
    try:
        from PIL import Image
        import io

        fig_files = sorted(output_dir.glob("fig*.png"))
        if not fig_files:
            print("⊘ No figures found for PDF consolidation")
            return

        images = [Image.open(f).convert("RGB") for f in fig_files]

        pdf_path = output_dir / "consolidated_figures.pdf"
        images[0].save(pdf_path, save_all=True, append_images=images[1:])
        print(f"✓ consolidated_figures.pdf ({len(images)} pages)")
    except Exception as e:
        print(f"⊘ PDF consolidation failed: {e}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main analysis pipeline."""
    print("\n" + "="*70)
    print("RHEUMATO BIAS PIPELINE: Multi-Model Analysis Suite")
    print("="*70)

    # Input/output paths
    if len(sys.argv) > 1:
        input_path = Path(sys.argv[1])
    else:
        input_path = None

    if len(sys.argv) > 2:
        output_dir = Path(sys.argv[2])
    else:
        output_dir = None

    if not input_path or not input_path.exists():
        input_path, output_dir = prompt_for_paths()
    else:
        if not output_dir:
            output_dir = Path("./output")
        output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading data from: {input_path}")
    df = load_data(input_path)
    df = add_provider_column(df)

    print(f"✓ Loaded {len(df)} records")
    print(f"  Models: {df['model'].nunique()}")
    print(f"  Providers: {df['provider'].nunique()}")
    print(f"  Personas: {df['persona'].nunique()}")
    print(f"  Cases: {df['case_id'].nunique()}")

    # Compute deltas
    print("\nComputing decision deltas...")
    deltas = compute_deltas(df)
    print(f"✓ Computed {len(deltas)} baseline/injection pairs")

    # Generate tables
    print("\nGenerating tables...")
    tables = {}

    tables["T1_Baseline_Pooled"] = table_baseline_accuracy_pooled(df)
    tables["T2_Baseline_by_Model"] = table_baseline_by_model(df)
    tables["T3_Baseline_by_Persona"] = table_baseline_by_persona(df)
    tables["T4_Baseline_by_Provider"] = table_baseline_by_provider(df)
    tables["T5_Decision_Shifts_Pooled"] = table_decision_shifts_pooled(deltas)
    tables["T6_Decision_Shifts_by_Model"] = table_decision_shifts_by_model(deltas)
    tables["T7_Decision_Shifts_by_Persona"] = table_decision_shifts_by_persona(deltas)
    tables["T8_Decision_Shifts_by_Provider"] = table_decision_shifts_by_provider(deltas)
    tables["T9_Psychologization_Rates"] = table_psychologization_rates(df, deltas)
    tables["T10_Urgency_Direction"] = table_urgency_direction(deltas)
    tables["T11_Composite_Deltas"] = table_composite_deltas(deltas)
    tables["T12_Statistical_Tests"] = table_statistical_tests_master(df, deltas)
    tables["T13_Model_Ranking"] = table_model_ranking(df)
    tables["T14_Dimension_Ranking"] = table_dimension_ranking(deltas)
    tables["T15_Pairwise_Comparisons"] = table_pairwise_comparisons(deltas)

    # Save tables to Excel
    excel_path = output_dir / "Analysis_Tables.xlsx"
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        for sheet_name, df_table in tables.items():
            df_table.to_excel(writer, sheet_name=sheet_name, index=False)
    print(f"✓ Analysis_Tables.xlsx ({len(tables)} sheets)")

    # Generate figures
    print("\nGenerating figures...")
    fig_baseline_accuracy_pooled(df, output_dir)
    fig_baseline_by_model(df, output_dir)
    fig_decision_change_heatmap_pooled(deltas, output_dir)
    fig_decision_change_heatmap_by_model(deltas, output_dir)
    fig_referral_urgency_changes(deltas, output_dir)
    fig_psychologization_by_dimension(df, deltas, output_dir)
    fig_urgency_direction_stacked(deltas, output_dir)
    fig_composite_delta_diverging(deltas, output_dir)
    fig_model_susceptibility_scatter(df, deltas, output_dir)
    fig_model_dimension_heatmap(deltas, output_dir)
    fig_provider_comparison(df, deltas, output_dir)
    fig_persona_susceptibility(deltas, output_dir)
    fig_dimension_group_comparison(deltas, output_dir)
    fig_disease_category_dimension_interaction(df, deltas, output_dir)
    fig_model_ranking_forest(df, output_dir)

    # Consolidate to PDF
    print("\nConsolidating figures to PDF...")
    consolidate_figures_to_pdf(output_dir)

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print(f"\nOutputs saved to: {output_dir}")
    print(f"  ✓ Analysis_Tables.xlsx")
    print(f"  ✓ 15 PNG figures (300 DPI)")
    print(f"  ✓ consolidated_figures.pdf")
    print("\n")

if __name__ == "__main__":
    main()
