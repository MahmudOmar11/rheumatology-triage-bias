# Sociodemographic Bias in LLM-Based Rheumatology Triage

Code and data for measuring sociodemographic bias in large language model (LLM) clinical triage decisions for rheumatology referrals.

## Overview

This pipeline evaluates whether LLMs change their clinical triage recommendations when identical rheumatology vignettes include sociodemographic pressure-point injections (e.g., race, socioeconomic status, pain expression tone). It compares structured model outputs against expert-adjudicated ground truth across 13 LLMs from three providers.

### Study Design

**30 clinical cases × 3 rephrasings = 90 vignettes**, each evaluated at baseline (no injection) and with **57 pressure-point injections** across **9 bias dimensions**:

| Dimension | Levels | Rationale |
|-----------|--------|-----------|
| Race / Ethnicity | Black, Hispanic, White | Documented disparities in rheumatology referral |
| Pain Expression (Tone) | Angry, Anxious, Emotional, Stoic | Tone may trigger differential workup |
| Socioeconomic Status | High, Low, Unhoused | SES cues may bias urgency assessment |
| Clinician Anchoring | Dismissive, MSK, Psychiatric | Prior clinician framing may anchor decisions |
| Health Literacy | High, Low | Literacy level may affect perceived severity |
| Language Barrier | Present | Non-English speakers may receive less workup |
| Psychiatric History | Present | Psych history may trigger psychologization |
| Weight / BMI | Obese | Obesity may lead to mechanical attribution |
| Substance Use | Active | Substance use may reduce referral priority |

Each call uses **4 system-prompt personas** (physician, helpful AI, conservative PCP, no persona) and **5 repeats** per condition.

### Models Tested

| Provider | Models |
|----------|--------|
| OpenAI | GPT-4.1-mini, GPT-4.1, GPT-5-mini, GPT-5.2, o4-mini |
| Anthropic | Claude Haiku 4.5, Claude Sonnet 4.5, Claude Sonnet 4.6, Claude Opus 4.6 |
| Google | Gemini 2.5 Flash-Lite, Gemini 2.5 Flash, Gemini 3 Flash Preview |

### Structured Output (16 Fields)

Each model call returns a structured JSON assessment:

| Field | Type | Description |
|-------|------|-------------|
| `refer_rheumatology` | Yes / No | Referral decision |
| `referral_urgency` | Emergent / Urgent / Routine / None | Urgency level |
| `order_labs` | Yes / No | Lab workup ordered |
| `labs_ordered` | Free text | Specific tests |
| `order_imaging` | Yes / No | Imaging ordered |
| `imaging_ordered` | Free text | Specific imaging |
| `joint_aspiration` | Yes / No | Aspiration indicated |
| `attributes_to_psychological` | Yes / No | Psychological attribution |
| `recommend_reassurance_only` | Yes / No | Reassurance without workup |
| `acuity_assessment` | Emergent / Urgent / Semi-urgent / Routine | Clinical acuity |
| `red_flags_identified` | Free text | Red flags found |
| `immediate_action` | Free text | Urgent actions needed |
| `primary_diagnosis` | Free text | Top diagnostic suspicion |
| `differential_2` / `differential_3` | Free text | 2nd and 3rd differentials |
| `rationale` | Free text | Clinical reasoning |

### Grading Metrics (19 Concordance Measures)

Each response is graded against expert ground truth:

- **Binary concordance**: referral, urgency, labs, imaging, aspiration, acuity, diagnosis (primary + top-3)
- **Directional errors**: urgency downgrade/upgrade, acuity downgrade/upgrade, under-referral, over-referral
- **Psychologization error**: model attributes symptoms to psychological cause when ground truth says No
- **Reassurance error**: model recommends reassurance-only when workup is indicated
- **Composite score**: mean of 5 binary concordance checks (referral, urgency, labs, imaging, aspiration)

## Repository Structure

```
├── README.md
├── LICENSE
├── requirements.txt
├── data/
│   ├── rheum_bias_cases.xlsx      # 90 clinical vignettes (30 cases × 3 rephrasings)
│   └── rheum_bias_iterations.xlsx # 57 pressure-point injections (9 dimensions)
├── src/
│   ├── pipeline.py                # Main experiment pipeline (async, multi-provider)
│   └── analysis.py                # Publication figures (11) and tables (10)
└── output/                        # Generated results (gitignored)
```

## Quickstart

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set API Keys

```bash
# Set one or more, depending on which provider you want to run:
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GOOGLE_API_KEY="AI..."
```

### 3. Run the Pipeline

```bash
python src/pipeline.py
```

The interactive CLI will prompt you to select:
- **Provider**: openai, anthropic, or google
- **Model**: from the preset list for each provider
- **Personas**: all 4 or a subset
- **Number of cases**: 1–90 (use a small number for trial runs)
- **Repeats**: default 5
- **Temperature**: default 0.3
- **Concurrency**: default 30 (15 for Google)

A test API call runs before the main experiment to catch configuration errors.

#### Trial Run (Recommended First)

When prompted "How many cases to run?", enter `3` to do a quick trial (~180 API calls instead of ~36,000 per model).

### 4. Run the Analysis

```bash
python src/analysis.py output/results_YYYYMMDD_HHMMSS.xlsx output/figures
```

This generates 11 publication-quality PNG figures (300 DPI), 10 Excel tables, and a consolidated PDF.

## Pipeline Details

### Experiment Flow

For each model (one per run):

```
For each persona (4):
  For each case vignette (90):
    For each repeat (5):
      1. BASELINE: run vignette without injection → grade vs ground truth
      2. For each matching injection (19 per rephrase):
         ITERATION: run vignette with injection → grade vs ground truth
```

### Checkpoint and Resume

The pipeline writes each API response to a JSONL checkpoint file as it completes. If a run is interrupted, re-running with the same output directory will skip already-completed calls and resume from where it stopped.

### Structured Output Enforcement

- **OpenAI**: Responses API with `text.format` JSON schema (strict mode)
- **Anthropic**: `tool_use` with `tool_choice` forcing the assessment tool
- **Google**: `response_mime_type="application/json"` with `response_schema`

### Reasoning Model Handling

Reasoning models (o-series, GPT-5.x) receive special handling:
- Temperature parameter is omitted (not supported)
- `max_output_tokens` is set to 16,384 to accommodate internal chain-of-thought
- `reasoning.effort` is set to `"low"` to keep outputs focused

## Analysis Outputs

### Figures

| # | Figure | Description |
|---|--------|-------------|
| 1 | Baseline Accuracy | Bar chart with Wilson confidence intervals |
| 2 | Decision Change Heatmap | Metric × dimension change rates |
| 3 | Referral & Urgency Changes | Paired bars by dimension |
| 4 | Psychologization Dual Panel | Error rates + diverging deltas |
| 5 | Urgency Direction | Stacked bars (downgraded / correct / upgraded) |
| 6 | Composite Delta | Diverging bars by dimension × level |
| 7 | Composite by Group | Box plots with strip overlay |
| 8 | Disease × Dimension | Category interaction heatmap |
| 9 | Persona Susceptibility | Shift rates by system prompt |
| 10 | Model × Persona | Interaction heatmap |
| 11 | Model Shift | Paired dot plot across models |

### Tables

| # | Table | Description |
|---|-------|-------------|
| 1 | Baseline Accuracy | Concordance rates per metric with Wilson CI |
| 2 | Model × Persona | Baseline accuracy by model and persona |
| 3 | Decision Shifts | Change rates by dimension × level |
| 4 | GT Concordance | Per-group concordance + delta from baseline |
| 5 | Psychologization | Rates and errors by dimension |
| 6 | Urgency Shifts | Downgrade / correct / upgrade proportions |
| 7 | Statistical Tests | Binomial tests, FDR-corrected p-values, Cohen's h |
| 8 | Composite Scores | Composite concordance by group with CI |
| 9 | Shift by Persona | Decision shift susceptibility per persona |
| 10 | Model × Dimension | Interaction effect sizes |

### Statistical Methods

- **Wilson confidence intervals** for proportions
- **Binomial tests** comparing injection rates to baseline
- **Benjamini–Hochberg FDR** correction for multiple comparisons
- **Cohen's h** effect sizes for proportion differences

## Data Dictionary

### `rheum_bias_cases.xlsx` — Cases Sheet

| Column | Description |
|--------|-------------|
| `case_id` | Integer 1–30 |
| `rephrase_id` | Rephrase variant (1, 2, or 3) |
| `Age` | Patient age |
| `Sex` | Patient sex |
| `Clinical_Vignette` | Full clinical vignette text |
| `GT_Dx` | Ground-truth primary diagnosis |
| `GT_Category` | Disease category (CTD, Crystal, Infectious, Inflammatory, Mechanical, Mimic, Non-inflammatory, Systemic) |
| `GT_Acuity` | Ground-truth acuity (Emergent, Urgent, Routine) |
| `GT_Refer_Rheum` | Should refer to rheumatology (Yes / No) |
| `GT_Refer_Urgency` | Referral urgency (Emergent / Urgent / Routine / None) |
| `GT_Labs` | Labs indicated (Yes / No) |
| `GT_Imaging` | Imaging indicated (Yes / No) |
| `GT_Aspiration` | Aspiration indicated (Yes / No) |
| `GT_Psychologization_Appropriate` | Is psychological attribution appropriate (Yes / No) |
| `GT_Immediate_Action` | Immediate action needed (free text or None) |

### `rheum_bias_iterations.xlsx` — Iterations Sheet

| Column | Description |
|--------|-------------|
| `iteration_id` | Unique injection identifier |
| `dimension` | Bias dimension (race, tone, ses, etc.) |
| `level` | Specific level within dimension |
| `rephrase_id` | Matching rephrase variant (1, 2, or 3) |
| `injection_text` | Text injected into the vignette after the first sentence |

## Reproducing Results

To reproduce the full experiment across all 13 models:

```bash
# Run each provider/model combination
for provider in openai anthropic google; do
    python src/pipeline.py
    # Select provider, model, 'all' personas, 'all' cases, 5 repeats
done
```

Each model run produces an Excel file in `output/`. To merge and analyze all runs:

```bash
python src/analysis.py output/results_*.xlsx output/figures
```

## Citation

If you use this code or data, please cite:

```
@article{omar2026rheum_bias,
  title={Sociodemographic Bias in Large Language Model Clinical Triage for Rheumatology},
  author={Omar, Mahmud and others},
  year={2026}
}
```

## License

MIT License. See [LICENSE](LICENSE).
