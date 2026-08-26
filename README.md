# LLM Orthodontic Hallucination Study — Analysis Pipeline

Code accompanying the manuscript:

> **Most Hallucinations Are Not Clinically Dangerous: Severity Stratification of Large Language Model Errors in Orthodontics**
>
> Ferdi Allaf, DDS, Specialist in Orthodontics · Mustafa Özcan, DDS, Specialist in Orthodontics (corresponding)
> Department of Orthodontics, Faculty of Dentistry
> Istanbul Health and Technology University (İSTÜN), Istanbul, Türkiye
> Under review, 2026.

## Overview

Five large language models — GPT-4o, Claude Opus 4.5, Gemini 2.5 Flash, DeepSeek-V3, and Llama 3.3 70B Instruct — were queried with 100 standardised orthodontic items across five domains (clinical protocols, materials and appliances, cephalometric and biomechanical concepts, cases and landmark studies, adversarial items). Forty-five items contained a fabricated element: 20 forming a dedicated adversarial domain and 25 embedded within otherwise realistic clinical questions.

Responses were coded with a seven-category hallucination taxonomy (H1–H7), then stratified into three severity tiers by the action a clinician acting on the content would take.

## Why severity stratification

The prevailing convention in this literature counts every unsupported element equally: a fabricated citation accompanying otherwise sound advice scores the same as a fabricated treatment protocol. Because citation errors dominate by frequency, aggregate prevalence is driven by them, and model rankings derived from it largely rank citation behaviour rather than clinical safety.

| Tier | Categories | Definition |
|---|---|---|
| **Tier 1 — clinical** | H3, H5 | Could alter a treatment decision |
| **Tier 2 — operational** | H2 | Could cause futile procurement or workflow cost |
| **Tier 3 — epistemic** | H1, H4 | Impedes verification without altering the recommendation |
| Unresolved | H6 | Numeric claims without verified support; adjudicated by two orthodontists |
| Modifier | H7 | Absence of epistemic hedging; analysed as a predictor, not a tier |

## Repository contents

| File | Purpose |
|---|---|
| `config.py` | Pipeline configuration (models, prompts, paths) |
| `query_runner.py` | Sends the 100-item benchmark to each of the 5 models |
| `verifier.py` | Citation verification via PubMed E-utilities and CrossRef |
| `coder_v3.py` | Rule-based seven-category taxonomy coding |
| `severity_analysis.py` | Severity tier assignment, inferential statistics, figures |
| `analyzer.py` | Descriptive statistics and original figure generation |
| `main.py` | Pipeline orchestration |
| `remove_truncated.py` | Handling of API-truncated responses |

## Two citation definitions

Automated verification returns three outcomes: **verified**, **fabricated** (no record retrievable), and **indeterminate** (partial match, missing year, ambiguous title). Two definitions are reported:

- **Permissive** — any non-verified citation counts (`total − verified`)
- **Strict** — only confirmed fabrications count (`fabricated > 0`)

The strict definition is primary. The choice matters: permissive coding raised overall prevalence from 74.2% to 89.8%, because indeterminate outcomes reflect the coverage and matching behaviour of bibliographic APIs as much as model behaviour. `severity_analysis.py` computes both.

## Data

Raw responses, coded outputs, severity-tiered data, statistical tables, and figures are deposited on the Open Science Framework:

**OSF Project**: https://osf.io/dk2pm/
**DOI**: [10.17605/OSF.IO/DK2PM](https://doi.org/10.17605/OSF.IO/DK2PM)

## Reproducibility

```bash
git clone https://github.com/drmustafaozcan/llm-ortho-hallucination.git
cd llm-ortho-hallucination

pip3 install -r requirements.txt

# create .env with your own API keys (see below)
touch .env

# download the query benchmark from OSF:
#   LLM_Orthodontics_Hallucination_Study_Protocol.xlsx  ->  data/

python3 main.py                  # query, verify, code
python3 severity_analysis.py     # tiers, statistics, figures
```

## API keys required

Set the following in a `.env` file (never committed):

```
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
GOOGLE_API_KEY=
DEEPSEEK_API_KEY=
TOGETHER_API_KEY=
NCBI_API_KEY=
NCBI_EMAIL=
```

## Snapshot note

Model versions and parameters at data collection:

- GPT-4o (`gpt-4o-2024-11-20`)
- Claude Opus 4.5 (`claude-opus-4-5-20251101`)
- Gemini 2.5 Flash (`gemini-2.5-flash`, `thinking_budget=0`)
- DeepSeek-V3 (`deepseek-chat`)
- Llama 3.3 70B Instruct via Together AI
- Common: `temperature=0`, `max_tokens=4000`, `seed=42`
- Collection window: 26 May – 2 June 2026

Gemini was queried with extended reasoning disabled; its results characterise that configuration only. Findings apply to these snapshot versions and parameters. Model behaviour changes with version.

## Reporting standards

Reporting follows the CHART statement for chatbot health advice studies and the cross-sectional items of STROBE. The protocol, benchmark, and taxonomy were deposited before data collection. The severity stratification is a secondary, exploratory reanalysis of pre-registered data; the tier structure itself was not pre-registered.

## Citation

> Allaf F, Özcan M. Most Hallucinations Are Not Clinically Dangerous: Severity Stratification of Large Language Model Errors in Orthodontics. Under review, 2026. OSF: https://doi.org/10.17605/OSF.IO/DK2PM

## License

Code: **MIT**. Data on OSF: **CC-BY 4.0**.

## Contact

- Mustafa Özcan (corresponding) — mustafa.ozcan@istun.edu.tr — [ORCID 0009-0008-8331-9493](https://orcid.org/0009-0008-8331-9493)
- Ferdi Allaf — ferdi.allaf@istun.edu.tr — [ORCID 0009-0005-3756-7415](https://orcid.org/0009-0005-3756-7415)

Department of Orthodontics, Faculty of Dentistry, Istanbul Health and Technology University (İSTÜN), Sütlüce Campus, Istanbul, Türkiye
