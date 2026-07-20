# LLM Orthodontic Hallucination Study — Pipeline Code

Analysis pipeline accompanying the manuscript:

> **Five Leading Large Language Models Hallucinate Orthodontic Content in 9 of 10 Responses: A Domain-Specific Taxonomy and Multi-Model Audit**
>
> Mustafa Özcan, DDS, Specialist in Orthodontics · Ferdi Allaf, DDS, Specialist in Orthodontics
> Department of Orthodontics, Faculty of Dentistry
> Istanbul Health and Technology University (İSTÜN)
> Submitted to *Journal of Dentistry* (Elsevier), 2026.

## Overview

Five large language models — GPT-4o, Claude Opus 4.5, Gemini 2.5 Flash, DeepSeek-V3, and Llama 3.3 70B Instruct — were queried on 100 standardised orthodontic questions across five categories (clinical protocols, materials & appliances, cephalometric & biomechanical concepts, landmark studies, adversarial trap items). Responses were coded via a 7-category hallucination taxonomy (H1–H7).

## Repository Contents

| File | Purpose |
|---|---|
| `config.py` | Pipeline configuration (models, prompts, paths) |
| `query_runner.py` | Sends 100 queries to each of the 5 LLMs |
| `verifier.py` | PubMed (E-utilities) + CrossRef citation verification |
| `coder_v3.py` | Rule-based 7-category hallucination coding |
| `analyzer.py` | Statistical analysis + figure generation |
| `main.py` | Pipeline orchestration |
| `remove_truncated.py` | Utility for handling API-truncated responses |

## Data

Raw responses, coded outputs, statistical tables, and figures are deposited on the Open Science Framework:

**OSF Project**: https://osf.io/dk2pm/
**DOI**: [10.17605/OSF.IO/DK2PM](https://doi.org/10.17605/OSF.IO/DK2PM)

## Reproducibility

```bash
# 1. Clone this repo
git clone https://github.com/drmustafaozcan/llm-ortho-hallucination.git
cd llm-ortho-hallucination

# 2. Install dependencies (see requirements.txt if included)
pip3 install openai anthropic google-generativeai requests pandas openpyxl tqdm python-dotenv

# 3. Create a .env file with your own API keys
touch .env
# Then edit .env — see 'API Keys Required' below

# 4. Download the query benchmark from OSF:
# LLM_Orthodontics_Hallucination_Study_Protocol.xlsx

# 5. Run pipeline
python3 main.py
```

## API Keys Required

Set the following in a `.env` file (NOT committed to git):

```
OPENAI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here
GOOGLE_API_KEY=your_key_here
DEEPSEEK_API_KEY=your_key_here
TOGETHER_API_KEY=your_key_here
NCBI_API_KEY=your_key_here
NCBI_EMAIL=your@email.com
```

## Statistical Highlights

- **Overall hallucination rate**: 449/500 responses (89.8%; 95% CI 86.8–92.2)
- **Range**: 73% (Claude Opus 4.5) to 100% (Llama 3.3)
- **Content validity (external panel)**: S-CVI/Ave = 0.923 (excellent)
- Full statistical output: Supplementary S4 on OSF

## Snapshot Note

Model versions and API parameters at data collection:
- GPT-4o (`gpt-4o-2024-11-20`)
- Claude Opus 4.5 (`claude-opus-4-5-20251101`)
- Gemini 2.5 Flash (`gemini-2.5-flash`, `thinking_budget=0`)
- DeepSeek-V3 (`deepseek-chat`)
- Llama 3.3 70B Instruct via Together AI
- Common parameters: `temperature=0`, `max_tokens=4000`, `seed=42`
- Data collection window: 26 May – 2 June 2026

Results apply to these specific model versions and API parameters.

## Citation

If you use this code or the associated data:

> Özcan M, Allaf F. Five Leading Large Language Models Hallucinate Orthodontic Content in 9 of 10 Responses: A Domain-Specific Taxonomy and Multi-Model Audit. *Journal of Dentistry* (submitted, 2026). OSF: https://doi.org/10.17605/OSF.IO/DK2PM

## License

Code released under **MIT License**.
Data on OSF released under **CC-BY 4.0**.

## Contact

- Mustafa Özcan (corresponding author): mustafa.ozcan@istun.edu.tr
- Ferdi Allaf: ferdi.allaf@istun.edu.tr
- Department of Orthodontics, Faculty of Dentistry
- Istanbul Health and Technology University (İSTÜN), Sütlüce Campus, İstanbul, Türkiye

ORCID: [0009-0008-8331-9493](https://orcid.org/0009-0008-8331-9493) (MÖ) · [0009-0005-3756-7415](https://orcid.org/0009-0005-3756-7415) (FA)
