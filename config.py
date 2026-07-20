"""
Configuration for LLM Orthodontics Hallucination Study Pipeline.

Set your API keys via environment variables (recommended) or directly here.
DO NOT commit this file with real keys to a public repo.
"""
import os
from pathlib import Path

# ============================================================
# PATHS
# ============================================================
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
RAW_DIR = RESULTS_DIR / "raw_outputs"
FIGURES_DIR = RESULTS_DIR / "figures"

for d in (DATA_DIR, RESULTS_DIR, RAW_DIR, FIGURES_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Input query file (place the xlsx here)
QUERY_FILE = DATA_DIR / "LLM_Orthodontics_Hallucination_Study_Protocol.xlsx"
QUERY_SHEET = "Query Set"

# Output files
RAW_RESPONSES_FILE = RESULTS_DIR / "raw_responses.jsonl"
VERIFIED_FILE = RESULTS_DIR / "verified_citations.jsonl"
CODED_FILE = RESULTS_DIR / "coded_hallucinations.csv"
SUMMARY_FILE = RESULTS_DIR / "summary_statistics.csv"

# ============================================================
# API KEYS (use environment variables)
# ============================================================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
# For Llama: use a provider like Together, Groq, or Fireworks
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY", "")

# PubMed - register at https://www.ncbi.nlm.nih.gov/account/
# Higher rate limits with API key (10 req/sec vs 3 req/sec)
NCBI_API_KEY = os.getenv("NCBI_API_KEY", "")
NCBI_EMAIL = os.getenv("NCBI_EMAIL", "your.email@example.com")  # required by NCBI

# ============================================================
# MODELS TO TEST
# ============================================================
# Each entry: (display_name, provider, model_id)
# Adjust model IDs to whatever is current when you run this.
MODELS = [
    ("GPT-4o",        "openai",    "gpt-4o-2024-11-20"),
    ("Claude Opus",   "anthropic", "claude-opus-4-5"),
    ("Gemini Pro",    "google",    "gemini-2.5-flash"),
    ("DeepSeek-V3",   "deepseek",  "deepseek-chat"),
    ("Llama 3.3",     "together",  "meta-llama/Llama-3.3-70B-Instruct-Turbo"),
]

# ============================================================
# QUERY PARAMETERS
# ============================================================
TEMPERATURE = 0.0            # Lowest available; some models clamp to 0.01
MAX_TOKENS = 4000
LANGUAGE = "EN"              # "EN" or "TR" - change for the parallel Turkish run

# Standardized prompt template (English)
PROMPT_TEMPLATE_EN = """You are a clinical orthodontist responding to a colleague's question. \
Provide a detailed, evidence-based answer with peer-reviewed references in APA format. \
Always include at least 3 specific citations with authors, year, journal/source, and DOI when possible.

Question: {query}"""

# Standardized prompt template (Turkish)
PROMPT_TEMPLATE_TR = """Bir meslektaşınızın sorusuna yanıt veren klinik bir ortodontistsiniz. \
Hakemli referanslarla APA formatında detaylı, kanıta dayalı bir yanıt verin. \
Yazarlar, yıl, dergi/kaynak ve mümkünse DOI ile birlikte en az 3 spesifik atıf ekleyin.

Soru: {query}"""

# ============================================================
# RATE LIMITING / RETRIES
# ============================================================
MAX_CONCURRENT_REQUESTS = 3   # Be polite to APIs
RETRY_ATTEMPTS = 3
RETRY_BACKOFF_SECONDS = 5

# PubMed rate limits: 3/sec without key, 10/sec with key
PUBMED_DELAY = 0.35 if NCBI_API_KEY else 0.5

# ============================================================
# STUDY METADATA (for reproducibility)
# ============================================================
STUDY_NAME = "LLM-OrthoHallucination-2026"
DATE_WINDOW_START = "2026-05-26"  # Update to actual data collection start
DATE_WINDOW_END = "2026-06-02"    # Update to actual data collection end
