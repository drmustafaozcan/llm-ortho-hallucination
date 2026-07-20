"""
query_runner.py
Send each of the 100 queries to each of the 5 LLMs and save raw responses.

Output: results/raw_responses.jsonl
Each line: {query_id, category, model_name, provider, model_id,
           query_text, response_text, timestamp, language, temperature}
"""
import json
import time
import logging
from datetime import datetime
from pathlib import Path
import pandas as pd

import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger(__name__)


# ============================================================
# PROVIDER CALL FUNCTIONS
# Each returns response_text (str) or raises Exception
# ============================================================

def call_openai(model_id: str, prompt: str) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=config.OPENAI_API_KEY)
    resp = client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": prompt}],
        temperature=config.TEMPERATURE,
        max_tokens=config.MAX_TOKENS,
    )
    return resp.choices[0].message.content


def call_anthropic(model_id: str, prompt: str) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    resp = client.messages.create(
        model=model_id,
        max_tokens=config.MAX_TOKENS,
        temperature=config.TEMPERATURE,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(b.text for b in resp.content if hasattr(b, "text"))


def call_google(model_id: str, prompt: str) -> str:
    import google.generativeai as genai
    genai.configure(api_key=config.GOOGLE_API_KEY)
    model = genai.GenerativeModel(model_id)
    resp = model.generate_content(
        prompt,
        generation_config={
            "temperature": config.TEMPERATURE,
            "max_output_tokens": config.MAX_TOKENS,
        },
    )
    return resp.text


def call_deepseek(model_id: str, prompt: str) -> str:
    # DeepSeek uses an OpenAI-compatible API
    from openai import OpenAI
    client = OpenAI(api_key=config.DEEPSEEK_API_KEY,
                    base_url="https://api.deepseek.com/v1")
    resp = client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": prompt}],
        temperature=config.TEMPERATURE,
        max_tokens=config.MAX_TOKENS,
    )
    return resp.choices[0].message.content


def call_together(model_id: str, prompt: str) -> str:
    # Together AI for Llama models (OpenAI-compatible)
    from openai import OpenAI
    client = OpenAI(api_key=config.TOGETHER_API_KEY,
                    base_url="https://api.together.xyz/v1")
    resp = client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": prompt}],
        temperature=config.TEMPERATURE,
        max_tokens=config.MAX_TOKENS,
    )
    return resp.choices[0].message.content


PROVIDER_DISPATCH = {
    "openai":    call_openai,
    "anthropic": call_anthropic,
    "google":    call_google,
    "deepseek":  call_deepseek,
    "together":  call_together,
}


# ============================================================
# UTILITIES
# ============================================================

def load_queries() -> pd.DataFrame:
    """Load query set from xlsx."""
    df = pd.read_excel(config.QUERY_FILE, sheet_name=config.QUERY_SHEET)
    log.info(f"Loaded {len(df)} queries from {config.QUERY_FILE}")
    return df


def get_query_text(row, language: str) -> str:
    """Return query in chosen language."""
    return row["Query_EN"] if language == "EN" else row["Query_TR"]


def build_prompt(query_text: str, language: str) -> str:
    template = (config.PROMPT_TEMPLATE_EN if language == "EN"
                else config.PROMPT_TEMPLATE_TR)
    return template.format(query=query_text)


def already_done(query_id: str, model_name: str, language: str,
                 done_set: set) -> bool:
    return (query_id, model_name, language) in done_set


def load_done_set() -> set:
    """Read existing JSONL to allow resuming."""
    done = set()
    if config.RAW_RESPONSES_FILE.exists():
        with open(config.RAW_RESPONSES_FILE, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done.add((r["query_id"], r["model_name"], r["language"]))
                except (json.JSONDecodeError, KeyError):
                    continue
    log.info(f"Resuming: {len(done)} already-completed query/model/lang triples")
    return done


def call_with_retry(provider: str, model_id: str, prompt: str) -> str:
    """Call provider with retries and backoff."""
    fn = PROVIDER_DISPATCH[provider]
    last_err = None
    for attempt in range(1, config.RETRY_ATTEMPTS + 1):
        try:
            return fn(model_id, prompt)
        except Exception as e:
            last_err = e
            wait = config.RETRY_BACKOFF_SECONDS * attempt
            log.warning(f"Attempt {attempt} failed for {provider}/{model_id}: "
                        f"{type(e).__name__}: {e}. Retrying in {wait}s.")
            time.sleep(wait)
    raise RuntimeError(f"All retries failed for {provider}/{model_id}: {last_err}")


# ============================================================
# MAIN
# ============================================================

def run(language: str = None):
    """Run all queries on all models in chosen language."""
    language = language or config.LANGUAGE
    log.info(f"=== Running pipeline. Language={language} ===")

    queries = load_queries()
    done = load_done_set()

    total = len(queries) * len(config.MODELS)
    completed = 0
    errors = 0

    with open(config.RAW_RESPONSES_FILE, "a", encoding="utf-8") as out_f:
        for _, row in queries.iterrows():
            qid = row["Query_ID"]
            qtext = get_query_text(row, language)
            prompt = build_prompt(qtext, language)

            for display_name, provider, model_id in config.MODELS:
                completed += 1
                if already_done(qid, display_name, language, done):
                    log.info(f"[{completed}/{total}] {qid} | {display_name} "
                             f"| {language} -> already done, skipping")
                    continue

                log.info(f"[{completed}/{total}] {qid} | {display_name} | {language}")
                try:
                    response = call_with_retry(provider, model_id, prompt)
                except Exception as e:
                    log.error(f"FAILED: {qid} / {display_name}: {e}")
                    response = f"__ERROR__: {type(e).__name__}: {e}"
                    errors += 1

                record = {
                    "query_id": qid,
                    "category": row["Category"],
                    "subcategory": row.get("Subcategory", ""),
                    "trap_type": row.get("Trap_Type", ""),
                    "expected_hall_category": row.get("Expected_Hallucination_Category", ""),
                    "model_name": display_name,
                    "provider": provider,
                    "model_id": model_id,
                    "language": language,
                    "temperature": config.TEMPERATURE,
                    "query_text": qtext,
                    "response_text": response,
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "study_name": config.STUDY_NAME,
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                out_f.flush()

                # Polite delay between calls
                time.sleep(0.5)

    log.info(f"=== Done. Completed={completed}, Errors={errors} ===")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--language", choices=["EN", "TR"], default=None)
    args = p.parse_args()
    run(args.language)
