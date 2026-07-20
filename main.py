"""
main.py
Orchestrate the full pipeline:
  1. Run all queries on all models  (query_runner.py)
  2. Extract & verify citations      (verifier.py)
  3. Apply taxonomy coding           (coder.py)
  4. Generate stats & figures        (analyzer.py)

Usage:
  python main.py                 # full pipeline, English
  python main.py --language TR   # Turkish run (for language-bias analysis)
  python main.py --stage 3       # only run from stage 3 onward
  python main.py --stage 4 --only  # run only stage 4

Stages:
  1 = query_runner   (LLM calls)
  2 = verifier       (PubMed/CrossRef checks)
  3 = coder          (taxonomy)
  4 = analyzer       (stats + figures)
"""
import argparse
import logging
import sys

import config
import query_runner
import verifier
import coder
import analyzer

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger(__name__)


STAGES = {
    1: ("Query LLMs", lambda lang: query_runner.run(language=lang)),
    2: ("Verify citations", lambda lang: verifier.run()),
    3: ("Code hallucinations", lambda lang: coder.run()),
    4: ("Analyze & plot", lambda lang: analyzer.run()),
}


def check_api_keys():
    missing = []
    if not config.OPENAI_API_KEY:    missing.append("OPENAI_API_KEY")
    if not config.ANTHROPIC_API_KEY: missing.append("ANTHROPIC_API_KEY")
    if not config.GOOGLE_API_KEY:    missing.append("GOOGLE_API_KEY")
    if not config.DEEPSEEK_API_KEY:  missing.append("DEEPSEEK_API_KEY")
    if not config.TOGETHER_API_KEY:  missing.append("TOGETHER_API_KEY")
    if not config.NCBI_EMAIL or "example.com" in config.NCBI_EMAIL:
        missing.append("NCBI_EMAIL")
    if missing:
        log.warning(f"Missing or default API keys: {', '.join(missing)}")
        log.warning("Set them via environment variables before running stage 1.")
    return missing


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--language", choices=["EN", "TR"], default=config.LANGUAGE)
    p.add_argument("--stage", type=int, default=1,
                   help="Start from this stage (1-4)")
    p.add_argument("--only", action="store_true",
                   help="Run only the specified stage")
    args = p.parse_args()

    if args.stage == 1:
        missing = check_api_keys()
        if missing and not config.RAW_RESPONSES_FILE.exists():
            log.error("Set API keys before running stage 1.")
            sys.exit(1)

    stages_to_run = ([args.stage] if args.only
                     else range(args.stage, 5))

    for s in stages_to_run:
        name, fn = STAGES[s]
        log.info(f"\n{'='*60}\nSTAGE {s}: {name}\n{'='*60}")
        fn(args.language)

    log.info("\n=== Pipeline complete ===")


if __name__ == "__main__":
    main()
