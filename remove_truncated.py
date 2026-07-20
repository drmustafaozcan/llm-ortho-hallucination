"""
remove_truncated.py
Identify and remove truncated responses from raw_responses.jsonl
so they can be re-run with higher max_tokens.

Usage:
  python3 remove_truncated.py --dry-run    # show what would be removed
  python3 remove_truncated.py --execute    # actually remove
"""
import json
import argparse
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config


def is_truncated(text: str) -> bool:
    """Detect truncated responses by checking if the last sentence is complete."""
    text = text.rstrip()
    if not text:
        return False
    last_char = text[-1]
    # A complete sentence ends with punctuation
    if last_char in '.!?:)"\']}':
        return False
    # If ends with digit, letter, or comma -> truncated
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true', help='Show what would be removed')
    parser.add_argument('--execute', action='store_true', help='Actually remove truncated entries')
    args = parser.parse_args()
    
    if not args.dry_run and not args.execute:
        print("Usage: python3 remove_truncated.py --dry-run | --execute")
        sys.exit(1)
    
    raw_file = config.RAW_RESPONSES_FILE
    verified_file = config.VERIFIED_FILE
    
    if not raw_file.exists():
        print(f"ERROR: {raw_file} not found")
        sys.exit(1)
    
    # Read all responses
    all_records = []
    truncated = []
    keep = []
    
    with open(raw_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                r = json.loads(line)
                all_records.append(r)
                # Don't re-run error responses (already retry-failed)
                if r['response_text'].startswith('__ERROR__'):
                    keep.append(line)
                    continue
                if is_truncated(r['response_text']):
                    truncated.append(r)
                else:
                    keep.append(line)
            except json.JSONDecodeError:
                continue
    
    print(f"=== TRUNCATION ANALYSIS ===")
    print(f"Total records: {len(all_records)}")
    print(f"Truncated (will be removed): {len(truncated)}")
    print(f"Keep (already complete): {len(keep)}")
    print()
    
    # Per model breakdown
    by_model = {}
    for r in truncated:
        m = r['model_name']
        by_model[m] = by_model.get(m, 0) + 1
    
    print("=== Per model ===")
    for m, n in sorted(by_model.items()):
        print(f"  {m}: {n} truncated responses to re-run")
    print()
    
    if args.dry_run:
        print("=== DRY RUN — Nothing changed ===")
        print("If this looks correct, run with --execute")
        return
    
    # Backup original
    backup_file = raw_file.with_suffix('.jsonl.backup')
    print(f"Backing up to: {backup_file}")
    with open(raw_file, 'r', encoding='utf-8') as src, open(backup_file, 'w', encoding='utf-8') as dst:
        dst.write(src.read())
    
    # Also backup verified file if exists
    if verified_file.exists():
        v_backup = verified_file.with_suffix('.jsonl.backup')
        print(f"Backing up verified file to: {v_backup}")
        with open(verified_file, 'r', encoding='utf-8') as src, open(v_backup, 'w', encoding='utf-8') as dst:
            dst.write(src.read())
    
    # Write new raw file (keep only non-truncated)
    print(f"Writing new raw file with {len(keep)} kept records...")
    with open(raw_file, 'w', encoding='utf-8') as f:
        for line in keep:
            f.write(line)
    
    # Remove truncated records from verified file (so they will be re-verified)
    if verified_file.exists():
        print("Updating verified file...")
        truncated_keys = set()
        for r in truncated:
            truncated_keys.add((r['query_id'], r['model_name'], r['language']))
        
        v_keep = []
        with open(verified_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    r = json.loads(line)
                    key = (r['query_id'], r['model_name'], r['language'])
                    if key not in truncated_keys:
                        v_keep.append(line)
                except json.JSONDecodeError:
                    continue
        
        with open(verified_file, 'w', encoding='utf-8') as f:
            for line in v_keep:
                f.write(line)
        print(f"Kept {len(v_keep)} verified records, removed {len(truncated_keys)} for re-verification")
    
    print()
    print("=== DONE ===")
    print(f"Removed {len(truncated)} truncated responses.")
    print(f"Now run: source .env && python3 main.py")
    print("Pipeline will re-run only the missing entries with the current MAX_TOKENS setting.")


if __name__ == '__main__':
    main()
