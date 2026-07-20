"""
verifier.py
Parse citations from LLM responses, verify them against PubMed and CrossRef.

Output: results/verified_citations.jsonl
Each line: full record from raw_responses.jsonl plus 'citations' field with
verification results.

Citation parsing strategy:
- Regex for APA-style references with author-year-journal patterns
- Regex for DOIs (most reliable signal)
- Regex for PMIDs
- All extracted citations are then verified against PubMed/CrossRef.
"""
import re
import json
import time
import logging
import urllib.parse
from typing import List, Dict, Optional
import requests

import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger(__name__)


# ============================================================
# CITATION EXTRACTION
# ============================================================

# DOI regex (handles most real-world variants)
DOI_RE = re.compile(r'\b(10\.\d{4,9}/[-._;()/:A-Z0-9]+)\b', re.IGNORECASE)

# PMID regex
PMID_RE = re.compile(r'PMID[:\s]+(\d{6,9})', re.IGNORECASE)

# APA-ish: Author(s) (Year). Title. Journal, vol(issue), pages.
# This is intentionally permissive; many LLM outputs deviate from strict APA.
APA_RE = re.compile(
    r'([A-Z][a-zA-Z\-\']+(?:,\s*[A-Z]\.(?:\s*[A-Z]\.)*)?'      # First author
    r'(?:,?\s*(?:&|and|et\s+al\.?)\s*(?:[A-Z][a-zA-Z\-\']+(?:,\s*[A-Z]\.(?:\s*[A-Z]\.)*)?,?\s*)*)?)'  # Other authors
    r'\s*\((\d{4})\)\.?\s*'                                     # (Year)
    r'([^.]{10,200}?)\.\s*'                                     # Title
    r'([A-Z][^,.]{3,100})',                                     # Journal
    re.MULTILINE
)


def extract_citations(text: str) -> List[Dict]:
    """Extract citations from a response. Returns list of citation dicts."""
    citations = []

    # 1. Find DOIs first (most reliable)
    for m in DOI_RE.finditer(text):
        doi = m.group(1).rstrip('.,;)')
        # Surrounding context: 200 chars
        start = max(0, m.start() - 200)
        end = min(len(text), m.end() + 100)
        context = text[start:end].replace("\n", " ").strip()
        citations.append({
            "type": "doi",
            "value": doi,
            "context": context,
        })

    # 2. Find PMIDs
    for m in PMID_RE.finditer(text):
        pmid = m.group(1)
        start = max(0, m.start() - 200)
        end = min(len(text), m.end() + 100)
        context = text[start:end].replace("\n", " ").strip()
        citations.append({
            "type": "pmid",
            "value": pmid,
            "context": context,
        })

    # 3. APA-style references (fallback when no DOI)
    for m in APA_RE.finditer(text):
        authors, year, title, journal = m.groups()
        citations.append({
            "type": "apa",
            "value": {
                "authors": authors.strip(),
                "year": year.strip(),
                "title": title.strip(),
                "journal": journal.strip(),
            },
            "context": m.group(0)[:400],
        })

    # Deduplicate
    seen = set()
    unique = []
    for c in citations:
        key = (c["type"], str(c["value"]))
        if key not in seen:
            seen.add(key)
            unique.append(c)
    return unique


# ============================================================
# VERIFICATION VIA APIs
# ============================================================

def verify_doi_crossref(doi: str) -> Dict:
    """Check if DOI resolves on CrossRef."""
    url = f"https://api.crossref.org/works/{urllib.parse.quote(doi)}"
    try:
        r = requests.get(url, timeout=15,
                         headers={"User-Agent": f"{config.STUDY_NAME} (mailto:{config.NCBI_EMAIL})"})
        if r.status_code == 200:
            data = r.json().get("message", {})
            return {
                "verified": True,
                "title": (data.get("title") or [""])[0],
                "authors": [f"{a.get('family','')}, {a.get('given','')}"
                            for a in data.get("author", [])],
                "year": str((data.get("issued", {}).get("date-parts") or [[""]])[0][0]),
                "journal": (data.get("container-title") or [""])[0],
                "doi": data.get("DOI", doi),
            }
        return {"verified": False, "error": f"HTTP {r.status_code}"}
    except Exception as e:
        return {"verified": False, "error": f"{type(e).__name__}: {e}"}


def verify_pmid(pmid: str) -> Dict:
    """Check if PMID exists via NCBI ESummary."""
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
    params = {
        "db": "pubmed",
        "id": pmid,
        "retmode": "json",
        "email": config.NCBI_EMAIL,
    }
    if config.NCBI_API_KEY:
        params["api_key"] = config.NCBI_API_KEY
    try:
        r = requests.get(url, params=params, timeout=15)
        if r.status_code == 200:
            data = r.json().get("result", {})
            if pmid in data and "error" not in data[pmid]:
                rec = data[pmid]
                return {
                    "verified": True,
                    "title": rec.get("title", ""),
                    "authors": [a.get("name", "") for a in rec.get("authors", [])],
                    "year": rec.get("pubdate", "")[:4],
                    "journal": rec.get("source", ""),
                    "pmid": pmid,
                }
            return {"verified": False, "error": "PMID not found"}
        return {"verified": False, "error": f"HTTP {r.status_code}"}
    except Exception as e:
        return {"verified": False, "error": f"{type(e).__name__}: {e}"}


def search_pubmed_apa(c: Dict) -> Dict:
    """Search PubMed for an APA-style reference. Returns verification dict."""
    v = c["value"]
    # Build search query: first author surname + year + title fragment
    first_author = v["authors"].split(",")[0].strip().split()[-1] if v["authors"] else ""
    title_frag = " ".join(v["title"].split()[:6])
    term = f'({first_author}[Author]) AND ({v["year"]}[Year])'
    if title_frag:
        term += f' AND ("{title_frag}"[Title])'

    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {
        "db": "pubmed",
        "term": term,
        "retmode": "json",
        "retmax": "5",
        "email": config.NCBI_EMAIL,
    }
    if config.NCBI_API_KEY:
        params["api_key"] = config.NCBI_API_KEY
    try:
        r = requests.get(url, params=params, timeout=15)
        if r.status_code == 200:
            ids = r.json().get("esearchresult", {}).get("idlist", [])
            if ids:
                # Found at least one matching PMID - verify the first
                time.sleep(config.PUBMED_DELAY)
                return verify_pmid(ids[0])
            # No hits - try a looser search (title only)
            params["term"] = f'"{title_frag}"[Title]'
            time.sleep(config.PUBMED_DELAY)
            r2 = requests.get(url, params=params, timeout=15)
            if r2.status_code == 200:
                ids2 = r2.json().get("esearchresult", {}).get("idlist", [])
                if ids2:
                    time.sleep(config.PUBMED_DELAY)
                    res = verify_pmid(ids2[0])
                    res["match_quality"] = "title_only"
                    return res
            return {"verified": False, "error": "No PubMed hits"}
        return {"verified": False, "error": f"HTTP {r.status_code}"}
    except Exception as e:
        return {"verified": False, "error": f"{type(e).__name__}: {e}"}


def verify_citation(c: Dict) -> Dict:
    """Dispatch verification based on citation type."""
    if c["type"] == "doi":
        time.sleep(config.PUBMED_DELAY)
        return verify_doi_crossref(c["value"])
    if c["type"] == "pmid":
        time.sleep(config.PUBMED_DELAY)
        return verify_pmid(c["value"])
    if c["type"] == "apa":
        time.sleep(config.PUBMED_DELAY)
        return search_pubmed_apa(c)
    return {"verified": False, "error": "Unknown citation type"}


# ============================================================
# MAIN
# ============================================================

def run():
    """Read raw responses, extract & verify citations, write verified file."""
    if not config.RAW_RESPONSES_FILE.exists():
        log.error(f"Raw responses file not found: {config.RAW_RESPONSES_FILE}")
        log.error("Run query_runner.py first.")
        return

    # Resume support
    done_set = set()
    if config.VERIFIED_FILE.exists():
        with open(config.VERIFIED_FILE, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done_set.add((r["query_id"], r["model_name"], r["language"]))
                except (json.JSONDecodeError, KeyError):
                    continue
        log.info(f"Resuming: {len(done_set)} already-verified records")

    with open(config.RAW_RESPONSES_FILE, "r", encoding="utf-8") as in_f, \
         open(config.VERIFIED_FILE, "a", encoding="utf-8") as out_f:

        for line in in_f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            key = (rec["query_id"], rec["model_name"], rec["language"])
            if key in done_set:
                continue

            if rec["response_text"].startswith("__ERROR__"):
                rec["citations"] = []
                rec["citation_stats"] = {"total": 0, "verified": 0,
                                          "fabricated": 0, "errored": 0}
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out_f.flush()
                continue

            log.info(f"Verifying {rec['query_id']} / {rec['model_name']} / {rec['language']}")
            citations = extract_citations(rec["response_text"])
            log.info(f"  Found {len(citations)} citation candidates")

            verified_list = []
            for c in citations:
                v = verify_citation(c)
                c["verification"] = v
                verified_list.append(c)

            verified_count = sum(1 for c in verified_list
                                 if c["verification"].get("verified"))
            fabricated_count = sum(1 for c in verified_list
                                   if not c["verification"].get("verified")
                                   and "error" in c["verification"]
                                   and "HTTP" not in c["verification"].get("error", ""))
            errored_count = sum(1 for c in verified_list
                                if "HTTP" in c["verification"].get("error", ""))

            rec["citations"] = verified_list
            rec["citation_stats"] = {
                "total": len(verified_list),
                "verified": verified_count,
                "fabricated": fabricated_count,
                "errored": errored_count,
            }
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out_f.flush()

    log.info("=== Verification complete ===")


if __name__ == "__main__":
    run()
