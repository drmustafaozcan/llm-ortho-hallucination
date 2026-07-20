"""
setup_keys.py
Interactive setup for API keys. Tests each key after entry and saves to .env file.

Run with:  python3 setup_keys.py
"""
import getpass
import os
import sys
from pathlib import Path

# ============================================================
# CONFIG
# ============================================================
ENV_FILE = Path(__file__).parent / ".env"

KEYS = [
    ("OPENAI_API_KEY",    "OpenAI (ChatGPT)",     "sk-proj-... veya sk-..."),
    ("ANTHROPIC_API_KEY", "Anthropic (Claude)",   "sk-ant-api03-..."),
    ("GOOGLE_API_KEY",    "Google (Gemini)",      "AIzaSy..."),
    ("DEEPSEEK_API_KEY",  "DeepSeek",             "sk-..."),
    ("TOGETHER_API_KEY",  "Together AI (Llama)",  "uzun bir token"),
    ("NCBI_API_KEY",      "NCBI / PubMed",        "uzun hex"),
    ("NCBI_EMAIL",        "NCBI e-postan",        "ornek@gmail.com"),
]


def banner(text):
    print()
    print("=" * 60)
    print(f"  {text}")
    print("=" * 60)


def test_openai(key):
    try:
        from openai import OpenAI
        client = OpenAI(api_key=key)
        r = client.models.list()
        next(iter(r), None)
        return True, "OK"
    except Exception as e:
        return False, str(e)[:200]


def test_anthropic(key):
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=key)
        r = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=5,
            messages=[{"role": "user", "content": "hi"}]
        )
        return True, "OK"
    except Exception as e:
        # Try fallback to older model name
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=key)
            r = client.messages.create(
                model="claude-3-5-haiku-20241022",
                max_tokens=5,
                messages=[{"role": "user", "content": "hi"}]
            )
            return True, "OK (fallback model)"
        except Exception as e2:
            return False, str(e)[:200]


def test_google(key):
    try:
        import google.generativeai as genai
        genai.configure(api_key=key)
        models = list(genai.list_models())
        if not models:
            return False, "No models returned"
        return True, "OK"
    except Exception as e:
        return False, str(e)[:200]


def test_deepseek(key):
    try:
        from openai import OpenAI
        client = OpenAI(api_key=key, base_url="https://api.deepseek.com/v1")
        r = client.chat.completions.create(
            model="deepseek-chat",
            max_tokens=5,
            messages=[{"role": "user", "content": "hi"}]
        )
        return True, "OK"
    except Exception as e:
        return False, str(e)[:200]


def test_together(key):
    try:
        from openai import OpenAI
        client = OpenAI(api_key=key, base_url="https://api.together.xyz/v1")
        r = client.chat.completions.create(
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
            max_tokens=5,
            messages=[{"role": "user", "content": "hi"}]
        )
        return True, "OK"
    except Exception as e:
        return False, str(e)[:200]


def test_ncbi(key):
    import requests
    try:
        r = requests.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/einfo.fcgi",
            params={"db": "pubmed", "api_key": key, "retmode": "json"},
            timeout=10
        )
        if r.status_code == 200:
            return True, "OK"
        return False, f"HTTP {r.status_code}"
    except Exception as e:
        return False, str(e)[:200]


TESTERS = {
    "OPENAI_API_KEY":    test_openai,
    "ANTHROPIC_API_KEY": test_anthropic,
    "GOOGLE_API_KEY":    test_google,
    "DEEPSEEK_API_KEY":  test_deepseek,
    "TOGETHER_API_KEY":  test_together,
    "NCBI_API_KEY":      test_ncbi,
}


def main():
    banner("LLM Ortho Pipeline - API Key Setup")
    print("Her API anahtarini sirasiyla yapistiracaksin.")
    print("Yapistirirken anahtar EKRANDA GORUNMEYECEK (guvenlik icin).")
    print("Anahtari kopyala, Terminal'e gel, sag tikla -> Paste, Enter.")
    print()
    print("Cikis icin: anahtar yerine 'q' yaz + Enter")
    print()

    saved = {}

    for var_name, label, hint in KEYS:
        banner(f"{label}  [{var_name}]")
        print(f"Beklenen format: {hint}")
        print()

        while True:
            try:
                value = getpass.getpass(f"Yapistir ve Enter (gizli): ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nIptal edildi.")
                sys.exit(1)

            if value.lower() == 'q':
                print("Atlanidi.")
                break

            if not value:
                print("Bos giris. Tekrar dene.")
                continue

            # Sanity check for shell artifacts
            if value.startswith('export ') or '=' in value[:30]:
                print("UYARI: 'export ...' yapistirdin gibi gorunuyor.")
                print("       Sadece anahtarin kendisini yapistir, tirnak yok, 'export' yok.")
                continue

            # Email validation
            if var_name == "NCBI_EMAIL":
                if "@" not in value or "." not in value:
                    print("Gecerli bir e-posta gibi gorunmuyor. Tekrar dene.")
                    continue
                saved[var_name] = value
                print(f"  Kaydedildi: {value}")
                break

            # Test the key
            print(f"  Test ediliyor... ", end="", flush=True)
            tester = TESTERS.get(var_name)
            if tester:
                ok, msg = tester(value)
                if ok:
                    print(f"BASARILI ({msg})")
                    saved[var_name] = value
                    break
                else:
                    print(f"BASARISIZ")
                    print(f"  Hata: {msg}")
                    print(f"  Tekrar dene veya 'q' ile atla.")
                    continue
            else:
                saved[var_name] = value
                print("  Kaydedildi (test edilmedi)")
                break

    # Write .env file
    banner("Kaydediliyor")
    if not saved:
        print("Hicbir anahtar kaydedilmedi. Cikis.")
        return

    with open(ENV_FILE, "w") as f:
        for k, v in saved.items():
            f.write(f'{k}="{v}"\n')

    os.chmod(ENV_FILE, 0o600)  # readable only by user

    print(f"Kaydedildi: {ENV_FILE}")
    print(f"Toplam: {len(saved)} anahtar")
    print()
    print("SIRADAKI ADIM:")
    print("  source .env && python3 main.py")
    print()


if __name__ == "__main__":
    main()
