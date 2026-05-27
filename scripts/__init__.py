"""Auto-load MARGO/.env so scripts inherit MARGO_* and *_API_KEY env vars
without needing the caller to ``set -a; source .env`` in their shell."""

from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    pass
else:
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
