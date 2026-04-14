from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv


def patch_sqlite() -> None:
    try:
        __import__("pysqlite3")
        sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
    except Exception:
        pass


patch_sqlite()

APP_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = APP_ROOT.parent

if str(APP_ROOT) in sys.path:
    sys.path.remove(str(APP_ROOT))
sys.path.insert(0, str(APP_ROOT))

if str(PROJECT_ROOT) in sys.path:
    sys.path.remove(str(PROJECT_ROOT))
sys.path.insert(1, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")
