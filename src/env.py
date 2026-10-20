"""Load KEY=VALUE lines from .env into os.environ (no override of real env vars)."""
import os


def load_dotenv(path: str = ".env"):
    if not os.path.exists(path):
        return
    with open(path, encoding="utf8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
