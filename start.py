#!/usr/bin/env python3
"""
One-command launcher for macOS and Windows.

    macOS   :  python3 start.py      (or double-click start_mac.command)
    Windows :  py start.py           (or double-click start_windows.bat)

What it does:
  1. checks the Python version (3.10 - 3.13 supported)
  2. creates a private virtual environment in .venv (first run only)
  3. installs requirements.txt into it (first run, or when requirements change)
  4. starts the Streamlit app and opens it in your browser

Options:
  --port 8501      choose the port (default: first free port from 8501)
  --reinstall      force re-installing the dependencies
  --test           run the automated tests instead of the app
  --eval           run the retrieval evaluation instead of the app
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import socket
import subprocess
import sys
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"
IS_WIN = os.name == "nt"
VENV_PY = VENV / ("Scripts/python.exe" if IS_WIN else "bin/python")
REQ = ROOT / "requirements.txt"
STAMP = VENV / ".requirements.sha256"


def say(msg: str) -> None:
    print(f"[start] {msg}", flush=True)


def check_python() -> None:
    v = sys.version_info
    if v < (3, 10):
        say(f"Python {v.major}.{v.minor} is too old. Please install Python 3.11 or 3.12 from https://www.python.org/downloads/")
        sys.exit(1)
    if v >= (3, 14):
        say(f"Warning: Python {v.major}.{v.minor} is newer than the tested versions (3.10-3.13). "
            "If installation fails, install Python 3.12 and run this script with it.")


def ensure_venv() -> None:
    if VENV_PY.exists():
        return
    say("Creating virtual environment in .venv (first run only)...")
    try:
        venv.EnvBuilder(with_pip=True, clear=False, upgrade_deps=False).create(VENV)
    except Exception as e:  # noqa: BLE001
        say(f"Could not create the virtual environment: {e}")
        if not IS_WIN:
            say("On some systems you need the venv module: e.g. `sudo apt install python3-venv` (Linux).")
        sys.exit(1)


def requirements_hash() -> str:
    return hashlib.sha256(REQ.read_bytes()).hexdigest()


def ensure_requirements(force: bool = False) -> None:
    want = requirements_hash()
    if not force and STAMP.exists() and STAMP.read_text().strip() == want:
        return
    say("Installing dependencies (first run takes 2-5 minutes)...")
    run([str(VENV_PY), "-m", "pip", "install", "--upgrade", "pip"], check=False)
    rc = run([str(VENV_PY), "-m", "pip", "install", "-r", str(REQ)], check=False)
    if rc != 0:
        say("Dependency installation failed. Check your internet connection and try again with: "
            f"{'py' if IS_WIN else 'python3'} start.py --reinstall")
        sys.exit(rc)
    STAMP.write_text(want)
    say("Dependencies installed.")


def ensure_env_file() -> None:
    env, example = ROOT / ".env", ROOT / ".env.example"
    if not env.exists() and example.exists():
        shutil.copy(example, env)
        say("Created .env from .env.example (add your Gemini API key in the app's Settings panel).")


def free_port(start: int) -> int:
    for port in range(start, start + 20):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    return start


def run(cmd, check: bool = True) -> int:
    proc = subprocess.run(cmd, cwd=str(ROOT))
    if check and proc.returncode != 0:
        sys.exit(proc.returncode)
    return proc.returncode


EXCLUDE_DIRS = {".venv", "venv", "data", "dist", "__pycache__", ".pytest_cache", ".git", "results", ".idea", ".vscode"}
EXCLUDE_FILES = {".env", ".DS_Store", "source_members.csv", "source_members_full.csv"}


def package() -> None:
    """Create dist/capstone.zip containing only the source, sample data and docs (no secrets)."""
    import zipfile

    out = ROOT / "dist" / "capstone.zip"
    out.parent.mkdir(exist_ok=True)
    count = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(ROOT.rglob("*")):
            rel = p.relative_to(ROOT)
            if any(part in EXCLUDE_DIRS for part in rel.parts) or p.name in EXCLUDE_FILES or p.is_dir():
                continue
            if p.suffix in (".pyc", ".log"):
                continue
            info = zipfile.ZipInfo.from_file(p, arcname=str(Path("capstone-rag-agent") / rel).replace(os.sep, "/"))
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, p.read_bytes())
            count += 1
    say(f"Created {out.relative_to(ROOT)} with {count} files (no .venv, data, .env or API keys).")


def main() -> None:
    ap = argparse.ArgumentParser(description="Start the DocuMind app")
    ap.add_argument("--port", type=int, default=8501)
    ap.add_argument("--reinstall", action="store_true")
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--package", action="store_true")
    args = ap.parse_args()

    os.chdir(ROOT)
    if args.package:
        package()
        return
    check_python()
    ensure_venv()
    ensure_requirements(force=args.reinstall)
    ensure_env_file()

    if args.test:
        sys.exit(run([str(VENV_PY), "-m", "pytest", "-q"], check=False))
    if args.eval:
        sys.exit(run([str(VENV_PY), "-m", "evaluation.run_eval", "--mode", "retrieval"], check=False))

    port = free_port(args.port)
    say(f"Starting the app at http://localhost:{port}  (press Ctrl+C in this window to stop)")
    try:
        run([str(VENV_PY), "-m", "streamlit", "run", "app/ui.py", "--server.port", str(port),
             "--server.address", "localhost"], check=False)
    except KeyboardInterrupt:
        pass
    say("Stopped.")


if __name__ == "__main__":
    main()
