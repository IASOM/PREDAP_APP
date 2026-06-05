#!/usr/bin/env python3
import os
import sys
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parent.parent
PYTHON_BIN = os.environ.get("PYTHON_BIN", sys.executable)

if __name__ == "__main__":
    cli_path = PROJECT_ROOT / "predap_cli.py"
    if not cli_path.exists():
        raise FileNotFoundError(f"Cannot find CLI entry point: {cli_path}")
    args = [str(PYTHON_BIN), str(cli_path), *sys.argv[1:]]
    os.execv(str(PYTHON_BIN), args)
