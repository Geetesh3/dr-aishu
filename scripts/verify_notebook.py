"""Executes the Colab notebook's code cells to verify it actually runs.

This is the real check: it runs each code cell in order in a throwaway copy of the
project, skipping the optional cells that are disabled by default and the Colab
`!`/`%` magics. It fails loudly on the first exception.

Usage:  python scripts/verify_notebook.py
"""
from __future__ import annotations

import json
import re
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NB = ROOT / "notebooks" / "Dr_Aishu_Training.ipynb"

MAGIC = re.compile(r"^\s*[!%]")
# Cells that are guarded by a False flag or need Colab/network.
SKIP_MARKERS = [
    "google.colab",
    "RUN_TRANSFORMER = False",
    "RETRAIN_WITH_LIVE_DATA = False",
    "RUN_APP = False",
    "REPO_URL",
    "dr_aishu_model",       # downloads via Colab
]


def strip_magics(src: str) -> str:
    out = []
    for line in src.split("\n"):
        if MAGIC.match(line):
            indent = line[: len(line) - len(line.lstrip())]
            out.append(f"{indent}pass")
        else:
            out.append(line)
    return "\n".join(out)


def main() -> int:
    nb = json.loads(NB.read_text())
    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]

    workdir = Path(tempfile.mkdtemp(prefix="draisu_nb_"))
    print(f"running notebook in {workdir}")

    # Copy the project files the notebook needs.
    for rel in ("app", "scripts", "data/training"):
        src = ROOT / rel
        dst = workdir / rel
        if src.is_dir():
            shutil.copytree(src, dst)
    (workdir / "app" / "__init__.py").touch(exist_ok=True)

    namespace = {"__name__": "__main__", "__file__": str(workdir / "notebook.py")}
    ran = skipped = 0

    for idx, cell in enumerate(code_cells):
        src = "".join(cell["source"])
        if any(marker in src for marker in SKIP_MARKERS):
            print(f"  cell {idx:2d}: SKIP (optional / Colab-only)")
            skipped += 1
            continue
        cleaned = strip_magics(src)
        # nbconvert-style: the notebook chdir's into the project
        import os
        os.chdir(workdir)
        try:
            exec(compile(cleaned, f"<cell {idx}>", "exec"), namespace)
            print(f"  cell {idx:2d}: OK")
            ran += 1
        except Exception:
            print(f"  cell {idx:2d}: FAILED")
            traceback.print_exc()
            return 1

    print(f"\n{ran} cells ran, {skipped} skipped (optional)")
    print("NOTEBOOK VERIFIED")
    shutil.rmtree(workdir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
