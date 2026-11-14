"""Build Colab notebooks from plain Python sources.

Sources live in notebooks/src/NN_name.py and use "# %%" to start a code cell and
"# %% [markdown]" to start a markdown cell (lines in markdown cells start with "# ").
Every notebook reads QUICK=1 from the environment to run a fast smoke version.

    python tools/build_notebooks.py            # build all
    python tools/build_notebooks.py 07         # build the ones whose name starts with 07
"""
import re
import sys
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent.parent
SRC, OUT = ROOT / "notebooks" / "src", ROOT / "notebooks"
REPO = "srivtx/ai-anatomy"


def cells(text):
    out, kind, buf = [], None, []

    def flush():
        if kind is None:
            return
        body = "\n".join(buf).strip("\n")
        if not body.strip():
            return
        if kind == "md":
            body = "\n".join(re.sub(r"^# ?", "", ln) for ln in body.splitlines())
            out.append(nbf.v4.new_markdown_cell(body))
        else:
            out.append(nbf.v4.new_code_cell(body))

    for line in text.splitlines():
        if line.startswith("# %%"):
            flush()
            kind, buf = ("md" if "[markdown]" in line else "code"), []
        else:
            buf.append(line)
    flush()
    return out


def build(path):
    nb = nbf.v4.new_notebook()
    name = path.stem
    badge = f"[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/{REPO}/blob/main/notebooks/{name}.ipynb)"
    nb.cells = [nbf.v4.new_markdown_cell(badge)] + cells(path.read_text())
    nb.metadata = {"kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"},
                   "colab": {"provenance": [], "gpuType": "T4"}, "accelerator": "GPU"}
    nbf.write(nb, OUT / f"{name}.ipynb")
    return name


if __name__ == "__main__":
    prefix = sys.argv[1] if len(sys.argv) > 1 else ""
    built = [build(p) for p in sorted(SRC.glob(f"{prefix}*.py"))]
    print(f"built {len(built)} notebooks: {', '.join(built)}")
