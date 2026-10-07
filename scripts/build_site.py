"""Copy the Python modules the web page needs into docs/py and write the manifest.

The page's worker fetches ``py/manifest.json`` and every listed file, writes them into the
Python engine's file system and imports them, so ``docs/py`` must match ``src``.  Run this
after changing the package; ``--check`` exits with 1 if ``docs/py`` is out of date.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src" / "stl_smoothing"
DEST = ROOT / "docs" / "py"

# Everything web.py needs.  cli.py, report.py and __main__.py are command-line only.
MODULES = [
    "__init__.py", "stlio.py", "mesh.py", "layers.py", "slicing.py", "params.py", "_compat.py",
    "_detect.py", "_deform.py", "flatten.py", "summary.py", "preview.py", "web.py",
]


def expected() -> dict[str, bytes]:
    files = {name: (SRC / name).read_bytes() for name in MODULES}
    digest = hashlib.sha1()
    for name in MODULES:
        digest.update(name.encode() + b"\0" + files[name] + b"\0")
    manifest = {"files": MODULES, "hash": digest.hexdigest()[:12]}
    out = {f"stl_smoothing/{n}": b for n, b in files.items()}
    out["manifest.json"] = (json.dumps(manifest, indent=2) + "\n").encode()
    return out


def current() -> dict[str, bytes]:
    if not DEST.exists():
        return {}
    return {
        str(p.relative_to(DEST)).replace("\\", "/"): p.read_bytes()
        for p in sorted(DEST.rglob("*"))
        if p.is_file() and "__pycache__" not in p.parts
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="only verify that docs/py is up to date")
    a = ap.parse_args()
    want = expected()
    if a.check:
        have = current()
        stale = sorted(set(want) ^ set(have) | {k for k in want if have.get(k) != want[k]})
        if stale:
            print("docs/py is out of date; run `python scripts/build_site.py`. Differences:", *stale, sep="\n  ")
            return 1
        print("docs/py is up to date")
        return 0
    if DEST.exists():
        shutil.rmtree(DEST)
    for rel, blob in want.items():
        path = DEST / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(blob)
    (ROOT / "docs" / ".nojekyll").write_text("")  # Jekyll would drop the underscore-prefixed modules
    print(f"wrote {len(want)} files to {DEST.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
