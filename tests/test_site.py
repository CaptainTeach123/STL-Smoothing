"""The static site: it must be in sync with the package and hang together."""

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import scenes
from stl_smoothing import stlio

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"


def test_site_python_sources_are_in_sync():
    """docs/py is what the browser runs; it must match src/ (run scripts/build_site.py)."""
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "build_site.py"), "--check"], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


def test_nojekyll_marker_exists():
    # without it GitHub Pages' Jekyll step silently drops _detect.py and _deform.py
    assert (DOCS / ".nojekyll").exists()


def test_manifest_lists_every_file_and_nothing_command_line_only():
    manifest = json.loads((DOCS / "py" / "manifest.json").read_text())
    for name in manifest["files"]:
        assert (DOCS / "py" / "stl_smoothing" / name).exists()
    assert "cli.py" not in manifest["files"] and "report.py" not in manifest["files"]
    assert any(n.startswith("_") and n != "__init__.py" for n in manifest["files"])  # the underscore modules ship


def test_site_modules_run_alone_without_matplotlib(tmp_path):
    """Only docs/py on the path (as inside the browser): process() must work and import no plotting library."""
    m = scenes.wavy_slab(cell=2.0).mesh
    stl = tmp_path / "m.stl"
    stlio.write_stl(stl, m.to_triangles())
    code = f"""
import sys
sys.path.insert(0, {str(DOCS / 'py')!r})
import json
from stl_smoothing import web
meta = json.loads(web.process({str(stl)!r}, {str(tmp_path / 'o.stl')!r}, '{{}}', preview_dir={str(tmp_path / 'pv')!r}))
assert meta['ok'] and meta['flattened'] == 1, meta
assert 'matplotlib' not in sys.modules
assert 'stl_smoothing.cli' not in sys.modules and 'stl_smoothing.report' not in sys.modules
print('isolated ok')
"""
    r = subprocess.run([sys.executable, "-I", "-B", "-c", code], capture_output=True, text=True)
    assert r.returncode == 0 and "isolated ok" in r.stdout, r.stdout + r.stderr


def test_every_element_the_script_looks_up_exists():
    js = (DOCS / "app.js").read_text()
    html = (DOCS / "index.html").read_text()
    ids = set(re.findall(r'\$\("([\w-]+)"\)', js))
    assert ids, "no element lookups found"
    for i in ids:
        assert f'id="{i}"' in html, f"app.js looks up #{i}, which index.html does not define"


def test_html_is_self_contained_and_safe():
    html = (DOCS / "index.html").read_text()
    assert not re.search(r"\son\w+\s*=", html), "no inline event handlers"
    assert "http://" not in html.replace("http://www.w3.org", "")
    for ref in re.findall(r'(?:src|href)="([^"#:]+)"', html):
        if ref.startswith("data"):
            continue
        assert (DOCS / ref).exists(), f"index.html references missing file {ref}"
    # the only third-party host is the Python-engine CDN, and only in the worker
    assert "jsdelivr" not in (DOCS / "app.js").read_text()
    assert "cdn.jsdelivr.net/pyodide/" in (DOCS / "worker.js").read_text()


def test_worker_protocol_under_node():
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not installed")
    r = subprocess.run([node, str(ROOT / "tests" / "web" / "worker_protocol.mjs")], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "all checks passed" in r.stdout
