"""End-to-end check of the site with the REAL Python engine (Pyodide in a real browser).

This needs internet (the engine and numpy/scipy come from cdn.jsdelivr.net), so it is not part of
the normal pytest run; GitHub Actions runs it (.github/workflows/site.yml).  It

  1. smooths a generated model natively (the reference),
  2. serves docs/ and opens the page in headless Chromium, waits for the engine,
  3. uploads the same model, waits for the result, downloads the smoothed STL,
  4. checks the browser result against the native one, and that bad files give a clean error.

Environment: E2E_OUT = directory for screenshots, logs and the downloaded STL (default e2e-out).
"""

import functools
import http.server
import json
import os
import re
import sys
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

import scenes  # noqa: E402
from playwright.sync_api import sync_playwright  # noqa: E402
from stl_smoothing import stlio, web  # noqa: E402

OUT = Path(os.environ.get("E2E_OUT", "e2e-out")).resolve()
OUT.mkdir(parents=True, exist_ok=True)
ENGINE_TIMEOUT_MS = 10 * 60 * 1000
RUN_TIMEOUT_S = 8 * 60
log_lines = []


def log(*a):
    line = " ".join(str(x) for x in a)
    print(line, flush=True)
    log_lines.append(line)


def wait_for_result(page, label: str) -> float:
    """Wait for the result card, logging every change of the page's status line (stage by stage),
    so a stall shows where it happened.  Returns the seconds taken; raises on error or timeout."""
    t0 = time.time()
    last = None
    while True:
        state = page.evaluate(
            """() => ({status: document.getElementById('status').textContent.trim(),
                       error: document.getElementById('error').hidden ? '' : document.getElementById('error').textContent.trim(),
                       results: !document.getElementById('results').hidden,
                       busy: document.getElementById('run').disabled})""")
        state["status"] = re.sub(r" \(\d+ s\)$", "", state["status"])  # the page appends a running timer
        key = (state["status"], state["error"], state["results"], state["busy"])
        if key != last:
            log(f"  [{label} +{time.time() - t0:5.1f}s] status={state['status']!r} busy={state['busy']} "
                f"results={state['results']} error={state['error']!r}")
            last = key
        if state["error"]:
            raise AssertionError(f"{label}: the page reported an error: {state['error']}")
        if state["results"]:
            return time.time() - t0
        if time.time() - t0 > RUN_TIMEOUT_S:
            raise TimeoutError(f"{label}: no result after {RUN_TIMEOUT_S} s (last status {state['status']!r})")
        page.wait_for_timeout(1000)


class Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *a):
        pass


def main() -> int:
    # ---------------------------------------------------------------- reference
    scene = scenes.panel_with_dome(cell=0.6, ptp=1.2, jitter=0.2)
    model = OUT / "e2e_model.stl"
    stlio.write_stl(model, scene.mesh.to_triangles(), header="e2e model")
    log(f"model: {scene.mesh.n_faces:,} triangles, {model.stat().st_size / 1e6:.1f} MB")
    t = time.time()
    native_out = OUT / "native_out.stl"
    native = json.loads(web.process(str(model), str(native_out), json.dumps({"layer_height": 0.2})))
    log(f"native: {time.time() - t:.1f} s; flattened={native['flattened']} moved={native['n_moved']} "
        f"edges {native['edges_before']:.0f}->{native['edges_after']:.0f}")
    assert native["ok"] and native["flattened"] == 1 and native["edges_after"] == 0

    # ------------------------------------------------------------------ browser
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(Quiet, directory=str(ROOT / "docs")))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    failure = None
    with sync_playwright() as p:
        browser = p.chromium.launch(args=["--no-sandbox"])
        ctx = browser.new_context(accept_downloads=True, viewport={"width": 1200, "height": 1000})
        page = ctx.new_page()
        page.on("console", lambda m: log(f"[console.{m.type}] {m.text}"))
        page.on("pageerror", lambda e: log(f"[pageerror] {e}"))
        page.on("requestfailed", lambda r: log(f"[requestfailed] {r.url} {r.failure}"))
        downloaded = {}

        def count(resp):
            host = resp.url.split("/")[2]
            size = int(resp.headers.get("content-length") or 0)
            downloaded[host] = downloaded.get(host, 0) + size

        page.on("response", count)
        try:
            t0 = time.time()
            page.goto(url)
            page.wait_for_function(
                "['ready','error'].includes(document.getElementById('engine').dataset.state)", timeout=ENGINE_TIMEOUT_MS)
            state = page.get_attribute("#engine", "data-state")
            engine_text = page.inner_text("#engine-text")
            log(f"engine state after {time.time() - t0:.0f} s: {state}: {engine_text}")
            log("engine download (content-length, uncompressed on the wire may differ): "
                + ", ".join(f"{h}: {b / 1e6:.1f} MB" for h, b in downloaded.items()))
            page.screenshot(path=str(OUT / "01_engine.png"))
            assert state == "ready", f"the engine did not start: {engine_text}"

            # ---- a bad file gives a clean message from the real Python code
            bad = OUT / "bad.stl"
            bad.write_bytes(b"this is not an stl file")
            page.set_input_files("#file", str(bad))
            page.click("#run")
            page.wait_for_selector("#error:not([hidden])", timeout=60000)
            msg = page.inner_text("#error")
            log("bad file ->", msg)
            assert "not a recognisable" in msg, msg

            # ---- a small model first: tells "the engine is slow" from "the engine is stuck"
            small_scene = scenes.panel_with_dome(cell=2.4, ptp=1.2, jitter=0.2)
            small = OUT / "e2e_small.stl"
            stlio.write_stl(small, small_scene.mesh.to_triangles(), header="e2e small model")
            small_native = json.loads(web.process(str(small), str(OUT / "small_native_out.stl"), json.dumps({"layer_height": 0.2})))
            log(f"small model: {small_scene.mesh.n_faces:,} triangles; native flattened={small_native['flattened']} "
                f"moved={small_native['n_moved']}")
            page.set_input_files("#file", str(small))
            page.click("#run")
            took = wait_for_result(page, "small")
            log(f"small model processed in the browser in {took:.1f} s")
            small_stats = page.inner_text("#stats")
            log("small stats:", small_stats.replace("\n", " | "))
            page.screenshot(path=str(OUT / "02a_small_result.png"), full_page=True)

            # ---- the real model
            page.set_input_files("#file", str(model))
            page.click("#run")
            took = wait_for_result(page, "large")
            log(f"processed in the browser in {took:.1f} s")
            page.wait_for_timeout(1500)
            page.screenshot(path=str(OUT / "02_result.png"), full_page=True)
            stats = page.inner_text("#stats")
            summary = page.inner_text("#summary")
            log("stats:", stats.replace("\n", " | "))
            assert "surface flattened" in stats and "→ 0 mm" in stats, stats
            with page.expect_download() as dl:
                page.click("#download")
            web_out = OUT / "web_out.stl"
            dl.value.save_as(str(web_out))
            assert dl.value.suggested_filename == "e2e_model_smoothed.stl"

            # ---- the picture was drawn (both canvases have pixels, edges before, far fewer after)
            def px(i):
                return page.evaluate(
                    """(i) => { const c = document.querySelectorAll('.canvas-box canvas')[i];
                      const d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data; let o = 0, e = 0;
                      for (let k = 0; k < d.length; k += 4) { if (d[k+3] > 0) o++;
                        if (d[k] > 220 && d[k+1] > 90 && d[k+1] < 140 && d[k+2] < 60) e++; } return [o, e]; }""", i)
            (bo, be), (ao, ae) = px(0), px(1)
            log(f"picture pixels: before opaque={bo} edge={be}; after opaque={ao} edge={ae}")
            assert bo > 5000 and ao > 5000 and be > 0 and ae < be

            # ---- same answer as the native code
            a = stlio.read_stl(native_out).tris
            b = stlio.read_stl(web_out).tris
            assert a.shape == b.shape, (a.shape, b.shape)
            close = np.abs(a - b).max(axis=(1, 2)) < 1e-4
            frac = float(close.mean())
            worst = float(np.abs(a - b).max())
            log(f"web vs native: {100 * frac:.3f}% of triangles identical (<1e-4 mm), worst difference {worst:.4f} mm")
            assert frac > 0.999, "the browser result differs from the native result"
            native_lines = native["summary"]
            web_lines = summary.split("\n")
            log("native summary:\n  " + "\n  ".join(native_lines[:3]))
            log("browser summary:\n  " + "\n  ".join(web_lines[:3]))
            assert web_lines[0] == native_lines[0], "summary heading differs"
            log("E2E OK")
        except Exception as exc:  # noqa: BLE001
            failure = exc
            try:
                page.screenshot(path=str(OUT / "failure.png"), full_page=True)
            except Exception:  # noqa: BLE001
                pass
            log(f"E2E FAILED: {type(exc).__name__}: {exc}")
        finally:
            browser.close()
            server.shutdown()
    (OUT / "e2e.log").write_text("\n".join(log_lines))
    return 1 if failure else 0


if __name__ == "__main__":
    sys.exit(main())
