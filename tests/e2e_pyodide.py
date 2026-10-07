"""End-to-end check of the site with the REAL Python engine (Pyodide in a real browser).

This needs internet (the engine and numpy/scipy come from cdn.jsdelivr.net), so it is not part of
the normal pytest run; GitHub Actions runs it (.github/workflows/site.yml).  For each model size it

  1. smooths a generated model natively (the reference),
  2. serves docs/ and opens the page in headless Chromium (a fresh page and engine per size), waits for it,
  3. uploads the same model, waits for the result, downloads the smoothed STL,
  4. checks the browser result against the native one and that the picture was drawn,

and, for the first size, that a bad file and a small model behave.  It also checks the privacy promise:
every request the page made was a GET to this server or the engine CDN, and none carried a body.

Environment:
  E2E_OUT     directory for screenshots, logs and the downloaded STLs (default e2e-out)
  E2E_SIZES   comma-separated grid cell sizes in mm of the generated model; smaller = more triangles
              (0.6 -> 111k triangles, 0.3 -> 445k, 0.2 -> 1.0M, 0.165 -> 1.46M, 0.12 -> 2.7M).  Default 0.6.
  E2E_EXTRAS  "0" skips the bad-file and small-model checks (the large-model job uses this)
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
from stl_smoothing.preview import MAX_PREVIEW_FACES as web_preview_limit  # noqa: E402

OUT = Path(os.environ.get("E2E_OUT", "e2e-out")).resolve()
OUT.mkdir(parents=True, exist_ok=True)
SIZES = [float(x) for x in os.environ.get("E2E_SIZES", "0.6").split(",") if x.strip()]
EXTRAS = os.environ.get("E2E_EXTRAS", "1") != "0"
ENGINE_TIMEOUT_MS = 10 * 60 * 1000
RUN_TIMEOUT_S = 12 * 60
ALLOWED_HOSTS = {"cdn.jsdelivr.net"}
log_lines = []


def log(*a):
    line = " ".join(str(x) for x in a)
    print(line, flush=True)
    log_lines.append(line)


class Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *a):
        pass


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


def wait_drawn(page, label: str):
    """The pictures are drawn in time slices; wait until all of them are finished."""
    t0 = time.time()
    page.wait_for_function(
        "[...document.querySelectorAll('figure.view')].length > 0 && "
        "[...document.querySelectorAll('figure.view')].every(f => f.dataset.drawing === 'false')",
        timeout=RUN_TIMEOUT_S * 1000)
    log(f"  [{label}] pictures drawn after {time.time() - t0:.1f} s")


def edge_pixels(page, i):
    return page.evaluate(
        """(i) => { const c = document.querySelectorAll('.canvas-box canvas')[i];
          const d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data; let o = 0, e = 0;
          for (let k = 0; k < d.length; k += 4) { if (d[k+3] > 0) o++;
            if (d[k] > 220 && d[k+1] > 90 && d[k+1] < 140 && d[k+2] < 60) e++; } return [o, e]; }""", i)


def exercise(p, url: str, cell: float, first: bool, requests: list) -> None:
    """One model size in a fresh page.  Raises on any failed check."""
    tag = f"cell{cell:g}"
    # ---------------------------------------------------------------- reference
    scene = scenes.panel_with_dome(cell=cell, ptp=1.2, jitter=0.2)
    model = OUT / f"e2e_model_{tag}.stl"
    stlio.write_stl(model, scene.mesh.to_triangles(), header="e2e model")
    n_tri = scene.mesh.n_faces
    del scene
    log(f"[{tag}] model: {n_tri:,} triangles, {model.stat().st_size / 1e6:.1f} MB")
    t = time.time()
    native_out = OUT / f"native_out_{tag}.stl"
    native = json.loads(web.process(str(model), str(native_out), json.dumps({"layer_height": 0.2})))
    native_s = time.time() - t
    log(f"[{tag}] native: {native_s:.1f} s; flattened={native['flattened']} moved={native['n_moved']} "
        f"edges {native['edges_before']:.0f}->{native['edges_after']:.0f}")
    assert native["ok"] and native["flattened"] == 1 and native["edges_after"] == 0

    # ------------------------------------------------------------------ browser
    browser = p.chromium.launch(args=["--no-sandbox"])
    try:
        ctx = browser.new_context(accept_downloads=True, viewport={"width": 1200, "height": 1000})
        page = ctx.new_page()
        page.on("console", lambda m: log(f"[console.{m.type}] {m.text}"))
        page.on("pageerror", lambda e: log(f"[pageerror] {e}"))
        page.on("requestfailed", lambda r: log(f"[requestfailed] {r.url} {r.failure}"))
        page.on("request", lambda r: requests.append((r.method, r.url.split("/")[2], bool(r.post_data_buffer))))
        downloaded = {}

        def count(resp):
            host = resp.url.split("/")[2]
            downloaded[host] = downloaded.get(host, 0) + int(resp.headers.get("content-length") or 0)

        page.on("response", count)
        try:
            t0 = time.time()
            page.goto(url)
            page.wait_for_function(
                "['ready','error'].includes(document.getElementById('engine').dataset.state)", timeout=ENGINE_TIMEOUT_MS)
            state = page.get_attribute("#engine", "data-state")
            engine_text = page.inner_text("#engine-text")
            log(f"[{tag}] engine state after {time.time() - t0:.0f} s: {state}: {engine_text}")
            log("  engine download (content-length): " + ", ".join(f"{h}: {b / 1e6:.1f} MB" for h, b in downloaded.items()))
            assert state == "ready", f"the engine did not start: {engine_text}"
            if first:
                page.screenshot(path=str(OUT / "01_engine.png"))

            if first and EXTRAS:
                # ---- a bad file gives a clean message from the real Python code
                bad = OUT / "bad.stl"
                bad.write_bytes(b"this is not an stl file")
                page.set_input_files("#file", str(bad))
                page.click("#run")
                page.wait_for_selector("#error:not([hidden])", timeout=60000)
                msg = page.inner_text("#error")
                log("  bad file ->", msg)
                assert "not a recognisable" in msg and msg.startswith("That file could not be read"), msg

                # ---- a small model first: tells "the engine is slow" from "the engine is stuck"
                small_scene = scenes.panel_with_dome(cell=2.4, ptp=1.2, jitter=0.2)
                small = OUT / "e2e_small.stl"
                stlio.write_stl(small, small_scene.mesh.to_triangles(), header="e2e small model")
                small_native = json.loads(web.process(str(small), str(OUT / "small_native_out.stl"), json.dumps({"layer_height": 0.2})))
                log(f"  small model: {small_scene.mesh.n_faces:,} triangles; native flattened={small_native['flattened']} "
                    f"moved={small_native['n_moved']}")
                page.set_input_files("#file", str(small))
                page.click("#run")
                took = wait_for_result(page, "small")
                log(f"  small model processed in the browser in {took:.1f} s: " + page.inner_text("#stats").replace("\n", " | "))

            # ---- the real model
            page.set_input_files("#file", str(model))
            page.click("#run")
            took = wait_for_result(page, tag)
            log(f"[{tag}] processed in the browser in {took:.1f} s (native {native_s:.1f} s)")
            if n_tri <= web_preview_limit:
                wait_drawn(page, tag)
            page.screenshot(path=str(OUT / f"02_result_{tag}.png"), full_page=True)
            stats = page.inner_text("#stats")
            summary = page.inner_text("#summary")
            log(f"[{tag}] stats:", stats.replace("\n", " | "))
            assert "surface flattened" in stats and "→ 0 mm" in stats and "0.2 mm\nlayer height used" in stats, stats
            with page.expect_download() as dl:
                page.click("#download")
            web_out = OUT / f"web_out_{tag}.stl"
            dl.value.save_as(str(web_out))
            assert dl.value.suggested_filename == f"e2e_model_{tag}_smoothed.stl", dl.value.suggested_filename

            # ---- the picture was drawn (models above the preview limit get no picture, by design)
            if n_tri > web_preview_limit:
                assert page.locator("figure.view").count() == 0 and "too large for the picture" in page.inner_text("#results")
                log(f"[{tag}] no picture, as expected above {web_preview_limit:,} triangles")
            else:
                (bo, be), (ao, ae) = edge_pixels(page, 0), edge_pixels(page, 1)
                log(f"[{tag}] picture pixels: before opaque={bo} edge={be}; after opaque={ao} edge={ae}")
                assert bo > 5000 and ao > 5000 and be > 0
                if abs(cell - 0.6) < 1e-9:
                    # With this scene the wobbling panels' edges disappear and the dome's rings stay, so the
                    # 'after' picture keeps about 89 % of the edge pixels (3155 of 3560 in the first real run).
                    # The band is wide enough for anti-aliasing differences between browsers and tight enough
                    # to notice a wrong picture.  (Finer meshes draw thinner lines, so the share differs there.)
                    assert 0.80 * be <= ae <= 0.95 * be, f"unexpected share of layer-edge pixels after smoothing ({ae} of {be})"
                else:
                    assert ae <= be, f"the 'after' picture must not have more layer edges than 'before' ({ae} vs {be})"

            # ---- same answer as the native code
            a = stlio.read_stl(native_out).tris
            b = stlio.read_stl(web_out).tris
            assert a.shape == b.shape, (a.shape, b.shape)
            close = np.abs(a - b).max(axis=(1, 2)) < 1e-4
            frac = float(close.mean())
            worst = float(np.abs(a - b).max())
            log(f"[{tag}] web vs native: {100 * frac:.3f}% of triangles identical (<1e-4 mm), worst difference {worst:.4f} mm")
            assert frac > 0.999, "the browser result differs from the native result"
            assert summary.split("\n")[0] == native["summary"][0], "summary heading differs"
            log(f"[{tag}] OK")
        except Exception:
            try:
                page.screenshot(path=str(OUT / f"failure_{tag}.png"), full_page=True)
            except Exception:  # noqa: BLE001
                pass
            raise
    finally:
        browser.close()


def main() -> int:
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(Quiet, directory=str(ROOT / "docs")))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    requests: list = []
    failures = []
    try:
        with sync_playwright() as p:
            for i, cell in enumerate(SIZES):
                try:
                    exercise(p, url, cell, i == 0, requests)
                except Exception as exc:  # noqa: BLE001
                    failures.append(f"cell {cell:g}: {type(exc).__name__}: {exc}")
                    log(f"[cell{cell:g}] FAILED: {type(exc).__name__}: {exc}")
    finally:
        server.shutdown()

    # ---- the privacy promise: nothing but GETs to this server and the engine CDN, none with a body
    hosts = {h for _, h, _ in requests}
    foreign = sorted(h for h in hosts if not h.startswith("127.0.0.1") and h not in ALLOWED_HOSTS)
    writes = [(m, h) for m, h, body in requests if m != "GET" or body]
    log(f"requests: {len(requests)} to {sorted(hosts)}; foreign hosts: {foreign}; non-GET or with a body: {writes}")
    if foreign:
        failures.append(f"requests to unexpected hosts: {foreign}")
    if writes:
        failures.append(f"requests that could carry data out: {writes[:5]}")

    log("E2E " + ("FAILED: " + "; ".join(failures) if failures else "OK"))
    (OUT / "e2e.log").write_text("\n".join(log_lines))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
