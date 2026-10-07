"""Drive the real page (docs/index.html + app.js) in headless Chromium.

The Python engine is replaced by tests/web/stub_worker.js, which replays a result produced
by the real Python code, so this checks the page logic and the canvas picture, not Pyodide
itself (that is checked on GitHub, see .github/workflows/site.yml).
"""

import functools
import glob
import http.server
import json
import os
import threading
from pathlib import Path

import numpy as np
import pytest

playwright_sync = pytest.importorskip("playwright.sync_api")

import scenes  # noqa: E402
from stl_smoothing import stlio, web  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
STUB = (ROOT / "tests" / "web" / "stub_worker.js").read_bytes()


def _chromium_path():
    for pat in ("/opt/pw-browsers/chromium-*/chrome-linux*/chrome",
                "/opt/pw-browsers/chromium_headless_shell-*/chrome-linux*/headless_shell"):
        found = sorted(glob.glob(pat))
        if found:
            return found[-1]
    return None


@pytest.fixture(scope="session")
def browser():
    with playwright_sync.sync_playwright() as p:
        try:
            b = p.chromium.launch(args=["--no-sandbox"])
        except Exception:
            exe = _chromium_path()
            if not exe:
                pytest.skip("no Chromium available for Playwright")
            b = p.chromium.launch(executable_path=exe, args=["--no-sandbox"])
        yield b
        b.close()


class _Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *a):  # keep the test output clean
        pass


@pytest.fixture(scope="session")
def site():
    handler = functools.partial(_Quiet, directory=str(DOCS))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}/"
    server.shutdown()


def _make_payload(d, name, scene):
    model = d / f"{name}.stl"
    stlio.write_stl(model, scene.mesh.to_triangles())
    out = d / f"{name}_out.stl"
    meta = json.loads(web.process(str(model), str(out), json.dumps({"layer_height": 0.2}), preview_dir=str(d / f"pv_{name}")))
    assert meta["ok"] and meta["flattened"] == 1
    return {"model": model, "meta": meta, "out": out.read_bytes()}


@pytest.fixture(scope="session")
def payload(tmp_path_factory):
    """Real engine results: a wobbly panel with a dome, a plain wobbly slab, and a model with nothing to do."""
    d = tmp_path_factory.mktemp("payload")
    result = _make_payload(d, "relief", scenes.panel_with_dome(cell=1.0))
    result["slab"] = _make_payload(d, "slab", scenes.wavy_slab(cell=1.0))
    flat = scenes.heightfield_solid(0, 40, 0, 40, 2.0, lambda x, y: 9.8 + 0 * x, 0.0)
    flat_path = d / "flat.stl"
    stlio.write_stl(flat_path, flat.to_triangles())
    result["flat"] = flat_path
    result["nometa"] = json.loads(web.process(str(flat_path), str(d / "o2.stl"), "{}", preview_dir=str(d / "pv2")))
    return result


def open_page(browser, site, payload, mode=None, meta=None, output=True):
    """A fresh page whose engine is the stub; ``mode`` is {'fatal': ..} or {'error': ..}."""
    ctx = browser.new_context(accept_downloads=True, viewport={"width": 1100, "height": 900})
    page = ctx.new_page()
    logs = []
    page.on("console", lambda m: logs.append((m.type, m.text)))
    page.on("pageerror", lambda e: logs.append(("pageerror", str(e))))
    the_meta = meta if meta is not None else payload["meta"]

    def serve(route):
        url = route.request.url
        if url.endswith("/worker.js"):
            return route.fulfill(body=STUB, content_type="application/javascript")
        name = url.split("/__stub/")[1]
        if name == "mode.json":
            return route.fulfill(body=json.dumps(mode or {}), content_type="application/json")
        if name == "meta.json":
            return route.fulfill(body=json.dumps(the_meta), content_type="application/json")
        if name == "output.stl":
            return route.fulfill(body=payload["out"], content_type="application/octet-stream")
        if name.startswith("files/"):
            info = the_meta["preview"]["files"][name[len("files/"):-len(".bin")]]
            return route.fulfill(body=Path(info["file"]).read_bytes(), content_type="application/octet-stream")
        return route.abort()

    page.route("**/worker.js", serve)
    page.route("**/__stub/**", serve)
    page.goto(site)
    return page, ctx, logs


def canvas_stats(page, index):
    """Pixels of canvas #index: (opaque, orange-ish)."""
    return page.evaluate(
        """(i) => {
          const c = document.querySelectorAll('.canvas-box canvas')[i];
          const d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data;
          let opaque = 0, orange = 0, sum = 0;
          for (let k = 0; k < d.length; k += 4) {
            if (d[k + 3] > 0) opaque++;
            if (d[k] > 220 && d[k + 1] > 90 && d[k + 1] < 140 && d[k + 2] < 60) orange++;
            sum = (sum * 31 + d[k] + d[k + 1] * 3 + d[k + 2] * 7) % 1000003;
          }
          return [opaque, orange, sum, c.width, c.height];
        }""",
        index,
    )


def run_model(page, path):
    page.set_input_files("#file", str(path))
    page.click("#run")
    page.wait_for_selector("#results:not([hidden])", timeout=20000)


# --------------------------------------------------------------------------- tests
def test_page_loads_and_waits_for_a_file(browser, site, payload):
    page, ctx, logs = open_page(browser, site, payload)
    page.wait_for_function("document.getElementById('engine').dataset.state === 'ready'")
    assert page.title() == "STL Smoothing"
    assert page.is_disabled("#run")
    assert "Engine ready" in page.inner_text("#engine-text")
    page.set_input_files("#file", str(payload["model"]))
    assert page.is_enabled("#run")
    info = page.inner_text("#file-info")
    assert "relief.stl" in info and "triangles" in info
    assert [t for t, _ in logs if t in ("error", "pageerror")] == []
    ctx.close()


def test_full_run_shows_summary_stats_and_download(browser, site, payload):
    page, ctx, logs = open_page(browser, site, payload)
    page.wait_for_function("document.getElementById('engine').dataset.state === 'ready'")
    run_model(page, payload["model"])
    text = page.inner_text("#stats")
    assert "surface flattened" in text and "layer edges" in text and "→ 0 mm" in text
    summary = page.inner_text("#summary")
    assert summary == "\n".join(payload["meta"]["summary"])
    # the form values reached the engine
    echo = page.evaluate("window.__lastOptions || null")
    assert page.is_visible("#download")
    with page.expect_download() as dl:
        page.click("#download")
    download = dl.value
    assert download.suggested_filename == "relief_smoothed.stl"
    assert Path(download.path()).read_bytes() == payload["out"]
    assert [t for t, _ in logs if t in ("error", "pageerror")] == []
    ctx.close()


def test_the_picture_shows_layer_edges_before_and_none_after(browser, site, payload):
    slab = payload["slab"]
    page, ctx, logs = open_page(browser, site, slab)
    page.wait_for_function("document.getElementById('engine').dataset.state === 'ready'")
    run_model(page, slab["model"])
    page.wait_for_function("document.querySelectorAll('.canvas-box canvas').length === 2")
    page.wait_for_timeout(500)
    b_opaque, b_orange, *_ = canvas_stats(page, 0)
    a_opaque, a_orange, *_ = canvas_stats(page, 1)
    assert b_opaque > 5000 and a_opaque > 5000, "both pictures must be drawn"
    assert b_orange > 400, f"layer edges should be visible before ({b_orange})"
    assert a_orange < 0.05 * b_orange, f"the flattened surface has no layer edges left ({a_orange} vs {b_orange})"
    # legend and captions
    assert "different layers" in page.inner_text(".pane:nth-child(1) h4 small")
    assert "one layer" in page.inner_text(".pane:nth-child(2) h4 small")
    assert page.locator(".legend li i").count() >= 3
    page.screenshot(path=str(ROOT / "tests" / "web" / "last_ui.png"), full_page=True)
    ctx.close()


def test_a_dome_keeps_its_rings_in_both_pictures(browser, site, payload):
    page, ctx, _ = open_page(browser, site, payload)
    page.wait_for_function("document.getElementById('engine').dataset.state === 'ready'")
    run_model(page, payload["model"])
    page.wait_for_timeout(500)
    b = canvas_stats(page, 0)[1]
    a = canvas_stats(page, 1)[1]
    assert a > 1500, "the dome's own layer rings are real shape and stay"
    assert a < b, "but the panel's wobble lines are gone"
    ctx.close()


def test_zoom_redraws_and_reset_restores(browser, site, payload):
    page, ctx, _ = open_page(browser, site, payload)
    page.wait_for_function("document.getElementById('engine').dataset.state === 'ready'")
    run_model(page, payload["model"])
    page.wait_for_timeout(400)
    base = canvas_stats(page, 0)
    page.click("button[aria-label='Zoom in']")
    page.click("button[aria-label='Zoom in']")
    page.wait_for_timeout(600)
    zoomed = canvas_stats(page, 0)
    assert zoomed[2] != base[2], "zooming must redraw the picture"
    page.click("button[aria-label='Reset the zoom']")
    page.wait_for_timeout(400)
    assert canvas_stats(page, 0)[2] == base[2], "reset must restore the original picture"
    # wheel zoom and drag-pan also change the picture
    box = page.locator(".canvas-box").first.bounding_box()
    page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
    page.mouse.wheel(0, -400)
    page.wait_for_timeout(600)
    assert canvas_stats(page, 0)[2] != base[2]
    ctx.close()


def test_invalid_layer_height_is_reported_and_nothing_runs(browser, site, payload):
    page, ctx, _ = open_page(browser, site, payload)
    page.wait_for_function("document.getElementById('engine').dataset.state === 'ready'")
    page.set_input_files("#file", str(payload["model"]))
    page.fill("#layer", "")
    page.click("#run")
    assert page.is_visible("#error") and "Layer height" in page.inner_text("#error")
    assert page.get_attribute("#layer", "aria-invalid") == "true"
    page.fill("#layer", "0")
    page.click("#run")
    assert "between" in page.inner_text("#error")
    assert page.is_hidden("#results")
    page.fill("#layer", "0.2")
    page.click("#run")
    page.wait_for_selector("#results:not([hidden])")
    assert page.is_hidden("#error")
    ctx.close()


def test_engine_errors_are_shown(browser, site, payload):
    page, ctx, _ = open_page(browser, site, payload, mode={"error": "boom from python"})
    page.wait_for_function("document.getElementById('engine').dataset.state === 'ready'")
    page.set_input_files("#file", str(payload["model"]))
    page.click("#run")
    page.wait_for_selector("#error:not([hidden])")
    assert "boom from python" in page.inner_text("#error")
    assert page.is_enabled("#run"), "the user can try again"
    ctx.close()


def test_engine_that_cannot_start_is_explained(browser, site, payload):
    page, ctx, _ = open_page(browser, site, payload, mode={"fatal": "Could not download the Python engine"})
    page.wait_for_function("document.getElementById('engine').dataset.state === 'error'")
    assert "Could not download" in page.inner_text("#engine-text")
    page.set_input_files("#file", str(payload["model"]))
    assert page.is_disabled("#run")
    ctx.close()


def test_a_model_with_nothing_to_do_has_no_download(browser, site, payload):
    page, ctx, _ = open_page(browser, site, payload, meta=payload["nometa"])
    page.wait_for_function("document.getElementById('engine').dataset.state === 'ready'")
    run_model(page, payload["flat"])
    assert page.is_hidden("#download")
    assert "Nothing needed changing" in page.inner_text("#download-note")
    assert "0" in page.inner_text("#stats")
    assert page.locator(".canvas-box canvas").count() == 0
    ctx.close()


def test_failure_of_the_engine_response_is_ok_false(browser, site, payload):
    bad = {"ok": False, "error": "The file contains no triangles."}
    page, ctx, _ = open_page(browser, site, payload, meta=bad)
    page.wait_for_function("document.getElementById('engine').dataset.state === 'ready'")
    run_model_error = lambda: (page.set_input_files("#file", str(payload["model"])), page.click("#run"))  # noqa: E731
    run_model_error()
    page.wait_for_selector("#error:not([hidden])")
    assert "no triangles" in page.inner_text("#error")
    ctx.close()


def test_file_names_are_not_interpreted_as_html(browser, site, payload, tmp_path):
    evil = tmp_path / "<img src=x onerror=window.__pwned=1>.stl"
    evil.write_bytes(Path(payload["model"]).read_bytes())
    page, ctx, _ = open_page(browser, site, payload)
    page.wait_for_function("document.getElementById('engine').dataset.state === 'ready'")
    page.set_input_files("#file", str(evil))
    page.wait_for_timeout(200)
    assert page.evaluate("window.__pwned || null") is None
    assert "onerror" in page.inner_text("#drop-title")
    ctx.close()


def test_phone_width_has_no_horizontal_scroll(browser, site, payload):
    page, ctx, _ = open_page(browser, site, payload)
    page.set_viewport_size({"width": 360, "height": 800})
    page.wait_for_function("document.getElementById('engine').dataset.state === 'ready'")
    run_model(page, payload["model"])
    page.wait_for_timeout(500)
    assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth + 1")
    ctx.close()


def test_form_controls_have_accessible_names(browser, site, payload):
    page, ctx, _ = open_page(browser, site, payload)
    for label in ("Layer height (mm)", "First layer height (mm)", "Largest wobble to flatten (mm)",
                  "Ignore flat areas smaller than (mm²)"):
        assert page.get_by_label(label, exact=True).count() == 1, label
    assert page.get_by_role("button", name="Smooth the model").count() == 1
    assert page.locator("input[type=file]").count() == 1
    ctx.close()
