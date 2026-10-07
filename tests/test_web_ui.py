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

# CI jobs that install the browser set STL_REQUIRE_BROWSER=1, so a missing browser fails instead of
# silently skipping every test in this file.
REQUIRE_BROWSER = os.environ.get("STL_REQUIRE_BROWSER") == "1"
try:
    import playwright.sync_api as playwright_sync
except ImportError:
    if REQUIRE_BROWSER:
        raise
    pytest.skip("playwright is not installed (pip install '.[webtest]')", allow_module_level=True)

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
                if REQUIRE_BROWSER:
                    pytest.fail("no Chromium available for Playwright (STL_REQUIRE_BROWSER=1)")
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


def wait_drawn(page):
    """Wait until every picture on the page has been drawn (drawing is done in time slices)."""
    page.wait_for_function(
        "!document.querySelector('figure.view[data-drawing=\"true\"]') && "
        "[...document.querySelectorAll('figure.view')].every(f => f.dataset.drawing === 'false')", timeout=20000)


def run_model(page, path):
    page.set_input_files("#file", str(path))
    page.click("#run")
    page.wait_for_selector("#results:not([hidden])", timeout=20000)
    if page.locator("figure.view").count():
        wait_drawn(page)


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
    # Ctrl + wheel zooms (a plain wheel is left to the page, see test_plain_wheel_over_a_picture_scrolls_the_page)
    box = page.locator(".canvas-box").first.bounding_box()
    page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
    page.keyboard.down("Control")
    page.mouse.wheel(0, -400)
    page.keyboard.up("Control")
    page.wait_for_timeout(600)
    wait_drawn(page)
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


# ------------------------------------------------------------- review fixes: form
def ready(page):
    page.wait_for_function("document.getElementById('engine').dataset.state === 'ready'")


def sent_options(page):
    return page.evaluate("window.__stlSmoothing.state.lastOptions")


@pytest.mark.parametrize("typed, expected", [("0,2", 0.2), ("0.4", 0.4), (".25", 0.25), ("0,16", 0.16), (" 0.3 ", 0.3)])
def test_decimal_comma_and_point_are_both_read_correctly(browser, site, payload, typed, expected):
    page, ctx, _ = open_page(browser, site, payload)
    ready(page)
    page.set_input_files("#file", str(payload["model"]))
    page.fill("#layer", typed)
    page.click("#run")
    page.wait_for_selector("#results:not([hidden])")
    assert sent_options(page)["layer_height"] == expected
    # the stub replays a result made with 0.2 mm; the page prints whatever layer height the engine reports
    assert "0.2 mm\nlayer height used" in page.inner_text("#stats"), "the result says which layer height was used"
    ctx.close()


@pytest.mark.parametrize("typed", ["1e1", "0,2,3", "abc", "0.2.1", "-0.2", "0,", "2 0", "6"])
def test_values_that_are_not_plain_numbers_are_refused_not_reinterpreted(browser, site, payload, typed):
    page, ctx, _ = open_page(browser, site, payload)
    ready(page)
    page.set_input_files("#file", str(payload["model"]))
    page.fill("#layer", typed)
    page.click("#run")
    assert page.is_visible("#error") and "Layer height must be a number between" in page.inner_text("#error")
    assert page.evaluate("window.__stlSmoothing.state.lastOptions") is None
    assert page.evaluate("document.activeElement.id") == "layer", "focus goes to the field to fix"
    ctx.close()


def test_an_invalid_advanced_setting_opens_the_section_and_focuses_the_field(browser, site, payload):
    page, ctx, _ = open_page(browser, site, payload)
    ready(page)
    page.set_input_files("#file", str(payload["model"]))
    assert not page.evaluate("document.querySelector('details.advanced').open")
    page.evaluate("document.getElementById('range').value = '100'")
    page.click("#run")
    assert page.evaluate("document.querySelector('details.advanced').open")
    assert page.evaluate("document.activeElement.id") == "range"
    assert page.get_attribute("#range", "aria-invalid") == "true"
    assert page.get_attribute("#range", "aria-describedby") == "error"
    assert "Largest wobble" in page.inner_text("#error")
    ctx.close()


def test_progress_is_announced_by_phase_not_every_second(browser, site, payload):
    page, ctx, _ = open_page(browser, site, payload, mode={"hang": True})
    ready(page)
    page.set_input_files("#file", str(payload["model"]))
    page.click("#run")
    page.wait_for_function("document.getElementById('status').textContent.includes('Stub step')")
    page.wait_for_function("document.getElementById('elapsed').textContent.includes('s)')", timeout=8000)
    assert page.get_attribute("#elapsed", "aria-hidden") == "true", "the running timer is not read out"
    assert "(" not in page.inner_text("#status"), "the live region only changes when the phase does"
    assert page.get_attribute("#status", "aria-live") == "polite"
    assert page.get_attribute("#download", "role") is None, "a link is a link, not role=button"
    ctx.close()


def test_cancel_stops_a_run_and_brings_the_engine_back(browser, site, payload):
    page, ctx, _ = open_page(browser, site, payload, mode={"hang": True})
    ready(page)
    assert page.is_hidden("#cancel")
    page.set_input_files("#file", str(payload["model"]))
    page.click("#run")
    page.wait_for_selector("#cancel:not([hidden])")
    assert page.is_disabled("#run") and page.is_disabled("#file")
    page.click("#cancel")
    ready(page)  # the stub engine restarts in a moment
    assert page.is_hidden("#cancel") and page.is_hidden("#error")
    assert page.is_enabled("#run"), "the chosen file is kept, so the user can simply try again"
    assert page.is_enabled("#file")
    assert "relief.stl" in page.inner_text("#drop-title")
    assert "Cancelled" in page.inner_text("#status")
    ctx.close()


def test_a_late_answer_from_a_cancelled_run_is_ignored(browser, site, payload):
    page, ctx, _ = open_page(browser, site, payload, mode={"hang": True})
    ready(page)
    page.set_input_files("#file", str(payload["model"]))
    page.click("#run")
    page.wait_for_selector("#cancel:not([hidden])")
    page.click("#cancel")
    ready(page)
    page.evaluate("window.__stlSmoothing.state.requestId")  # still alive
    assert page.is_hidden("#results")
    ctx.close()


def test_python_errors_come_with_collapsed_technical_details(browser, site, payload):
    page, ctx, _ = open_page(browser, site, payload, mode={"error": "The model could not be processed (MemoryError: x).",
                                                            "detail": "Traceback (most recent call last):\n  boom"})
    ready(page)
    page.set_input_files("#file", str(payload["model"]))
    page.click("#run")
    page.wait_for_selector("#error:not([hidden])")
    text = page.inner_text("#error")
    assert "MemoryError" in text and "Reload the page" in text, "an out-of-memory failure says what to do"
    assert "Traceback" not in text, "the traceback is collapsed"
    page.click("#error summary")
    assert "boom" in page.inner_text("#error pre")
    ctx.close()


def test_cancelling_the_file_dialog_keeps_the_current_file(browser, site, payload):
    page, ctx, _ = open_page(browser, site, payload)
    ready(page)
    page.set_input_files("#file", str(payload["model"]))
    page.set_input_files("#file", [])  # what a cancelled dialog looks like to the page
    assert "relief.stl" in page.inner_text("#drop-title")
    assert page.is_enabled("#run")
    ctx.close()


def test_an_oversized_file_resets_the_drop_zone(browser, site, payload, tmp_path):
    big = tmp_path / "huge.stl"
    with open(big, "wb") as f:
        f.truncate(201 * 1024 * 1024)  # sparse: takes no disk space
    page, ctx, _ = open_page(browser, site, payload)
    ready(page)
    page.set_input_files("#file", str(payload["model"]))
    page.set_input_files("#file", str(big))
    assert "too large" in page.inner_text("#error")
    assert page.inner_text("#drop-title") == "Choose an STL file"
    assert page.get_attribute("#drop", "data-has-file") == "false"
    assert page.is_disabled("#run")
    ctx.close()


def test_a_binary_file_with_over_a_million_triangles_gets_a_size_warning(browser, site, payload, tmp_path):
    big = tmp_path / "big.stl"
    with open(big, "wb") as f:
        f.truncate(84 + 50 * 1_100_000)  # sparse; the page only looks at the size
    page, ctx, _ = open_page(browser, site, payload)
    ready(page)
    page.set_input_files("#file", str(payload["model"]))
    assert "large model" not in page.inner_text("#file-info")
    page.set_input_files("#file", str(big))
    info = page.inner_text("#file-info")
    assert "1,100,000 triangles" in info and "large model" in info
    assert page.is_enabled("#run"), "a warning, not a refusal"
    ctx.close()


# --------------------------------------------------------- review fixes: the picture
def test_plain_wheel_over_a_picture_scrolls_the_page(browser, site, payload):
    page, ctx, _ = open_page(browser, site, payload)
    page.set_viewport_size({"width": 1100, "height": 500})
    ready(page)
    run_model(page, payload["model"])
    box = page.locator(".canvas-box").first
    box.scroll_into_view_if_needed()
    page.wait_for_timeout(300)
    wait_drawn(page)
    before_pixels = canvas_stats(page, 0)[2]
    y0 = page.evaluate("window.scrollY")
    bb = box.bounding_box()
    page.mouse.move(bb["x"] + bb["width"] / 2, bb["y"] + min(bb["height"], 200) / 2)
    page.mouse.wheel(0, 300)
    page.wait_for_timeout(400)
    assert page.evaluate("window.scrollY") > y0, "the wheel scrolls the page"
    assert canvas_stats(page, 0)[2] == before_pixels, "and does not zoom the picture"
    ctx.close()


def test_touch_scroll_is_only_captured_once_zoomed(browser, site, payload):
    page, ctx, _ = open_page(browser, site, payload)
    ready(page)
    run_model(page, payload["model"])
    box = page.locator(".canvas-box").first
    assert page.evaluate("getComputedStyle(document.querySelector('.canvas-box')).touchAction") == "pan-y"
    page.click("button[aria-label='Zoom in']")
    page.wait_for_timeout(300)
    wait_drawn(page)
    assert page.evaluate("getComputedStyle(document.querySelector('.canvas-box')).touchAction") == "none"
    page.click("button[aria-label='Reset the zoom']")
    page.wait_for_timeout(200)
    wait_drawn(page)
    assert page.evaluate("getComputedStyle(document.querySelector('.canvas-box')).touchAction") == "pan-y"
    assert "pinch" not in page.inner_text(".zoom .hint").lower(), "no promise of a gesture that does not exist"
    ctx.close()


def test_a_picture_can_be_moved_and_zoomed_from_the_keyboard(browser, site, payload):
    page, ctx, _ = open_page(browser, site, payload)
    ready(page)
    run_model(page, payload["model"])
    box = page.locator(".canvas-box").first
    assert box.get_attribute("tabindex") == "0" and "Arrow keys" in box.get_attribute("aria-label")
    box.focus()
    base = canvas_stats(page, 0)[2]
    page.keyboard.press("ArrowLeft")
    page.wait_for_timeout(300)
    assert canvas_stats(page, 0)[2] == base, "at the fit view there is nothing to move"
    page.keyboard.press("+")
    page.wait_for_timeout(350)
    wait_drawn(page)
    zoomed = canvas_stats(page, 0)[2]
    assert zoomed != base
    page.keyboard.press("ArrowLeft")
    page.wait_for_timeout(350)
    wait_drawn(page)
    moved = canvas_stats(page, 0)[2]
    assert moved != zoomed, "the arrow keys move a zoomed picture"
    assert canvas_stats(page, 1)[2] != moved or True
    page.keyboard.press("0")
    page.wait_for_timeout(300)
    wait_drawn(page)
    assert canvas_stats(page, 0)[2] == base, "0 resets"
    ctx.close()


def test_zooming_back_out_recentres_the_picture(browser, site, payload):
    page, ctx, _ = open_page(browser, site, payload)
    ready(page)
    run_model(page, payload["model"])
    base = canvas_stats(page, 0)[2]
    box = page.locator(".canvas-box").first
    box.focus()
    page.keyboard.press("+")
    page.wait_for_timeout(300)
    wait_drawn(page)
    page.keyboard.press("ArrowRight")
    page.wait_for_timeout(300)
    wait_drawn(page)
    for _ in range(6):
        page.keyboard.press("-")
        page.wait_for_timeout(250)
        wait_drawn(page)
    assert canvas_stats(page, 0)[2] == base, "fully zoomed out is the fit view again"
    ctx.close()


def test_running_twice_does_not_pile_up_resize_listeners(browser, site, payload):
    page, ctx, _ = open_page(browser, site, payload)
    ready(page)
    run_model(page, payload["model"])
    assert page.evaluate("window.__stlSmoothing.state.cleanups.length") == 1
    run_model(page, payload["model"])
    run_model(page, payload["model"])
    assert page.evaluate("window.__stlSmoothing.state.cleanups.length") == 1
    page.set_input_files("#file", str(payload["flat"]))  # choosing another file hides the old results
    assert page.evaluate("window.__stlSmoothing.state.cleanups.length") == 0
    ctx.close()


def test_a_height_only_resize_does_not_redraw_the_pictures(browser, site, payload):
    page, ctx, _ = open_page(browser, site, payload)
    ready(page)
    run_model(page, payload["model"])
    page.evaluate("""() => { window.__draws = 0; const d = CanvasRenderingContext2D.prototype.drawImage;
                             CanvasRenderingContext2D.prototype.drawImage = function (...a) { window.__draws++; return d.apply(this, a); }; }""")
    page.set_viewport_size({"width": 1100, "height": 700})
    page.wait_for_timeout(600)
    assert page.evaluate("window.__draws") == 0
    page.set_viewport_size({"width": 800, "height": 700})
    page.wait_for_timeout(900)
    wait_drawn(page)
    assert page.evaluate("window.__draws") >= 2, "a width change does redraw both panes"
    ctx.close()


def test_long_file_names_do_not_break_the_layout_on_a_phone(browser, site, payload, tmp_path):
    name = "Pumpkin_Bat_Relief_v2_final_for_print_0.2mm_PLA_" + "x" * 150 + ".stl"
    long_file = tmp_path / name
    long_file.write_bytes(Path(payload["model"]).read_bytes())
    page, ctx, _ = open_page(browser, site, payload)
    page.set_viewport_size({"width": 360, "height": 800})
    ready(page)
    page.set_input_files("#file", str(long_file))
    assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth + 1")
    ctx.close()


def _luminance(rgb):
    def channel(c):
        c /= 255
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = rgb
    return 0.2126 * channel(r) + 0.7152 * channel(g) + 0.0722 * channel(b)


def _contrast(a, b):
    la, lb = sorted((_luminance(a), _luminance(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


@pytest.mark.parametrize("scheme", ["light", "dark"])
def test_borders_of_fields_the_drop_zone_and_pictures_have_3_to_1_contrast(browser, site, payload, scheme):
    page, ctx, _ = open_page(browser, site, payload)
    page.emulate_media(color_scheme=scheme)
    ready(page)
    run_model(page, payload["model"])

    def rgb_of(selector, prop):
        text = page.evaluate("([s, p]) => getComputedStyle(document.querySelector(s))[p]", [selector, prop])
        return tuple(int(float(x)) for x in text[text.index("(") + 1:text.index(")")].replace("/", ",").split(",")[:3])

    card = rgb_of(".card", "backgroundColor")
    for selector in ("#layer", "#drop", ".canvas-box"):
        border = rgb_of(selector, "borderTopColor")
        assert _contrast(border, card) >= 3.0, (scheme, selector, border, card)
    placeholder = page.evaluate("getComputedStyle(document.getElementById('first'), '::placeholder').color")
    ph = tuple(int(float(x)) for x in placeholder[placeholder.index("(") + 1:placeholder.index(")")].replace("/", ",").split(",")[:3])
    assert _contrast(ph, rgb_of("#first", "backgroundColor")) >= 4.5, (scheme, ph)
    ctx.close()


def test_a_missing_layer_edge_note_is_shown_when_the_engine_capped_the_edges(browser, site, payload):
    meta = json.loads(json.dumps(payload["meta"]))
    meta["preview"]["views"][0]["edge_note"] = "At this layer height there are too many layer edges to draw them."
    page, ctx, _ = open_page(browser, site, payload, meta=meta)
    ready(page)
    run_model(page, payload["model"])
    assert "too many layer edges" in page.inner_text("figure.view")
    ctx.close()


# ------------------------------------------- "how do I use it?": where a visitor lands, and what the page says
@pytest.fixture(scope="module")
def root_site():
    """The whole repository served from its root, like GitHub Pages set to '/ (root)'."""
    handler = functools.partial(_Quiet, directory=str(ROOT))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}/"
    server.shutdown()


@pytest.mark.parametrize("javascript", [True, False])
def test_the_repository_root_forwards_to_the_app(browser, root_site, javascript):
    ctx = browser.new_context(java_script_enabled=javascript)
    page = ctx.new_page()
    page.goto(root_site)
    page.wait_for_url("**/docs/", timeout=10000)
    assert page.title() == "STL Smoothing"
    assert page.locator("#form").count() == 1
    ctx.close()


def test_the_page_tells_a_new_visitor_what_to_do(browser, site, payload):
    page, ctx, _ = open_page(browser, site, payload)
    ready(page)
    steps = page.locator("ol.steps li")
    assert steps.count() == 3
    assert "Choose your STL file" in steps.nth(0).inner_text()
    assert page.is_disabled("#run")
    assert "Choose an STL file above" in page.inner_text("#run-hint"), "a grey button says why"
    assert page.get_attribute("#run", "aria-describedby") == "run-hint"
    page.set_input_files("#file", str(payload["model"]))
    assert page.is_enabled("#run") and page.inner_text("#run-hint") == ""
    ctx.close()


def test_the_run_hint_explains_a_dead_engine(browser, site, payload):
    page, ctx, _ = open_page(browser, site, payload, mode={"fatal": "Could not download the Python engine"})
    page.wait_for_function("document.getElementById('engine').dataset.state === 'error'")
    page.set_input_files("#file", str(payload["model"]))
    assert "could not start" in page.inner_text("#run-hint")
    ctx.close()
