/* STL Smoothing: page logic.
   The Python engine runs in worker.js; this file handles the form, talks to the worker,
   and draws the before/after picture on canvases from the arrays the engine returns. */
(function () {
  "use strict";

  var $ = function (id) { return document.getElementById(id); };
  var els = {
    engine: $("engine"), engineText: $("engine-text"), form: $("form"), file: $("file"), drop: $("drop"),
    dropTitle: $("drop-title"), fileInfo: $("file-info"), layer: $("layer"), first: $("first"), range: $("range"),
    area: $("area"), snap: $("snap"), run: $("run"), cancel: $("cancel"), status: $("status"), elapsed: $("elapsed"),
    error: $("error"), runHint: $("run-hint"), advanced: document.querySelector("details.advanced"),
    results: $("results"), stats: $("stats"), notes: $("notes"), download: $("download"),
    downloadNote: $("download-note"), figures: $("figures"), summary: $("summary"),
  };

  var SOFT_LIMIT_TRIANGLES = 1000000;       // above this the page warns that the tab may run out of memory
  var HARD_LIMIT_BYTES = 200 * 1024 * 1024; // refuse files larger than this
  var EDGE_COLOUR = "#f97316";

  var state = {
    engineReady: false, busy: false, file: null, runFile: null, requestId: 0, downloadUrl: null, timer: null,
    started: 0, lastProgress: "", lastOptions: null, cleanups: [],
  };
  var fmt = new Intl.NumberFormat(undefined);

  // ----------------------------------------------------------------- engine
  var worker = null;

  function setEngine(kind, text) {
    els.engine.dataset.state = kind;
    els.engineText.textContent = text;
  }

  function startEngine() {
    if (location.protocol === "file:") {
      setEngine("error", "Browsers do not run the engine from a file on disk. Open the hosted page, or serve this folder " +
        "(for example: python -m http.server --directory docs) and open http://localhost:8000/.");
      return;
    }
    try {
      worker = new Worker("worker.js");
    } catch (err) {
      setEngine("error", "This browser could not start the engine (" + err.message + ").");
      return;
    }
    worker.onmessage = onWorkerMessage;
    worker.onmessageerror = function () {
      if (state.busy) {
        finishBusy();
        showError("The engine sent something this page could not read. Try again; if it keeps happening, reload the page.");
      }
    };
    worker.onerror = function (event) {
      var message = (event && event.message) || "the worker stopped unexpectedly";
      if (state.busy) {
        finishBusy();
        showError("The engine stopped: " + message + ". A very large model can exhaust the browser's memory; try a smaller one or the command-line version.");
        setEngine("error", "The engine stopped. Reload the page to restart it.");
      } else {
        setEngine("error", "The engine could not start: " + message);
      }
      state.engineReady = false;
      updateRunButton();
    };
    worker.postMessage({ type: "init" });
  }

  function cancelRun() {
    if (!state.busy) return;
    state.requestId += 1;  // anything the old worker still says is ignored
    worker.onmessage = worker.onmessageerror = worker.onerror = null;
    worker.terminate();
    finishBusy();
    state.engineReady = false;
    updateRunButton();
    setEngine("loading", "Run cancelled. Restarting the engine…");
    els.status.textContent = "Cancelled. Your file and settings are still here.";
    startEngine();
  }

  function onWorkerMessage(event) {
    var msg = event.data || {};
    switch (msg.type) {
      case "status":
        setEngine("loading", msg.text);
        break;
      case "ready":
        state.engineReady = true;
        setEngine("ready", "Engine ready (Python " + msg.python + ", Pyodide " + msg.pyodide + ").");
        updateRunButton();
        break;
      case "fatal":
        setEngine("error", msg.message);
        updateRunButton();
        break;
      case "progress":
        if (msg.id === state.requestId) {
          state.lastProgress = msg.text + "…";
          els.status.textContent = state.lastProgress;  // phases only: the timer is shown separately
        }
        break;
      case "result":
        if (msg.id === state.requestId) onResult(msg);
        break;
      case "error":
        if (msg.id === state.requestId) {
          finishBusy();
          showError("Something went wrong while processing the model: " + msg.message, msg.detail);
        }
        break;
    }
  }

  // ------------------------------------------------------------------- form
  function updateRunButton() {
    els.run.disabled = !(state.engineReady && state.file && !state.busy);
    // say why the button is grey, so it never looks broken
    var hint = "";
    if (!state.busy && els.run.disabled) {
      if (els.engine.dataset.state === "error") hint = "The engine could not start, see the message above.";
      else if (!state.file) hint = state.engineReady ? "Choose an STL file above to continue." : "Choose an STL file above; the engine is still starting.";
      else hint = "Waiting for the engine to start…";
    }
    els.runHint.textContent = hint;
  }

  function showError(text, detail) {
    if (/MemoryError|out of memory|Aborted\(/i.test(text + " " + (detail || ""))) {
      text += " The browser ran out of memory. Reload the page to free it, or use the command-line version for models this large.";
    }
    els.error.replaceChildren();
    var p = document.createElement("p");
    p.textContent = text;
    els.error.append(p);
    if (detail) {
      var d = document.createElement("details");
      var s = document.createElement("summary");
      s.textContent = "Technical details";
      var pre = document.createElement("pre");
      pre.textContent = detail;
      d.append(s, pre);
      els.error.append(d);
    }
    els.error.hidden = false;
  }

  function clearError() {
    els.error.hidden = true;
    els.error.replaceChildren();
  }

  function stem(name) {
    return name.replace(/\.[^.]*$/, "") || "model";
  }

  function describeFile(file, triangles) {
    var size = (file.size / (1024 * 1024)).toFixed(file.size > 10 * 1024 * 1024 ? 0 : 1);
    var text = file.name + " · " + size + " MB";
    if (triangles) text += " · " + fmt.format(triangles) + " triangles";
    return text;
  }

  function resetFileDisplay() {
    els.fileInfo.textContent = "";
    els.dropTitle.textContent = "Choose an STL file";
    els.drop.dataset.hasFile = "false";
  }

  function chooseFile(file) {
    if (!file) return;  // the file dialog was cancelled: keep whatever was chosen before
    clearError();
    hideResults();
    state.file = null;
    if (file.size > HARD_LIMIT_BYTES) {
      resetFileDisplay();
      showError("That file is " + (file.size / 1048576).toFixed(0) + " MB, too large for a browser tab. Use the command-line version instead.");
      updateRunButton();
      return;
    }
    state.file = file;
    els.drop.dataset.hasFile = "true";
    els.dropTitle.textContent = file.name;
    // binary STL: size = 84 + 50 * triangles, so the count is known without reading the file
    // (an ASCII STL takes about 250 bytes per triangle, so one under the size limit stays below the warning)
    var triangles = file.size >= 84 && (file.size - 84) % 50 === 0 ? (file.size - 84) / 50 : 0;
    els.fileInfo.textContent = describeFile(file, triangles);
    if (triangles > SOFT_LIMIT_TRIANGLES) {
      els.fileInfo.textContent += ". This is a large model; the browser may run out of memory. The command-line version handles it better.";
    }
    updateRunButton();
  }

  function hideResults() {
    state.cleanups.splice(0).forEach(function (fn) { fn(); });
    els.results.hidden = true;
  }

  els.file.addEventListener("change", function () {
    chooseFile(els.file.files && els.file.files[0]);
  });

  ["dragenter", "dragover"].forEach(function (name) {
    els.drop.addEventListener(name, function (e) {
      e.preventDefault();
      els.drop.classList.add("over");
    });
  });
  ["dragleave", "drop"].forEach(function (name) {
    els.drop.addEventListener(name, function (e) {
      e.preventDefault();
      els.drop.classList.remove("over");
    });
  });
  els.drop.addEventListener("drop", function (e) {
    if (state.busy) return;
    var files = e.dataTransfer && e.dataTransfer.files;
    if (files && files.length) chooseFile(files[0]);
  });

  function fieldError(input, message) {
    var err = new Error(message);
    err.input = input;
    return err;
  }

  // Accepts "0.2" and "0,2" (a decimal comma); anything else that is not a plain number is refused,
  // so a typo can never turn into a different, valid-looking value.
  function readNumber(input, label, min, max, optional) {
    var raw = input.value.trim();
    input.removeAttribute("aria-invalid");
    if (raw === "") {
      if (optional) return null;
      throw fieldError(input, label + " is required.");
    }
    var normalised = raw.replace(/^(\d*),(\d+)$/, "$1.$2");
    var value = /^(\d+\.?\d*|\.\d+)$/.test(normalised) ? Number(normalised) : NaN;
    if (!isFinite(value) || value < min || value > max) {
      throw fieldError(input, label + " must be a number between " + min + " and " + max + " (for example 0.2).");
    }
    return value;
  }

  function readOptions() {
    [els.layer, els.first, els.range, els.area].forEach(function (input) { input.removeAttribute("aria-invalid"); });
    return {
      layer_height: readNumber(els.layer, "Layer height", 0.005, 5, false),
      first_layer: readNumber(els.first, "First layer height", 0.005, 5, true),
      max_range: readNumber(els.range, "Largest wobble", 0.05, 20, true),
      min_area: readNumber(els.area, "Smallest area", 0, 100000, true),
      snap_exact: els.snap.checked,
    };
  }

  function reportFieldError(err) {
    showError(err.message);
    var input = err.input;
    if (!input) return;
    input.setAttribute("aria-invalid", "true");
    input.setAttribute("aria-describedby", "error");
    if (els.advanced && els.advanced.contains(input)) els.advanced.open = true;
    input.focus();
  }

  function finishBusy() {
    state.busy = false;
    clearInterval(state.timer);
    els.status.textContent = "";
    els.elapsed.textContent = "";
    els.cancel.hidden = true;
    els.file.disabled = false;
    updateRunButton();
  }

  els.cancel.addEventListener("click", cancelRun);

  els.form.addEventListener("submit", function (event) {
    event.preventDefault();
    if (!state.engineReady || !state.file || state.busy) return;
    clearError();
    var options;
    try {
      options = readOptions();
    } catch (err) {
      reportFieldError(err);
      return;
    }
    state.lastOptions = options;
    try {
      localStorage.setItem("stl-smoothing-layer", String(options.layer_height));
    } catch (_) { /* storage may be unavailable */ }

    state.busy = true;
    state.requestId += 1;
    var id = state.requestId;
    state.started = Date.now();
    els.run.disabled = true;
    els.cancel.hidden = false;
    hideResults();
    els.status.textContent = "Reading the file…";
    state.lastProgress = "Reading the file…";
    state.timer = setInterval(function () {
      var seconds = Math.round((Date.now() - state.started) / 1000);
      els.elapsed.textContent = seconds >= 3 ? "(" + seconds + " s)" : "";
    }, 1000);

    var file = state.file;
    state.runFile = file;
    els.file.disabled = true;  // keep the chosen file stable while it is being processed
    file.arrayBuffer().then(function (buffer) {
      if (id !== state.requestId) return;
      worker.postMessage({ type: "run", id: id, buffer: buffer, options: options }, [buffer]);
    }).catch(function (err) {
      finishBusy();
      showError("Could not read that file: " + err.message);
    });
  });

  // ---------------------------------------------------------------- results
  function stat(value, label) {
    var box = document.createElement("div");
    box.className = "stat";
    var b = document.createElement("b");
    b.textContent = value;
    var s = document.createElement("span");
    s.textContent = label;
    box.append(b, s);
    return box;
  }

  function note(text) {
    var p = document.createElement("p");
    p.className = "note";
    p.textContent = text;
    return p;
  }

  function onResult(msg) {
    var seconds = ((Date.now() - state.started) / 1000).toFixed(1);
    finishBusy();
    var meta = msg.meta;
    if (!meta.ok) {
      showError(meta.error || "The file could not be processed.", meta.detail);
      return;
    }
    hideResults();  // drops the previous figures' listeners before they are replaced
    els.results.hidden = false;
    els.stats.replaceChildren();
    els.notes.replaceChildren();
    els.figures.replaceChildren();
    els.summary.textContent = meta.summary.join("\n");

    var flattened = meta.flattened;
    els.stats.append(stat(fmt.format(meta.triangles), "triangles"));
    els.stats.append(stat(String(meta.layer_height) + " mm", "layer height used"));
    els.stats.append(stat(String(flattened), flattened === 1 ? "surface flattened" : "surfaces flattened"));
    if (meta.edges_before !== null && meta.edges_before !== undefined) {
      els.stats.append(stat(fmt.format(Math.round(meta.edges_before)) + " → " + fmt.format(Math.round(meta.edges_after)) + " mm", "layer edges on those surfaces"));
    }
    els.stats.append(stat(fmt.format(meta.n_moved), "vertices moved (z only)"));
    els.stats.append(stat(seconds + " s", "time taken"));

    if (!meta.closed) {
      els.notes.append(note("The mesh is not closed (" + fmt.format(meta.open_edges) + " open edges). Results may be incomplete."));
    }
    if (meta.flipped > 0) {
      els.notes.append(note(meta.flipped + " faces would have flipped; they were left as they were."));
    }
    if (meta.preview_skipped) els.notes.append(note(meta.preview_skipped));

    if (state.downloadUrl) URL.revokeObjectURL(state.downloadUrl);
    state.downloadUrl = null;
    if (msg.output) {
      var blob = new Blob([msg.output], { type: "model/stl" });
      state.downloadUrl = URL.createObjectURL(blob);
      els.download.href = state.downloadUrl;
      els.download.setAttribute("download", stem(state.runFile.name) + "_smoothed.stl");
      els.download.hidden = false;
      els.downloadNote.textContent = fmt.format(Math.round(msg.output.byteLength / 1024)) + " KB";
    } else {
      els.download.hidden = true;
      els.downloadNote.textContent = "Nothing needed changing, so there is no new file to download. " +
        "If you expected changes, check the layer height, or raise “Largest wobble to flatten” in the advanced settings.";
    }

    if (meta.preview && meta.preview.views && meta.preview.views.length) {
      var data = {};
      Object.keys(meta.preview.files).forEach(function (name) {
        var info = meta.preview.files[name];
        var buf = msg.files[name];
        data[name] = info.dtype === "float32" ? new Float32Array(buf)
          : info.dtype === "uint32" ? new Uint32Array(buf) : new Uint8Array(buf);
      });
      meta.preview.views.forEach(function (view) {
        els.figures.append(buildFigure(view, data, meta.preview.bbox, meta.layer_height));
      });
    } else if (msg.output && !meta.preview_skipped) {
      els.figures.append(note("No surface changed layers; the model was only nudged slightly, so there is no picture."));
    }
    els.results.scrollIntoView({ behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth", block: "start" });
    els.results.focus({ preventScroll: true });
  }

  // ---------------------------------------------------------------- drawing
  var greys = [];
  for (var g = 0; g < 256; g++) greys.push("rgb(" + g + "," + g + "," + g + ")");

  function buildFigure(view, data, bbox, layerHeight) {
    var fig = document.createElement("figure");
    fig.className = "view";
    var title = document.createElement("h3");
    title.textContent = view.what;
    fig.append(title);

    var panes = document.createElement("div");
    panes.className = "panes";
    fig.append(panes);

    var shared = { zoom: 1, cx: 0.5, cy: 0.5 };
    var paneList = [];
    [["before", "Before", view.before_text], ["after", "After", view.after_text]].forEach(function (spec) {
      var pane = document.createElement("div");
      pane.className = "pane";
      var h = document.createElement("h4");
      h.textContent = spec[1];
      var small = document.createElement("small");
      small.textContent = spec[2];
      h.append(small);
      var box = document.createElement("div");
      box.className = "canvas-box";
      box.tabIndex = 0;
      box.setAttribute("role", "group");
      box.setAttribute("aria-label", spec[1] + " picture. Arrow keys move the view, plus and minus zoom, 0 resets.");
      var canvas = document.createElement("canvas");
      canvas.setAttribute("role", "img");
      canvas.setAttribute("aria-label", spec[1] + ": " + spec[2] + ". Layer edges are drawn in orange.");
      box.append(canvas);
      pane.append(h, box);
      panes.append(pane);
      paneList.push({ key: spec[0], label: spec[1], canvas: canvas, box: box, buffer: null, info: null });
    });

    var legend = document.createElement("ul");
    legend.className = "legend";
    legend.setAttribute("aria-label", "Layer colours");
    var legendTitle = document.createElement("li");
    legendTitle.className = "legend-title";
    legendTitle.textContent = "Layer the surface prints on, at " + layerHeight + " mm layers:";
    legend.append(legendTitle);
    view.levels.forEach(function (level, i) {
      var li = document.createElement("li");
      var sw = document.createElement("i");
      sw.style.background = view.colours[i];
      li.append(sw, document.createTextNode(level.toFixed(2) + " mm"));
      legend.append(li);
    });
    var edge = document.createElement("li");
    var line = document.createElement("i");
    line.className = "edge";
    edge.append(line, document.createTextNode("layer edge (a contour line in the slicer)"));
    legend.append(edge);
    fig.append(legend);
    if (view.edge_note) {
      var edgeNote = document.createElement("p");
      edgeNote.className = "hint";
      edgeNote.textContent = view.edge_note;
      fig.append(edgeNote);
    }

    var zoom = document.createElement("div");
    zoom.className = "zoom";
    var hint = document.createElement("span");
    hint.className = "hint";
    hint.textContent = "Zoom with + and −, or Ctrl + scroll. Drag to move the picture once zoomed (or click it and use the arrow keys). Double-click resets.";
    ["+", "−", "Reset"].forEach(function (label) {
      var b = document.createElement("button");
      b.type = "button";
      b.textContent = label;
      b.setAttribute("aria-label", label === "+" ? "Zoom in" : label === "−" ? "Zoom out" : "Reset the zoom");
      b.addEventListener("click", function () {
        if (label === "Reset") resetView();
        else zoomBy(label === "+" ? 1.6 : 1 / 1.6);
      });
      zoom.append(b);
    });
    zoom.append(hint);
    fig.append(zoom);

    // geometry shared by the two panes
    var bw = Math.max(bbox[2] - bbox[0], 1e-9);
    var bh = Math.max(bbox[3] - bbox[1], 1e-9);

    function layout(p) {
      var cssW = Math.max(p.box.clientWidth, 200);
      var cssH = Math.round(Math.min(Math.max(cssW * bh / bw, 160), 640));
      var dpr = Math.min(window.devicePixelRatio || 1, 2);
      var info = { cssW: cssW, cssH: cssH, dpr: dpr, W: Math.round(cssW * dpr), H: Math.round(cssH * dpr) };
      var changed = !p.info || p.info.W !== info.W || p.info.H !== info.H;
      p.info = info;
      if (changed) {  // assigning a canvas size clears it, so only do it when the size really changed
        p.canvas.width = info.W;
        p.canvas.height = info.H;
        p.canvas.style.width = cssW + "px";
        p.canvas.style.height = cssH + "px";
        p.box.style.height = cssH + "px";
      }
    }

    function mapping(p) {
      var W = p.info.W, H = p.info.H, pad = 10 * p.info.dpr;
      var s0 = Math.min((W - 2 * pad) / bw, (H - 2 * pad) / bh);
      var s = s0 * shared.zoom;
      return {
        s: s, W: W, H: H,
        wx: bbox[0] + shared.cx * bw, wy: bbox[1] + shared.cy * bh,
        mx: view.mirror_x ? -1 : 1,
      };
    }

    // A picture is drawn in slices of about 8 ms into an off-screen canvas, then swapped in, so the page
    // stays responsive on big models and the old picture stays on screen until the new one is complete.
    var generation = 0, drawing = 0;

    function draw(p, gen, done) {
      if (!p.buffer) p.buffer = document.createElement("canvas");
      var buf = p.buffer;
      if (buf.width !== p.info.W || buf.height !== p.info.H) { buf.width = p.info.W; buf.height = p.info.H; }
      var ctx = buf.getContext("2d");
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.clearRect(0, 0, p.info.W, p.info.H);
      var m = mapping(p);
      var key = view.key + "_" + p.key + "_";
      var order = data[key + "order"], shade = data[key + "shade"], level = data[key + "level"], segs = data[key + "segs"], xy = data.xy;
      var W = m.W, H = m.H, s = m.s, hw = W / 2, hh = H / 2, sx = s * m.mx;
      ctx.lineJoin = "round";
      ctx.lineWidth = 0.7 * p.info.dpr; // a hairline in the fill colour hides seams between triangles
      var last = "";
      var k = 0;

      function finish() {
        var visible = p.canvas.getContext("2d");
        visible.setTransform(1, 0, 0, 1, 0, 0);
        visible.clearRect(0, 0, p.info.W, p.info.H);
        visible.drawImage(buf, 0, 0);
        p.canvas.style.transform = "";
        done();
      }

      function drawEdges() {
        ctx.strokeStyle = EDGE_COLOUR;
        ctx.lineWidth = 0.8 * p.info.dpr;
        ctx.beginPath();
        for (var j = 0; j < segs.length; j += 4) {
          var ax = hw + (segs[j] - m.wx) * sx, ay = hh - (segs[j + 1] - m.wy) * s;
          var bx = hw + (segs[j + 2] - m.wx) * sx, by = hh - (segs[j + 3] - m.wy) * s;
          if ((ax < 0 && bx < 0) || (ay < 0 && by < 0) || (ax > W && bx > W) || (ay > H && by > H)) continue;
          ctx.moveTo(ax, ay); ctx.lineTo(bx, by);
        }
        ctx.stroke();
        finish();
      }

      function slice() {
        if (gen !== generation) return;  // a newer redraw superseded this one
        var t0 = performance.now();
        for (; k < order.length; k++) {
          if ((k & 255) === 0 && performance.now() - t0 > 8) { requestAnimationFrame(slice); return; }
          var f = order[k], b = f * 6;
          var x0 = hw + (xy[b] - m.wx) * sx, y0 = hh - (xy[b + 1] - m.wy) * s;
          var x1 = hw + (xy[b + 2] - m.wx) * sx, y1 = hh - (xy[b + 3] - m.wy) * s;
          var x2 = hw + (xy[b + 4] - m.wx) * sx, y2 = hh - (xy[b + 5] - m.wy) * s;
          if ((x0 < 0 && x1 < 0 && x2 < 0) || (y0 < 0 && y1 < 0 && y2 < 0) ||
              (x0 > W && x1 > W && x2 > W) || (y0 > H && y1 > H && y2 > H)) continue;
          var li = level[f];
          var style = li === 255 ? greys[shade[f]] : view.colours[li];
          if (style !== last) { ctx.fillStyle = style; ctx.strokeStyle = style; last = style; }
          ctx.beginPath();
          ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.lineTo(x2, y2); ctx.closePath();
          ctx.fill(); ctx.stroke();
        }
        drawEdges();
      }
      slice();
    }

    function updateTouchAction() {
      // a one-finger swipe scrolls the page while the picture is at 1x; once zoomed it moves the picture
      var zoomed = shared.zoom * pending.k > 1.0001;
      paneList.forEach(function (p) { p.box.classList.toggle("zoomed", zoomed); });
    }

    function redraw() {
      generation += 1;
      var gen = generation;
      drawing = paneList.length;
      fig.dataset.drawing = "true";
      pending = { k: 1, tx: 0, ty: 0 };
      updateTouchAction();
      paneList.forEach(function (p) {
        layout(p);
        draw(p, gen, function () {
          if (gen !== generation) return;
          drawing -= 1;
          if (drawing === 0) fig.dataset.drawing = "false";
        });
      });
    }

    function resetView() {
      shared.zoom = 1; shared.cx = 0.5; shared.cy = 0.5;
      redraw();
    }

    // --- zoom / pan: move the existing bitmap while interacting, redraw once when idle.
    // Gestures that arrive while a picture is still being drawn are ignored, so the transform in
    // flight always refers to the bitmap on screen.
    var pending = { k: 1, tx: 0, ty: 0 };
    var settleTimer = null;

    function applyPending() {
      var t = "translate(" + pending.tx + "px," + pending.ty + "px) scale(" + pending.k + ")";
      paneList.forEach(function (p) { p.canvas.style.transform = t; });
      updateTouchAction();
      clearTimeout(settleTimer);
      settleTimer = setTimeout(commit, 160);
    }

    function commit() {
      if (pending.k === 1 && pending.tx === 0 && pending.ty === 0) return;
      var p0 = paneList[0], m = mapping(p0);
      var dpr = p0.info.dpr;
      // the point of the old bitmap that is now at the centre of the box
      var qx = (p0.info.cssW / 2 - pending.tx) / pending.k * dpr;
      var qy = (p0.info.cssH / 2 - pending.ty) / pending.k * dpr;
      var wx = m.wx + (qx - m.W / 2) / (m.s * m.mx);
      var wy = m.wy - (qy - m.H / 2) / m.s;
      shared.zoom = Math.min(Math.max(shared.zoom * pending.k, 1), 400);
      if (shared.zoom <= 1.0001) {  // fully zoomed out is the "fit" view: centre it again
        shared.zoom = 1; shared.cx = 0.5; shared.cy = 0.5;
      } else {
        shared.cx = (wx - bbox[0]) / bw;
        shared.cy = (wy - bbox[1]) / bh;
      }
      redraw();
    }

    function zoomAbout(factor, px, py) {
      if (drawing) return;
      var k = Math.min(Math.max(shared.zoom * pending.k * factor, 1), 400) / (shared.zoom * pending.k);
      if (k === 1) return;
      pending.tx = px - k * (px - pending.tx);
      pending.ty = py - k * (py - pending.ty);
      pending.k *= k;
      applyPending();
    }

    function zoomBy(factor) {
      var p0 = paneList[0];
      zoomAbout(factor, p0.info.cssW / 2, p0.info.cssH / 2);
    }

    function panBy(dx, dy) {
      if (drawing || shared.zoom * pending.k <= 1.0001) return;  // nothing to move at the fit view
      pending.tx += dx;
      pending.ty += dy;
      applyPending();
    }

    paneList.forEach(function (p) {
      var drag = null;
      p.box.addEventListener("wheel", function (e) {
        if (!(e.ctrlKey || e.metaKey)) return;  // plain scrolling belongs to the page; Ctrl/pinch-on-trackpad zooms
        e.preventDefault();
        var r = p.box.getBoundingClientRect();
        zoomAbout(Math.exp(-e.deltaY * (e.deltaMode === 1 ? 0.05 : 0.0015)), e.clientX - r.left, e.clientY - r.top);
      }, { passive: false });
      p.box.addEventListener("pointerdown", function (e) {
        drag = { x: e.clientX, y: e.clientY };
        p.box.setPointerCapture(e.pointerId);
        p.box.classList.add("dragging");
      });
      p.box.addEventListener("pointermove", function (e) {
        if (!drag) return;
        panBy(e.clientX - drag.x, e.clientY - drag.y);
        drag = { x: e.clientX, y: e.clientY };
      });
      var end = function () { drag = null; p.box.classList.remove("dragging"); };
      p.box.addEventListener("pointerup", end);
      p.box.addEventListener("pointercancel", end);
      p.box.addEventListener("dblclick", resetView);
      p.box.addEventListener("keydown", function (e) {
        if (e.altKey || e.ctrlKey || e.metaKey) return;
        var step = 0.2 * Math.min(p.info.cssW, p.info.cssH);
        var handled = true;
        switch (e.key) {
          case "ArrowLeft": panBy(step, 0); break;
          case "ArrowRight": panBy(-step, 0); break;
          case "ArrowUp": panBy(0, step); break;
          case "ArrowDown": panBy(0, -step); break;
          case "+": case "=": zoomBy(1.6); break;
          case "-": case "_": zoomBy(1 / 1.6); break;
          case "0": resetView(); break;
          default: handled = false;
        }
        if (handled) e.preventDefault();
      });
    });

    // draw once the figure is in the page (widths are known), and again if the window gets a new width
    requestAnimationFrame(function () { redraw(); });
    var resizeTimer = null;
    function onResize() {
      clearTimeout(resizeTimer);
      resizeTimer = setTimeout(function () {
        if (!document.body.contains(fig)) return;
        var dpr = Math.min(window.devicePixelRatio || 1, 2);
        var same = paneList.every(function (p) {
          return p.info && Math.max(p.box.clientWidth, 200) === p.info.cssW && dpr === p.info.dpr;
        });
        if (!same) redraw();  // a height-only change (a phone's address bar) needs no new picture
      }, 200);
    }
    window.addEventListener("resize", onResize);
    state.cleanups.push(function () {
      window.removeEventListener("resize", onResize);
      clearTimeout(resizeTimer);
      clearTimeout(settleTimer);
      generation += 1;  // abandon any slice that is still queued
    });
    return fig;
  }

  // ------------------------------------------------------------------- init
  try {
    var saved = parseFloat(localStorage.getItem("stl-smoothing-layer"));
    if (isFinite(saved) && saved >= 0.005 && saved <= 5) els.layer.value = String(saved);
  } catch (_) { /* ignore */ }

  updateRunButton();

  // expose a tiny hook for automated tests
  window.__stlSmoothing = { state: state };
  startEngine();
})();
