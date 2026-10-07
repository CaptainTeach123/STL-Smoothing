/* Runs the Python engine (Pyodide) off the main thread so the page never freezes.

   Protocol (all messages are plain objects):
     page -> worker   {type: "init"}
                      {type: "run", id, buffer: ArrayBuffer, options: {...}}
     worker -> page   {type: "status", text}                      engine loading progress
                      {type: "ready", python: "3.12.x", pyodide: "0.27.7"}
                      {type: "fatal", message}                    the engine could not start
                      {type: "progress", id, text}                while a model is processed
                      {type: "result", id, meta, output, files}   output/files are ArrayBuffers
                      {type: "error", id, message}
*/
"use strict";

// Tried in order: the first that loads wins. Both ship numpy 2 and scipy; numpy 1 engines are left out because the code has not been run on them.
const PYODIDE_VERSIONS = ["0.27.7", "0.27.5"];
const CDN = "https://cdn.jsdelivr.net/pyodide/";

let pyodide = null;
let processFn = null;
let engineVersion = "";

function post(message, transfer) {
  self.postMessage(message, transfer || []);
}

async function startPyodide() {
  const failures = [];
  let startFailure = null;
  for (const v of PYODIDE_VERSIONS) {
    post({ type: "status", text: `Downloading the Python engine (Pyodide ${v})…` });
    try {
      importScripts(`${CDN}v${v}/full/pyodide.js`);
    } catch (err) {
      failures.push(`${v}: ${err && err.message ? err.message : err}`);
      continue;
    }
    try {
      pyodide = await self.loadPyodide({ indexURL: `${CDN}v${v}/full/` });
      engineVersion = v;
      return;
    } catch (err) {
      // the download worked, so this is the browser (too old, no WebAssembly) or its memory
      startFailure = err && err.message ? err.message : String(err);
    }
  }
  if (startFailure !== null) {
    throw new Error(
      "The Python engine downloaded but could not start in this browser (" + startFailure + "). " +
        "It needs Chrome 112, Firefox 112 or Safari 16.4 or newer and enough free memory; " +
        "try closing other tabs or another browser."
    );
  }
  throw new Error(
    "Could not download the Python engine from cdn.jsdelivr.net (" + failures.join("; ") + "). " +
      "Check your connection or any content blocker, then reload the page."
  );
}

async function loadEngine() {
  await startPyodide();
  post({ type: "status", text: "Loading numpy and scipy (the engine is about 25 MB in all, cached by your browser after the first visit)…" });
  await pyodide.loadPackage(["numpy", "scipy"], {
    messageCallback: (m) => post({ type: "status", text: m }),
    errorCallback: (m) => post({ type: "status", text: m }),
  });
  post({ type: "status", text: "Loading the smoothing code…" });
  const manifestResponse = await fetch("py/manifest.json", { cache: "no-cache" });
  if (!manifestResponse.ok) {
    throw new Error(`Could not load py/manifest.json (HTTP ${manifestResponse.status}). Is the site being served from the docs/ folder?`);
  }
  const manifest = await manifestResponse.json();
  pyodide.FS.mkdirTree("/home/pyodide/stl_smoothing");
  for (const file of manifest.files) {
    const response = await fetch(`py/stl_smoothing/${file}?v=${manifest.hash}`);
    if (!response.ok) throw new Error(`Could not load ${file} (HTTP ${response.status})`);
    pyodide.FS.writeFile(`/home/pyodide/stl_smoothing/${file}`, await response.text());
  }
  pyodide.runPython('import sys\nif "/home/pyodide" not in sys.path:\n    sys.path.insert(0, "/home/pyodide")');
  const web = pyodide.pyimport("stl_smoothing.web");
  // Smooth a small built-in model before announcing "ready", so a numpy/scipy problem in this
  // browser engine is reported now and not on the first model a visitor tries.
  post({ type: "status", text: "Checking the engine on a small built-in model…" });
  const check = JSON.parse(web.selftest());
  if (!check.ok) {
    if (check.detail) console.error(check.detail);
    throw new Error("The Python engine started but failed its self-check: " + check.error);
  }
  processFn = web.process;
  post({ type: "ready", python: pyodide.runPython("import sys; sys.version.split()[0]"), pyodide: engineVersion });
}

function bufferOf(u8) {
  return u8.byteOffset === 0 && u8.byteLength === u8.buffer.byteLength
    ? u8.buffer
    : u8.buffer.slice(u8.byteOffset, u8.byteOffset + u8.byteLength);
}

function removeQuietly(path) {
  try {
    pyodide.FS.unlink(path);
  } catch (_) {
    /* already gone */
  }
}

function run(id, buffer, options) {
  const FS = pyodide.FS;
  const IN = "/tmp/model_in.stl";
  const OUT = "/tmp/model_out.stl";
  const PV = "/tmp/preview";
  const written = [];
  try {
    FS.writeFile(IN, new Uint8Array(buffer), { canOwn: true });  // hand the bytes over instead of copying them
    removeQuietly(OUT);
    const metaJson = processFn(IN, OUT, JSON.stringify(options || {}), (text) => post({ type: "progress", id, text }), PV);
    const meta = JSON.parse(metaJson);
    const transfer = [];
    let output = null;
    if (meta.ok && meta.output_bytes > 0) {
      output = bufferOf(FS.readFile(OUT));
      transfer.push(output);
    }
    const files = {};
    if (meta.ok && meta.preview && meta.preview.files) {
      for (const [name, info] of Object.entries(meta.preview.files)) {
        files[name] = bufferOf(FS.readFile(info.file));
        transfer.push(files[name]);
        written.push(info.file);
      }
    }
    post({ type: "result", id, meta, output, files }, transfer);
  } catch (err) {
    post({ type: "error", id, message: err && err.message ? err.message : String(err) });
  } finally {
    removeQuietly(IN);
    removeQuietly(OUT);
    written.forEach(removeQuietly);
  }
}

self.onmessage = async (event) => {
  const msg = event.data || {};
  if (msg.type === "init") {
    try {
      await loadEngine();
    } catch (err) {
      post({ type: "fatal", message: err && err.message ? err.message : String(err) });
    }
  } else if (msg.type === "run") {
    if (!processFn) {
      post({ type: "error", id: msg.id, message: "The Python engine is not ready yet." });
      return;
    }
    run(msg.id, msg.buffer, msg.options);
  }
};
