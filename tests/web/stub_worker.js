/* Stands in for docs/worker.js in the browser tests.
   It speaks the same protocol but, instead of running Python, replays a result that the
   real Python code produced earlier (served from __stub/ by the test). */
"use strict";
const BASE = new URL("__stub/", self.location.href);

async function fetchJson(name) {
  return (await fetch(new URL(name, BASE))).json();
}

self.onmessage = async (event) => {
  const msg = event.data || {};
  if (msg.type === "init") {
    const mode = await fetchJson("mode.json");
    self.postMessage({ type: "status", text: "stub engine starting" });
    if (mode.fatal) {
      self.postMessage({ type: "fatal", message: mode.fatal });
      return;
    }
    setTimeout(() => self.postMessage({ type: "ready", python: "3.12.0", pyodide: "stub" }), 30);
  } else if (msg.type === "run") {
    self.__lastRun = { size: msg.buffer.byteLength, options: msg.options };
    self.postMessage({ type: "progress", id: msg.id, text: "Stub step" });
    const mode = await fetchJson("mode.json");
    if (mode.error) {
      self.postMessage({ type: "error", id: msg.id, message: mode.error });
      return;
    }
    const meta = await fetchJson("meta.json");
    meta.__echo = self.__lastRun;
    let output = null;
    if (meta.output_bytes > 0) output = await (await fetch(new URL("output.stl", BASE))).arrayBuffer();
    const files = {};
    if (meta.preview && meta.preview.files) {
      for (const name of Object.keys(meta.preview.files)) {
        files[name] = await (await fetch(new URL("files/" + name + ".bin", BASE))).arrayBuffer();
      }
    }
    const transfer = [output, ...Object.values(files)].filter(Boolean);
    self.postMessage({ type: "result", id: msg.id, meta, output, files }, transfer);
  }
};
