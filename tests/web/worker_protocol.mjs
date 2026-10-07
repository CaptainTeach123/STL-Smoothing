// Runs docs/worker.js in a sandbox with a fake Pyodide and checks the message protocol.
// (The real Pyodide is exercised by the GitHub Actions workflow; this catches logic bugs early.)
import fs from "node:fs";
import vm from "node:vm";
import assert from "node:assert/strict";

const source = fs.readFileSync(new URL("../../docs/worker.js", import.meta.url), "utf8");

function makeWorld({ failVersions = 0, failAll = false, processThrows = false, processResult = null, selftest = { ok: true }, manifestStatus = 200 } = {}) {
  const posted = [];
  const fs_ = new Map();
  const writeOpts = [];
  const imported = [];
  const FS = {
    mkdirTree() {},
    writeFile(p, d, opts) { fs_.set(p, d); writeOpts.push(opts); },
    readFile(p) {
      if (!fs_.has(p)) throw new Error("ENOENT " + p);
      const v = fs_.get(p);
      return v instanceof Uint8Array ? v : new TextEncoder().encode(String(v));
    },
    unlink(p) { if (!fs_.delete(p)) throw new Error("ENOENT " + p); },
  };
  const calls = { loadPackage: [], process: [] };
  const pyodide = {
    FS,
    async loadPackage(names, opts) { calls.loadPackage.push(names); opts.messageCallback("loading numpy"); },
    runPython(code) { return code.includes("version") ? "3.12.1" : undefined; },
    pyimport(name) {
      assert.equal(name, "stl_smoothing.web");
      return {
        selftest() { return JSON.stringify(selftest); },
        process(pin, pout, optsJson, cb, pv) {
          calls.process.push({ pin, pout, opts: JSON.parse(optsJson), pv });
          if (processThrows) throw new Error("python exploded");
          cb("Reading the STL file");
          if (processResult) return JSON.stringify(processResult);
          fs_.set(pout, new Uint8Array([9, 8, 7]));
          fs_.set(pv + "/a.bin", new Uint8Array(16));
          return JSON.stringify({ ok: true, output_bytes: 3, preview: { files: { a: { file: pv + "/a.bin", dtype: "float32", length: 4 } } } });
        },
      };
    },
  };
  const fetched = [];
  const self = {
    postMessage(m, t) { posted.push({ m, t: t || [] }); },
    loadPyodide: async () => pyodide,
  };
  const ctx = vm.createContext({
    self,
    importScripts(url) {
      imported.push(url);
      if (failAll || imported.length <= failVersions) throw new Error("404 " + url);
    },
    fetch: async (url) => {
      fetched.push(String(url));
      if (String(url).startsWith("py/manifest.json")) {
        return { ok: manifestStatus === 200, status: manifestStatus, json: async () => ({ files: ["__init__.py", "web.py"], hash: "abc" }) };
      }
      return { ok: true, text: async () => "# python source for " + url };
    },
    console, TextEncoder, JSON, Object, Uint8Array, Error, String, Map,
  });
  vm.runInContext(source, ctx);
  return { self, posted, fs_, imported, fetched, calls, writeOpts, send: (data) => self.onmessage({ data }) };
}

const types = (w) => w.posted.map((p) => p.m.type);

// 1. normal start-up
{
  const w = makeWorld();
  await w.send({ type: "init" });
  assert.equal(JSON.stringify([...new Set(types(w))]), '["status","ready"]');
  const ready = w.posted.find((p) => p.m.type === "ready").m;
  assert.equal(ready.python, "3.12.1");
  assert.match(ready.pyodide, /^0\.\d+\.\d+$/);
  assert.equal(JSON.stringify(w.calls.loadPackage[0]), '["numpy","scipy"]');  // JSON: the sandbox has its own Array
  assert.match(w.imported[0], /^https:\/\/cdn\.jsdelivr\.net\/pyodide\/v[\d.]+\/full\/pyodide\.js$/);
  assert.ok(w.fetched.includes("py/stl_smoothing/web.py?v=abc"), "every manifest file is fetched with the cache-busting hash");
  assert.ok(w.fs_.has("/home/pyodide/stl_smoothing/web.py"), "sources are written into the engine's file system");
}

// 2. a version that cannot be downloaded falls back to the next one
{
  const w = makeWorld({ failVersions: 1 });
  await w.send({ type: "init" });
  assert.equal(w.imported.length, 2);
  assert.ok(types(w).includes("ready"));
  assert.notEqual(w.imported[0], w.imported[1]);
}

// 3. no version can be downloaded: a clear fatal message, never a hang
{
  const w = makeWorld({ failAll: true });
  await w.send({ type: "init" });
  const fatal = w.posted.find((p) => p.m.type === "fatal");
  assert.ok(fatal, "a fatal message is posted");
  assert.match(fatal.m.message, /jsdelivr/i);
  assert.ok(!types(w).includes("ready"));
}

// 4. a run: progress, a result with transferred buffers, and the temporary files cleaned up
{
  const w = makeWorld();
  await w.send({ type: "init" });
  w.posted.length = 0;
  await w.send({ type: "run", id: 7, buffer: new Uint8Array([1, 2, 3, 4]).buffer, options: { layer_height: 0.2 } });
  assert.equal(JSON.stringify(types(w)), '["progress","result"]');
  const result = w.posted[1];
  assert.equal(result.m.id, 7);
  assert.equal(result.m.output.byteLength, 3);
  assert.equal(result.m.files.a.byteLength, 16);
  assert.equal(result.t.length, 2, "the output and the preview array are transferred, not copied");
  assert.equal(JSON.stringify(w.calls.process[0].opts), '{"layer_height":0.2}');
  assert.equal(w.calls.process[0].pin, "/tmp/model_in.stl");
  assert.equal([...w.fs_.keys()].filter((k) => k.startsWith("/tmp/")).length, 0, "temporary files are removed");
}

// 5. nothing changed: no output buffer is read
{
  const w = makeWorld({ processResult: { ok: true, output_bytes: 0, preview: null } });
  await w.send({ type: "init" });
  w.posted.length = 0;
  await w.send({ type: "run", id: 1, buffer: new ArrayBuffer(4), options: {} });
  const r = w.posted.find((p) => p.m.type === "result");
  assert.equal(r.m.output, null);
  assert.equal(JSON.stringify(r.m.files), "{}");
}

// 6. a Python failure becomes an error message for the page, and the worker stays usable
{
  const w = makeWorld({ processThrows: true });
  await w.send({ type: "init" });
  w.posted.length = 0;
  await w.send({ type: "run", id: 3, buffer: new ArrayBuffer(4), options: {} });
  const e = w.posted.find((p) => p.m.type === "error");
  assert.equal(e.m.id, 3);
  assert.match(e.m.message, /python exploded/);
  assert.equal([...w.fs_.keys()].filter((k) => k.startsWith("/tmp/")).length, 0);
}

// 6b. the uploaded model is handed to the engine's file system without a copy
{
  const w = makeWorld();
  await w.send({ type: "init" });
  w.writeOpts.length = 0;
  await w.send({ type: "run", id: 4, buffer: new ArrayBuffer(8), options: {} });
  assert.equal(w.writeOpts[0] && w.writeOpts[0].canOwn, true);
}

// 7. a run before the engine is ready is refused politely
{
  const w = makeWorld();
  await w.send({ type: "run", id: 1, buffer: new ArrayBuffer(4), options: {} });
  assert.equal(w.posted[0].m.type, "error");
}

// 8. the engine that fails its self-check is never announced as ready, and refuses runs
{
  const w = makeWorld({ selftest: { ok: false, error: "boom in numpy", detail: "Traceback..." } });
  await w.send({ type: "init" });
  assert.ok(!types(w).includes("ready"));
  const fatal = w.posted.find((p) => p.m.type === "fatal");
  assert.match(fatal.m.message, /self-check: boom in numpy/);
  w.posted.length = 0;
  await w.send({ type: "run", id: 2, buffer: new ArrayBuffer(4), options: {} });
  assert.equal(w.posted[0].m.type, "error");
}

// 9. a missing manifest (wrong Pages folder, partial deploy) gives a message that names the file and the status
{
  const w = makeWorld({ manifestStatus: 404 });
  await w.send({ type: "init" });
  const fatal = w.posted.find((p) => p.m.type === "fatal");
  assert.match(fatal.m.message, /py\/manifest\.json \(HTTP 404\).*docs/);
}

console.log("worker protocol: all checks passed");
