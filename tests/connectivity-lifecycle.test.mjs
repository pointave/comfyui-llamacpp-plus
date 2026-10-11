import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const source = readFileSync(
  new URL("../web/js/LlamaCPPNode.js", import.meta.url),
  "utf8",
);

let extension;
const app = {
  extensionManager: { toast: { add() {} } },
  registerExtension(value) {
    extension = value;
  },
};

globalThis.__llamaCppTestApp = app;
const executableSource = source.replace(
  'import { app } from "/scripts/app.js";',
  "const app = globalThis.__llamaCppTestApp;",
);
await import(
  `data:text/javascript;base64,${Buffer.from(executableSource).toString("base64")}`
);
delete globalThis.__llamaCppTestApp;

class ConnectivityNode {
  constructor(url = "http://127.0.0.1:8081") {
    this.widgets = [
      { name: "url", value: url },
      { name: "model", value: "", options: { values: [] } },
    ];
  }

  addWidget(type, name, value, callback, options) {
    const widget = { type, name, value, callback, options };
    this.widgets.push(widget);
    return widget;
  }

  setDirtyCanvas() {}
}

await extension.beforeRegisterNodeDef(
  ConnectivityNode,
  { name: "LlamaCPPConnectivity" },
  app,
);

test("model discovery waits for a loaded workflow's restored URL", async () => {
  const originalFetch = globalThis.fetch;
  const originalSetTimeout = globalThis.setTimeout;
  const requests = [];
  const timers = [];

  globalThis.fetch = async (url, options) => {
    requests.push({ url, body: JSON.parse(options.body) });
    return { ok: true, json: async () => ["saved-model"] };
  };
  globalThis.setTimeout = (callback, delay) => {
    timers.push({ callback, delay });
    return timers.length;
  };

  try {
    const node = new ConnectivityNode();
    await node.onNodeCreated();

    assert.equal(requests.length, 0);
    assert.equal(timers.length, 1);
    assert.equal(timers[0].delay, 0);

    node.widgets.find((widget) => widget.name === "url").value =
      "http://192.168.0.223:8080";
    await timers[0].callback();

    assert.deepEqual(requests, [
      {
        url: "/llamacpp/get_models",
        body: { url: "http://192.168.0.223:8080" },
      },
    ]);
    assert.equal(
      node.widgets.find((widget) => widget.name === "model").value,
      "saved-model",
    );
  } finally {
    globalThis.fetch = originalFetch;
    globalThis.setTimeout = originalSetTimeout;
  }
});

test("a newly added node still performs its initial model discovery", async () => {
  const originalFetch = globalThis.fetch;
  const originalSetTimeout = globalThis.setTimeout;
  const requests = [];
  const timers = [];

  globalThis.fetch = async (url, options) => {
    requests.push({ url, body: JSON.parse(options.body) });
    return { ok: true, json: async () => ["default-model"] };
  };
  globalThis.setTimeout = (callback, delay) => {
    timers.push({ callback, delay });
    return timers.length;
  };

  try {
    const node = new ConnectivityNode();
    await node.onNodeCreated();
    await timers[0].callback();

    assert.deepEqual(requests, [
      {
        url: "/llamacpp/get_models",
        body: { url: "http://127.0.0.1:8081" },
      },
    ]);
  } finally {
    globalThis.fetch = originalFetch;
    globalThis.setTimeout = originalSetTimeout;
  }
});

test("runtime requirements declare the OpenAI client imported by nodes.py", () => {
  const requirements = readFileSync(
    new URL("../requirements.txt", import.meta.url),
    "utf8",
  )
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);
  const nodesSource = readFileSync(
    new URL("../nodes.py", import.meta.url),
    "utf8",
  );

  assert.match(nodesSource, /from openai import OpenAI/);
  assert.ok(requirements.includes("openai"));
});
