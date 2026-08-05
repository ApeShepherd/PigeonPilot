// Headless smoke test for the playground frontend.
//
// Runs render() against a real flight plan in jsdom with a stubbed 2D context,
// then drives the animation phases. Catches the class of bug that leaves the
// widget stuck on "Running the reservoir …": an exception in the change:plan
// handler aborts before the animation ever starts, and the browser console is
// the only place it shows up.
//
//   npm install --no-save jsdom
//   node tests/frontend_smoke.mjs [path/to/plan.json]

import fs from "node:fs";
import { JSDOM } from "jsdom";

const fixture = process.argv[2] ?? "/tmp/pp_plan.json";
const { plan, band } = JSON.parse(fs.readFileSync(fixture, "utf-8"));

const dom = new JSDOM("<!doctype html><html><body></body></html>", { pretendToBeVisual: true });
const { window } = dom;

// jsdom ships no canvas backend; every draw call becomes a no-op recorder.
const ctxStub = new Proxy(
  {
    canvas: null,
    measureText: () => ({ width: 10 }),
    createLinearGradient: () => ({ addColorStop() {} }),
    getImageData: () => ({ data: [] }),
    setTransform() {},
  },
  {
    get(target, prop) {
      if (prop in target) return target[prop];
      return typeof prop === "string" ? () => {} : undefined;
    },
    set() {
      return true;
    },
  },
);
window.HTMLCanvasElement.prototype.getContext = () => ctxStub;

const errors = [];
window.addEventListener("error", (e) => errors.push(e.error ?? e.message));

const frames = [];
window.requestAnimationFrame = (cb) => {
  frames.push(cb);
  return frames.length;
};

globalThis.window = window;
globalThis.document = window.document;
globalThis.requestAnimationFrame = window.requestAnimationFrame;
globalThis.devicePixelRatio = 1;
// A plain clock: jsdom's performance delegates to Node's, so leaving the two
// wired together makes them call each other forever.
let fakeClock = 0;
Object.defineProperty(window, "performance", {
  configurable: true,
  value: { now: () => (fakeClock += 16) },
});

class FakeModel {
  constructor(values) {
    this.values = values;
    this.handlers = {};
    this.saved = [];
  }
  get(key) {
    return this.values[key];
  }
  set(key, value) {
    this.values[key] = value;
    (this.handlers[`change:${key}`] ?? []).forEach((fn) => fn());
  }
  on(event, fn) {
    (this.handlers[event] ??= []).push(fn);
  }
  save_changes() {
    this.saved.push(structuredClone(this.values.request));
  }
}

const { default: widget } = await import("../pigeonpilot/static/playground.js");

const root = window.document.createElement("div");
window.document.body.appendChild(root);
const model = new FakeModel({ field_half: 7.1, band, plan: {}, status: "", busy: false });

const step = (label, fn) => {
  try {
    fn();
    console.log(`  ok   ${label}`);
  } catch (err) {
    console.log(`  FAIL ${label}\n       ${err && err.stack ? err.stack.split("\n").slice(0, 4).join("\n       ") : err}`);
    errors.push(err);
  }
};

console.log("frontend smoke test");
step("render() mounts", () => widget.render({ model, el: root }));

const statusText = () => root.querySelector(".pp-status")?.textContent ?? "";
step("initial status is the draw prompt", () => {
  if (!statusText().includes("Draw a route")) throw new Error(`unexpected status: ${statusText()}`);
});

const buttons = [...root.querySelectorAll("button")];
const click = (label) => {
  const btn = buttons.find((b) => b.textContent.startsWith(label));
  if (!btn) throw new Error(`no button "${label}"`);
  btn.dispatchEvent(new window.MouseEvent("click", { bubbles: true }));
};

step("Fly without a stroke warns instead of hanging", () => {
  click("Fly");
  if (!statusText().includes("Draw first")) throw new Error(`unexpected status: ${statusText()}`);
});

step("plan arrival starts the animation", () => {
  model.set("plan", plan);
  const text = statusText();
  if (text.includes("Running the reservoir")) throw new Error("stuck on 'Running the reservoir' — change:plan aborted");
  if (!text.includes("Displacement")) throw new Error(`animation did not start, status: ${text}`);
});

step("animation frames run without throwing", () => {
  let now = 0;
  for (let i = 0; i < 400 && frames.length; i++) {
    const cb = frames.shift();
    now += 16;
    cb(now);
  }
});

step("readout cards rendered", () => {
  const cards = root.querySelectorAll(".pp-card");
  if (cards.length !== plan.models.length + 1) throw new Error(`expected ${plan.models.length + 1} cards, got ${cards.length}`);
  if (!root.querySelector(".pp-warn")) throw new Error("head-to-head note missing");
  if (root.querySelectorAll(".pp-strip").length !== plan.models.length) throw new Error("distribution strips missing");
});

step("no ensemble controls are left in the toolbar", () => {
  if (root.querySelector('input[type="checkbox"]')) throw new Error("ensemble toggle still present");
});

step("error plan shows the message", () => {
  model.set("plan", { nonce: 99, error: "Route too short" });
  if (!statusText().includes("Route too short")) throw new Error(`unexpected status: ${statusText()}`);
});

step("Clear resets", () => click("Clear"));

// A kernel that still runs an older pigeonpilot sends a plan without the
// reference block. The widget must animate anyway rather than abort in
// change:plan and sit on "Running the reservoir …".
step("plan without reference block still animates", () => {
  const legacy = structuredClone(plan);
  legacy.models.forEach((m) => delete m.reference);
  delete legacy.reference.head_to_head;
  delete legacy.reference.n;
  model.set("plan", legacy);
  if (!statusText().includes("Displacement")) throw new Error(`did not start: ${statusText()}`);
});

step("plan with no reference key at all still animates", () => {
  const ancient = structuredClone(plan);
  ancient.models.forEach((m) => delete m.reference);
  delete ancient.reference;
  model.set("plan", ancient);
  if (!statusText().includes("Displacement")) throw new Error(`did not start: ${statusText()}`);
});

// The ensemble block is gone from the default plan; a plan that still carries
// one (ensemble_size > 0) must not change what the frontend draws.
step("plan carrying an ensemble block is ignored gracefully", () => {
  const withEnsemble = structuredClone(plan);
  withEnsemble.models.forEach((m) => {
    m.ensemble = { size: 4, bins: [0, 1, 2, 3], histogram: [], bin: 1, heading_deg: 10, error_deg: 5, circular_sd_deg: 20, resultant: 0.9 };
  });
  model.set("plan", withEnsemble);
  if (!statusText().includes("Displacement")) throw new Error(`did not start: ${statusText()}`);
});

if (errors.length) {
  console.log(`\n${errors.length} failure(s)`);
  process.exit(1);
}
console.log("\nall good");
