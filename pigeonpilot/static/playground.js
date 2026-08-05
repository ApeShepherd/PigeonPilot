// PigeonPilot live playground — canvas frontend.
//
// Python computes a complete flight plan once and pushes it over the widget
// comm; everything below only *replays* it. No inference, no geometry decisions
// and no numbers are invented here — every value drawn comes from the plan.

const TAU = Math.PI * 2;
const COLOR = {
  home: "#2e7d32",
  release: "#c62828",
  path: "#111827",
  raw: "#9ca3af",
  truth: "#1d4ed8",
  A: "#d81b8c",
  B: "#ea7317",
  grid: "#e5e7eb",
  axis: "#cbd5e1",
  band: "#94a3b8",
  ink: "#0f172a",
  muted: "#64748b",
};
const PHASE = { IDLE: "idle", OUTBOUND: "outbound", RELEASE: "release", HOME: "home", DONE: "done" };
const RELEASE_HOLD_S = 0.7;
const HOME_FLIGHT_S = 2.6;
const OUTBOUND_FLIGHT_S = 4.2;
// A flight plan takes ~1.5 s to compute; well past that, something is wrong.
const WATCHDOG_MS = 20000;

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text != null) node.textContent = text;
  return node;
}

/** Size a canvas for the device pixel ratio and return a ready 2D context. */
function fitCanvas(canvas, cssWidth, cssHeight) {
  const dpr = window.devicePixelRatio || 1;
  canvas.style.width = `${cssWidth}px`;
  canvas.style.height = `${cssHeight}px`;
  canvas.width = Math.round(cssWidth * dpr);
  canvas.height = Math.round(cssHeight * dpr);
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, cssWidth, cssHeight);
  return ctx;
}

/** Compass degrees (0 = North, clockwise) to a screen-space unit vector. */
function headingUnit(deg) {
  const rad = (deg * Math.PI) / 180;
  return { x: Math.sin(rad), y: Math.cos(rad) };
}

function circularDelta(a, b) {
  let d = ((a - b) % 360 + 540) % 360 - 180;
  return d;
}

function render({ model, el: root }) {
  root.classList.add("pp-root");

  // ---------------------------------------------------------------- layout
  const toolbar = el("div", "pp-toolbar");
  const flyBtn = el("button", "pp-btn pp-btn-primary", "Fly 🐦");
  const replayBtn = el("button", "pp-btn", "Replay");
  const clearBtn = el("button", "pp-btn", "Clear");
  const speedWrap = el("label", "pp-field");
  speedWrap.appendChild(el("span", null, "Tempo"));
  const speed = el("input", "pp-range");
  Object.assign(speed, { type: "range", min: "0.35", max: "2.5", step: "0.05", value: "1" });
  speedWrap.appendChild(speed);
  const ensembleWrap = el("label", "pp-field pp-check");
  const ensembleBox = el("input");
  ensembleBox.type = "checkbox";
  ensembleWrap.appendChild(ensembleBox);
  ensembleWrap.appendChild(el("span", null, "Seed-Ensemble"));
  toolbar.append(flyBtn, replayBtn, clearBtn, speedWrap, ensembleWrap);

  const status = el("div", "pp-status", "Zeichne einen Pfad vom grünen Stern aus — dann Fly.");

  const stage = el("div", "pp-stage");
  const fieldCanvas = el("canvas", "pp-canvas pp-field-canvas");
  const ringCanvas = el("canvas", "pp-canvas");
  stage.append(fieldCanvas, ringCanvas);

  const rasterCanvas = el("canvas", "pp-canvas pp-raster");
  const readout = el("div", "pp-readout");
  const note = el("div", "pp-note");

  root.append(toolbar, status, stage, rasterCanvas, readout, note);

  // ----------------------------------------------------------------- state
  const state = {
    stroke: [],
    drawing: false,
    plan: null,
    phase: PHASE.IDLE,
    clock: 0,
    lastFrame: 0,
    running: false,
    half: model.get("field_half") || 7,
    watchdog: null,
    reportedError: false,
  };

  const FIELD_PX = 460;
  const RING_PX = 230;
  const RASTER_H = 236;
  let requestNonce = 0;

  function fieldGeometry() {
    const half = state.half;
    const scale = (FIELD_PX / 2 - 18) / half;
    return {
      toScreen: (x, y) => ({ x: FIELD_PX / 2 + x * scale, y: FIELD_PX / 2 - y * scale }),
      toWorld: (px, py) => ({ x: (px - FIELD_PX / 2) / scale, y: (FIELD_PX / 2 - py) / scale }),
      scale,
    };
  }

  // --------------------------------------------------------------- drawing
  function drawField() {
    const ctx = fitCanvas(fieldCanvas, FIELD_PX, FIELD_PX);
    const g = fieldGeometry();
    const band = model.get("band") || {};

    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, FIELD_PX, FIELD_PX);

    // grid + axes
    ctx.strokeStyle = COLOR.grid;
    ctx.lineWidth = 1;
    const stepWorld = state.half <= 4 ? 1 : 2;
    for (let v = -Math.floor(state.half); v <= state.half; v += stepWorld) {
      const p = g.toScreen(v, v);
      ctx.beginPath(); ctx.moveTo(p.x, 0); ctx.lineTo(p.x, FIELD_PX); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(0, p.y); ctx.lineTo(FIELD_PX, p.y); ctx.stroke();
    }
    ctx.strokeStyle = COLOR.axis;
    const o = g.toScreen(0, 0);
    ctx.beginPath(); ctx.moveTo(o.x, 0); ctx.lineTo(o.x, FIELD_PX); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(0, o.y); ctx.lineTo(FIELD_PX, o.y); ctx.stroke();

    // training radius — the region the reservoir actually saw
    if (band.band_max_release) {
      ctx.save();
      ctx.setLineDash([5, 5]);
      ctx.strokeStyle = COLOR.band;
      ctx.globalAlpha = 0.55;
      ctx.beginPath();
      ctx.arc(o.x, o.y, band.band_max_release * g.scale, 0, TAU);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.globalAlpha = 0.75;
      ctx.fillStyle = COLOR.band;
      ctx.font = "10px ui-sans-serif, system-ui, sans-serif";
      ctx.textAlign = "center";
      const label = g.toScreen(0, band.band_max_release);
      ctx.fillText("Trainingsradius", label.x, label.y - 5);
      ctx.restore();
    }

    ctx.font = "11px ui-sans-serif, system-ui, sans-serif";
    ctx.fillStyle = COLOR.muted;
    ctx.textAlign = "center";
    ctx.fillText("N", o.x, 13);
    ctx.fillText("S", o.x, FIELD_PX - 4);
    ctx.textAlign = "left";
    ctx.fillText("W", 4, o.y - 4);
    ctx.textAlign = "right";
    ctx.fillText("E", FIELD_PX - 4, o.y - 4);

    // live stroke while drawing
    if (state.stroke.length > 1 && !state.plan) {
      ctx.strokeStyle = COLOR.raw;
      ctx.lineWidth = 2;
      ctx.lineJoin = "round";
      ctx.beginPath();
      state.stroke.forEach((p, i) => {
        const s = g.toScreen(p[0], p[1]);
        i === 0 ? ctx.moveTo(s.x, s.y) : ctx.lineTo(s.x, s.y);
      });
      ctx.stroke();
    }

    const plan = state.plan;
    if (plan) {
      // faint original stroke, so the snapping stays visible and honest
      ctx.strokeStyle = COLOR.raw;
      ctx.lineWidth = 1.5;
      ctx.globalAlpha = 0.45;
      ctx.beginPath();
      plan.raw_points.forEach((p, i) => {
        const s = g.toScreen(p[0] * plan.preprocessing.scale_factor, p[1] * plan.preprocessing.scale_factor);
        i === 0 ? ctx.moveTo(s.x, s.y) : ctx.lineTo(s.x, s.y);
      });
      ctx.stroke();
      ctx.globalAlpha = 1;

      const travelled = outboundProgress();
      drawSnappedPath(ctx, g, plan, travelled);

      if (state.phase === PHASE.HOME || state.phase === PHASE.DONE) {
        drawHomeLeg(ctx, g, plan);
      }
      if (state.phase !== PHASE.IDLE) drawRelease(ctx, g, plan);
      drawBirds(ctx, g, plan);
    }

    // home star last so it stays on top
    drawStar(ctx, o.x, o.y, 9, COLOR.home);
  }

  function drawSnappedPath(ctx, g, plan, upto) {
    ctx.lineWidth = 2.6;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.strokeStyle = COLOR.path;
    ctx.beginPath();
    const pts = plan.snapped_points;
    pts.forEach((p, i) => {
      const s = g.toScreen(p[0], p[1]);
      i === 0 ? ctx.moveTo(s.x, s.y) : ctx.lineTo(s.x, s.y);
    });
    ctx.globalAlpha = state.phase === PHASE.IDLE ? 1 : 0.22;
    ctx.stroke();
    ctx.globalAlpha = 1;

    // travelled portion, drawn solid on top
    if (upto > 0) {
      ctx.strokeStyle = COLOR.path;
      ctx.beginPath();
      const start = g.toScreen(pts[0][0], pts[0][1]);
      ctx.moveTo(start.x, start.y);
      for (const seg of plan.segments) {
        if (upto >= seg.t_end) {
          const p = g.toScreen(seg.x1, seg.y1);
          ctx.lineTo(p.x, p.y);
        } else if (upto > seg.t_start) {
          const f = (upto - seg.t_start) / (seg.t_end - seg.t_start);
          const p = g.toScreen(seg.x0 + (seg.x1 - seg.x0) * f, seg.y0 + (seg.y1 - seg.y0) * f);
          ctx.lineTo(p.x, p.y);
          break;
        } else break;
      }
      ctx.stroke();
    }
  }

  function drawRelease(ctx, g, plan) {
    const p = g.toScreen(plan.release_xy[0], plan.release_xy[1]);
    const pulse = state.phase === PHASE.RELEASE ? 1 + 0.5 * Math.sin(state.clock * 9) : 1;
    ctx.fillStyle = COLOR.release;
    ctx.beginPath();
    ctx.arc(p.x, p.y, 5.5 * pulse, 0, TAU);
    ctx.fill();
  }

  function drawHomeLeg(ctx, g, plan) {
    const rel = g.toScreen(plan.release_xy[0], plan.release_xy[1]);
    const home = g.toScreen(0, 0);
    ctx.save();
    ctx.setLineDash([6, 5]);
    ctx.strokeStyle = COLOR.truth;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(rel.x, rel.y);
    ctx.lineTo(home.x, home.y);
    ctx.stroke();
    ctx.restore();

    for (const m of plan.models) {
      const bin = activeBin(m);
      const dir = headingUnit((bin * 360) / plan.heading_bins);
      const end = g.toScreen(
        plan.release_xy[0] + dir.x * plan.home_distance,
        plan.release_xy[1] + dir.y * plan.home_distance,
      );
      ctx.strokeStyle = COLOR[m.name];
      ctx.lineWidth = 2;
      ctx.globalAlpha = 0.35;
      ctx.beginPath();
      ctx.moveTo(rel.x, rel.y);
      ctx.lineTo(end.x, end.y);
      ctx.stroke();
      ctx.globalAlpha = 1;
    }
  }

  /** Timesteps completed on the outbound leg at the current clock. */
  function outboundProgress() {
    const plan = state.plan;
    if (!plan) return 0;
    if (state.phase === PHASE.IDLE) return 0;
    if (state.phase !== PHASE.OUTBOUND) return plan.path_steps;
    return Math.min(plan.path_steps, (state.clock / OUTBOUND_FLIGHT_S) * plan.path_steps);
  }

  /** Timesteps the raster may reveal — includes the trailing silence once landed. */
  function revealedSteps() {
    const plan = state.plan;
    if (!plan || state.phase === PHASE.IDLE) return 0;
    return state.phase === PHASE.OUTBOUND ? outboundProgress() : plan.n_steps;
  }

  function positionAt(plan, steps) {
    for (const seg of plan.segments) {
      if (steps <= seg.t_end) {
        const f = Math.max(0, (steps - seg.t_start) / Math.max(1, seg.t_end - seg.t_start));
        return { x: seg.x0 + (seg.x1 - seg.x0) * f, y: seg.y0 + (seg.y1 - seg.y0) * f, seg };
      }
    }
    const last = plan.segments[plan.segments.length - 1];
    return { x: last.x1, y: last.y1, seg: last };
  }

  function activeBin(m) {
    return ensembleBox.checked ? m.ensemble.bin : m.pred_bin;
  }

  function drawBirds(ctx, g, plan) {
    ctx.font = "22px system-ui, 'Apple Color Emoji', sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";

    if (state.phase === PHASE.OUTBOUND || state.phase === PHASE.RELEASE) {
      const pos = positionAt(plan, outboundProgress());
      const s = g.toScreen(pos.x, pos.y);
      const bob = state.phase === PHASE.OUTBOUND ? Math.sin(state.clock * 12) * 1.6 : 0;
      ctx.fillText("🐦", s.x, s.y + bob);
      return;
    }
    if (state.phase === PHASE.HOME || state.phase === PHASE.DONE) {
      const f = state.phase === PHASE.DONE ? 1 : Math.min(1, state.clock / HOME_FLIGHT_S);
      for (const m of plan.models) {
        const bin = activeBin(m);
        const dir = headingUnit((bin * 360) / plan.heading_bins);
        const x = plan.release_xy[0] + dir.x * plan.home_distance * f;
        const y = plan.release_xy[1] + dir.y * plan.home_distance * f;
        const s = g.toScreen(x, y);
        ctx.beginPath();
        ctx.fillStyle = COLOR[m.name];
        ctx.globalAlpha = 0.22;
        ctx.arc(s.x, s.y, 15, 0, TAU);
        ctx.fill();
        ctx.globalAlpha = 1;
        ctx.fillStyle = COLOR.ink;
        ctx.fillText(m.name === "A" ? "🐦" : "🕊️", s.x, s.y);
        ctx.fillStyle = COLOR[m.name];
        ctx.font = "bold 11px ui-sans-serif, system-ui, sans-serif";
        ctx.fillText(m.name, s.x, s.y - 17);
        ctx.font = "22px system-ui, 'Apple Color Emoji', sans-serif";
      }
    }
  }

  function drawStar(ctx, cx, cy, r, color) {
    ctx.fillStyle = color;
    ctx.beginPath();
    for (let i = 0; i < 10; i++) {
      const rad = i % 2 === 0 ? r : r * 0.45;
      const a = (i * Math.PI) / 5 - Math.PI / 2;
      const x = cx + Math.cos(a) * rad;
      const y = cy + Math.sin(a) * rad;
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    }
    ctx.closePath();
    ctx.fill();
  }

  // ------------------------------------------------------------------ rings
  function drawRings() {
    const plan = state.plan;
    const ctx = fitCanvas(ringCanvas, RING_PX, FIELD_PX);
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, RING_PX, FIELD_PX);
    if (!plan) {
      ctx.fillStyle = COLOR.muted;
      ctx.font = "12px ui-sans-serif, system-ui, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("Ringe erscheinen", RING_PX / 2, FIELD_PX / 2 - 8);
      ctx.fillText("nach dem Flug", RING_PX / 2, FIELD_PX / 2 + 10);
      return;
    }
    const revealed = state.phase === PHASE.DONE || state.phase === PHASE.HOME;
    plan.models.forEach((m, i) => {
      drawRing(ctx, m, plan, RING_PX / 2, 118 + i * 224, 82, revealed);
    });
  }

  function drawRing(ctx, m, plan, cx, cy, r, revealed) {
    const bins = plan.heading_bins;
    ctx.save();
    ctx.strokeStyle = COLOR.grid;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.arc(cx, cy, r, 0, TAU);
    ctx.stroke();

    const useEnsemble = ensembleBox.checked;
    const values = useEnsemble ? m.ensemble.histogram : m.scores;
    const lo = Math.min(...values);
    const hi = Math.max(...values);
    const span = hi - lo || 1;

    for (let b = 0; b < bins; b++) {
      const norm = (values[b] - lo) / span;
      const dir = headingUnit((b * 360) / bins);
      const inner = r * 0.32;
      const outer = inner + norm * (r - inner) * 0.94;
      ctx.strokeStyle = revealed ? COLOR[m.name] : COLOR.grid;
      ctx.globalAlpha = revealed ? 0.25 + 0.65 * norm : 0.3;
      ctx.lineWidth = 5;
      ctx.lineCap = "round";
      ctx.beginPath();
      ctx.moveTo(cx + dir.x * inner, cy - dir.y * inner);
      ctx.lineTo(cx + dir.x * outer, cy - dir.y * outer);
      ctx.stroke();
    }
    ctx.globalAlpha = 1;

    if (revealed) {
      const truth = headingUnit((plan.true_bin * 360) / bins);
      ctx.strokeStyle = COLOR.truth;
      ctx.lineWidth = 2.5;
      ctx.setLineDash([5, 4]);
      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.lineTo(cx + truth.x * r, cy - truth.y * r);
      ctx.stroke();
      ctx.setLineDash([]);

      const bin = activeBin(m);
      const pred = headingUnit((bin * 360) / bins);
      ctx.strokeStyle = COLOR[m.name];
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.lineTo(cx + pred.x * r, cy - pred.y * r);
      ctx.stroke();
    }

    const err = useEnsemble ? m.ensemble.error_deg : m.error_deg;
    ctx.fillStyle = COLOR.ink;
    ctx.font = "bold 12px ui-sans-serif, system-ui, sans-serif";
    ctx.textAlign = "center";
    ctx.fillText(m.label, cx, cy - r - 20);
    ctx.fillStyle = revealed ? COLOR[m.name] : COLOR.muted;
    ctx.font = "11px ui-sans-serif, system-ui, sans-serif";
    ctx.fillText(revealed ? `Fehler ${err.toFixed(0)}°` : "…", cx, cy - r - 6);
    if (revealed && useEnsemble) {
      ctx.fillStyle = COLOR.muted;
      ctx.fillText(`±${m.ensemble.circular_sd_deg.toFixed(0)}° über ${m.ensemble.size} Seeds`, cx, cy + r + 16);
    }
    ctx.restore();
  }

  // ----------------------------------------------------------------- raster
  function drawRaster() {
    const width = FIELD_PX + RING_PX + 12;
    const ctx = fitCanvas(rasterCanvas, width, RASTER_H);
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, width, RASTER_H);
    const plan = state.plan;
    const padL = 74, padR = 12, padT = 18, gap = 24;
    const inH = 96, resH = 62;

    ctx.fillStyle = COLOR.muted;
    ctx.font = "11px ui-sans-serif, system-ui, sans-serif";
    ctx.textAlign = "right";
    ctx.fillText("Input", padL - 8, padT + inH / 2);
    ctx.fillText("36 Bins", padL - 8, padT + inH / 2 + 13);
    ctx.fillText("Reservoir", padL - 8, padT + inH + gap + resH / 2);

    if (!plan) {
      ctx.textAlign = "center";
      ctx.fillText("Spikes erscheinen während des Flugs", width / 2, RASTER_H / 2);
      return;
    }

    const plotW = width - padL - padR;
    const tx = (t) => padL + (t / plan.n_steps) * plotW;
    const t0 = revealedSteps();

    // segment bands, so a viewer sees which leg drives which spikes
    plan.segments.forEach((seg, i) => {
      ctx.fillStyle = i % 2 ? "#f8fafc" : "#f1f5f9";
      ctx.fillRect(tx(seg.t_start), padT, tx(seg.t_end) - tx(seg.t_start), inH);
    });
    ctx.fillStyle = "#f8fafc";
    ctx.fillRect(padL, padT + inH + gap, plotW, resH);

    // input spikes, revealed as the bird flies
    ctx.fillStyle = COLOR.ink;
    for (const [t, bin] of plan.input_spikes) {
      if (t > t0) continue;
      const y = padT + (bin / plan.heading_bins) * inH;
      ctx.fillRect(tx(t), y, 1.6, 3.4);
    }

    const maxUnit = Math.max(
      1,
      ...plan.models.map((m) => (m.reservoir_spikes.length ? Math.max(...m.reservoir_spikes.map((p) => p[1])) : 1)),
    );
    plan.models.forEach((m) => {
      ctx.fillStyle = COLOR[m.name];
      ctx.globalAlpha = 0.85;
      for (const [t, unit] of m.reservoir_spikes) {
        if (t > t0) continue;
        const y = padT + inH + gap + (unit / (maxUnit + 1)) * resH;
        ctx.fillRect(tx(t), y, 1.6, 2.6);
      }
    });
    ctx.globalAlpha = 1;

    // time cursor
    if (state.phase === PHASE.OUTBOUND) {
      ctx.strokeStyle = COLOR.release;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(tx(t0), padT - 6);
      ctx.lineTo(tx(t0), padT + inH + gap + resH + 4);
      ctx.stroke();
    }

    ctx.fillStyle = COLOR.muted;
    ctx.font = "10px ui-sans-serif, system-ui, sans-serif";
    ctx.textAlign = "left";
    ctx.fillText("0 ms", padL, RASTER_H - 20);
    ctx.textAlign = "right";
    ctx.fillText(`${plan.n_steps} ms`, width - padR, RASTER_H - 20);
    ctx.textAlign = "center";
    ctx.fillText(
      `${plan.input_spike_count} Input-Spikes · ` +
        plan.models
          .map((m) => `${m.name}: ${m.reservoir_spike_count} Reservoir-Spikes auf ${m.reservoir_active_units} Neuronen`)
          .join(" · "),
      width / 2,
      RASTER_H - 5,
    );
  }

  /**
   * Held-out error distribution for one model, with this route marked on it.
   * A single flight sits somewhere in a wide spread; drawing the spread is what
   * stops that one sample from reading like a result.
   */
  function drawDistributionStrip(canvas, m) {
    const ref = m.reference;
    const W = 172, H = 40;
    const ctx = fitCanvas(canvas, W, H);
    const plotH = H - 13;
    const counts = ref.histogram;
    const peak = Math.max(1, ...counts);
    const barW = W / counts.length;

    ctx.fillStyle = COLOR[m.name];
    ctx.globalAlpha = 0.28;
    counts.forEach((c, i) => {
      const h = (c / peak) * plotH;
      ctx.fillRect(i * barW, plotH - h, barW - 1, h);
    });
    ctx.globalAlpha = 1;

    const xOf = (deg) => Math.min(W - 1, (deg / 180) * W);

    ctx.strokeStyle = COLOR.muted;
    ctx.setLineDash([3, 3]);
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(xOf(ref.mean_deg), 0);
    ctx.lineTo(xOf(ref.mean_deg), plotH);
    ctx.stroke();
    ctx.setLineDash([]);

    const here = ensembleBox.checked ? m.ensemble.error_deg : m.error_deg;
    ctx.strokeStyle = COLOR[m.name];
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(xOf(here), 0);
    ctx.lineTo(xOf(here), plotH);
    ctx.stroke();

    ctx.fillStyle = COLOR.muted;
    ctx.font = "9px ui-sans-serif, system-ui, sans-serif";
    ctx.textAlign = "left";
    ctx.fillText("0°", 0, H - 2);
    ctx.textAlign = "center";
    ctx.fillText(`Ø ${ref.mean_deg.toFixed(0)}°`, W / 2, H - 2);
    ctx.textAlign = "right";
    ctx.fillText("180°", W, H - 2);
  }

  // ---------------------------------------------------------------- readout
  function drawReadout() {
    readout.innerHTML = "";
    note.innerHTML = "";
    const plan = state.plan;
    if (!plan) return;
    const useEnsemble = ensembleBox.checked;
    // Optional blocks: a kernel running an older pigeonpilot omits them, and a
    // missing panel must never cost the flight.
    const reference = plan.reference || {};
    const testMetrics = reference.test_metrics || {};

    const truth = el("div", "pp-card");
    truth.append(el("div", "pp-card-label", "Wahre Heimrichtung"));
    truth.append(el("div", "pp-card-value", `${plan.true_heading_deg.toFixed(0)}° · Bin ${plan.true_bin}`));
    truth.style.borderLeftColor = COLOR.truth;
    readout.append(truth);

    for (const m of plan.models) {
      const bin = useEnsemble ? m.ensemble.bin : m.pred_bin;
      const err = useEnsemble ? m.ensemble.error_deg : m.error_deg;
      const card = el("div", "pp-card");
      card.style.borderLeftColor = COLOR[m.name];
      card.append(el("div", "pp-card-label", m.label));
      card.append(el("div", "pp-card-value", `${((bin * 360) / plan.heading_bins).toFixed(0)}° · Fehler ${err.toFixed(0)}°`));
      const test = testMetrics[m.name];
      card.append(
        el(
          "div",
          "pp-card-sub",
          useEnsemble
            ? `Streuung ±${m.ensemble.circular_sd_deg.toFixed(0)}° · ${m.ensemble.size} Poisson-Seeds`
            : test
              ? `Testset Ø ${test.mean_deg.toFixed(1)}° · Zufall 90°`
              : "Zufall 90°",
        ),
      );
      if (m.reference) {
        const strip = el("canvas", "pp-strip");
        card.append(strip);
        card.append(
          el(
            "div",
            "pp-card-sub",
            `Diese Route ist besser als ${(100 * m.reference.better_than).toFixed(0)}% der ${m.reference.n} Testrouten.`,
          ),
        );
        // Canvas must be in the DOM before it can be sized and painted.
        requestAnimationFrame(() => drawDistributionStrip(strip, m));
      }
      readout.append(card);
    }

    const p = plan.preprocessing;
    const scaled = Math.abs(p.scale_factor - 1) > 1e-6;
    const bits = [
      scaled
        ? `Pfad auf Trainingsmaßstab normiert: ×${p.scale_factor.toFixed(2)} (Länge ${p.raw_total_distance.toFixed(1)} → ${p.total_distance.toFixed(1)}), Form unverändert.`
        : `Pfadlänge ${p.total_distance.toFixed(1)} liegt bereits im Trainingsband (${p.band_min.toFixed(1)}–${p.band_max.toFixed(1)}), nicht skaliert.`,
      `${p.n_blocks} Richtungsblöcke (Training: max ${p.band_max_blocks}).`,
      `Der Readout liefert eine Richtung (1 von ${plan.heading_bins} Bins), keinen Rückweg — die Heimstrecke nutzt die wahre Distanz.`,
    ];
    if (!p.in_distribution) bits.push("Achtung: außerhalb der Trainingsverteilung.");
    note.append(el("div", null, bits.join(" ")));

    // One route is one sample. Without this line a lucky flight reads as a result.
    const h2h = reference.head_to_head || {};
    if (h2h.pair && plan.models.length === 2) {
      const [first, second] = [plan.models[0].name, plan.models[1].name];
      const firstWins = h2h[`${first}_better`];
      const secondWins = h2h[`${second}_better`];
      if (firstWins != null && secondWins != null) {
        const warn = el(
          "div",
          "pp-warn",
          `Eine einzelne Route ist kein Ergebnis: über ${reference.n} Testrouten gewinnt ` +
            `${first} ${(100 * firstWins).toFixed(0)}%, ${second} ${(100 * secondWins).toFixed(0)}%, ` +
            `${(100 * (h2h.tie ?? 0)).toFixed(0)}% enden gleich. ` +
            `Dass hier einmal ${second} vorn liegt, ist daher zu erwarten und widerspricht dem Gesamtergebnis nicht.`,
        );
        note.append(warn);
      }
    }
  }

  // ------------------------------------------------------------------ loop

  /**
   * Run a panel's draw code so a failure in one panel cannot strand the widget.
   * An unguarded throw inside the change:plan handler used to abort before
   * startAnimation(), leaving the status stuck on "Reservoir läuft …" with the
   * reason visible only in the browser console.
   */
  function guard(label, fn) {
    try {
      fn();
      return true;
    } catch (err) {
      console.error(`PigeonPilot: ${label} failed`, err);
      if (!state.reportedError) {
        state.reportedError = true;
        setStatus(`Anzeigefehler in "${label}" — Details in der Browser-Konsole. Der Flug läuft weiter.`);
      }
      return false;
    }
  }

  function redraw() {
    guard("Feld", drawField);
    guard("Ringe", drawRings);
    guard("Raster", drawRaster);
  }

  function tick(now) {
    if (!state.running) return;
    const dt = Math.min(0.05, (now - state.lastFrame) / 1000) * parseFloat(speed.value);
    state.lastFrame = now;
    state.clock += dt;

    if (state.phase === PHASE.OUTBOUND && state.clock >= OUTBOUND_FLIGHT_S) {
      state.phase = PHASE.RELEASE;
      state.clock = 0;
      setStatus("Aussetzpunkt erreicht — Reservoir liest die Heimrichtung ab.");
    } else if (state.phase === PHASE.RELEASE && state.clock >= RELEASE_HOLD_S) {
      state.phase = PHASE.HOME;
      state.clock = 0;
      setStatus("Heimflug: A und B fliegen ihre jeweils vorhergesagte Richtung.");
    } else if (state.phase === PHASE.HOME && state.clock >= HOME_FLIGHT_S) {
      state.phase = PHASE.DONE;
      state.clock = 0;
      state.running = false;
      setStatus("Fertig — Ringe zeigen das volle Richtungsprofil beider Modelle.");
    }
    redraw();
    if (state.running) requestAnimationFrame(tick);
  }

  function startAnimation() {
    if (!state.plan) return;
    state.phase = PHASE.OUTBOUND;
    state.clock = 0;
    state.running = true;
    state.lastFrame = performance.now();
    setStatus("Verschleppung: die Taube fliegt, unten feuern die Bins live mit.");
    requestAnimationFrame(tick);
  }

  function setStatus(text) {
    status.textContent = text;
  }

  /**
   * A request that never comes back must say so. Silence here means the kernel
   * is busy, dead, or running a pigeonpilot version that predates this widget —
   * all of which used to look identical to "still computing".
   */
  function startWatchdog() {
    clearWatchdog();
    state.watchdog = setTimeout(() => {
      state.watchdog = null;
      setStatus(
        "Keine Antwort vom Kernel. Meist hilft: Kernel neu starten und alle Zellen erneut ausführen " +
          "(der Kernel hält sonst eine ältere pigeonpilot-Version im Speicher).",
      );
    }, WATCHDOG_MS);
  }

  function clearWatchdog() {
    if (state.watchdog) clearTimeout(state.watchdog);
    state.watchdog = null;
  }

  // -------------------------------------------------------------- pointer
  function strokePoint(event) {
    const rect = fieldCanvas.getBoundingClientRect();
    const g = fieldGeometry();
    const w = g.toWorld(event.clientX - rect.left, event.clientY - rect.top);
    return [w.x, w.y];
  }

  fieldCanvas.addEventListener("pointerdown", (event) => {
    if (model.get("busy")) return;
    fieldCanvas.setPointerCapture(event.pointerId);
    state.drawing = true;
    state.plan = null;
    state.phase = PHASE.IDLE;
    state.running = false;
    state.stroke = [[0, 0], strokePoint(event)];
    redraw();
    drawReadout();
  });

  fieldCanvas.addEventListener("pointermove", (event) => {
    if (!state.drawing) return;
    state.stroke.push(strokePoint(event));
    drawField();
  });

  const endStroke = () => {
    if (!state.drawing) return;
    state.drawing = false;
    setStatus(`Pfad erfasst (${state.stroke.length} Punkte) — klick Fly.`);
  };
  fieldCanvas.addEventListener("pointerup", endStroke);
  fieldCanvas.addEventListener("pointercancel", endStroke);

  // -------------------------------------------------------------- controls
  flyBtn.addEventListener("click", () => {
    if (state.stroke.length < 2) {
      setStatus("Erst zeichnen: vom grünen Stern aus mit gedrückter Maustaste.");
      return;
    }
    setStatus("Reservoir läuft …");
    model.set("request", { points: state.stroke.map((p) => [p[0], p[1]]), nonce: ++requestNonce });
    model.save_changes();
    startWatchdog();
  });

  replayBtn.addEventListener("click", () => {
    if (state.plan) startAnimation();
  });

  clearBtn.addEventListener("click", () => {
    clearWatchdog();
    state.stroke = [];
    state.plan = null;
    state.phase = PHASE.IDLE;
    state.running = false;
    state.reportedError = false;
    state.half = model.get("field_half") || 7;
    setStatus("Gelöscht — zeichne einen neuen Pfad vom grünen Stern aus.");
    redraw();
    drawReadout();
  });

  ensembleBox.addEventListener("change", () => {
    redraw();
    drawReadout();
  });

  // ----------------------------------------------------------------- comm
  model.on("change:plan", () => {
    const plan = model.get("plan");
    if (!plan || !plan.segments) {
      clearWatchdog();
      setStatus((plan && plan.error) || "Kein Flugplan erhalten — zeichne noch einmal.");
      return;
    }
    clearWatchdog();
    state.plan = plan;
    state.reportedError = false;
    const extent = Math.max(
      ...plan.snapped_points.flatMap((p) => [Math.abs(p[0]), Math.abs(p[1])]),
      (model.get("band") || {}).band_max_release || 6,
    );
    state.half = extent * 1.18;
    // Animation first: the flight must not depend on the readout rendering.
    startAnimation();
    guard("Kennzahlen", drawReadout);
  });

  model.on("change:status", () => setStatus(model.get("status")));

  redraw();
  drawReadout();
}

export default { render };
