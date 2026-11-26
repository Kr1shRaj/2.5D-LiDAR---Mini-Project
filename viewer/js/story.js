// corridor story: verdict card text, check vs driver vs reference, per-frame timeline strip. reads manifest only.
import { VERDICT_COLORS } from './view.js';

const CLEAR = 0, CAUTION = 1, BLOCKED = 2;

// arc index nearest to curvature k, or -1 when k missing / sharper than any arc
export function nearestArc(curvatures, k) {
  if (k === null || k === undefined || Math.abs(k) > Math.max(...curvatures.map(Math.abs)) + 1e-9) return -1;
  let best = 0;
  curvatures.forEach((c, a) => { if (Math.abs(c - k) < Math.abs(curvatures[best] - k)) best = a; });
  return best;
}

// arc in words: straight / left R 25 m / right R 10 m
export function arcName(curvatures, a) {
  const k = curvatures[a];
  return Math.abs(k) < 1e-9 ? 'straight' : `${k > 0 ? 'left' : 'right'}, radius ${Math.abs(1 / k).toFixed(0)} m`;
}

// one frame -> story facts: live straight, why, candidate, driver arc, reference straight, agreement words
export function story(man, meta) {
  const c = man.corridor, cor = meta.corridor, s = c.straight;
  const v = cor.v[s];
  const why = v === BLOCKED
    ? `obstacle cluster with map evidence at ${cor.b[s].toFixed(1)} m`
    : v === CAUTION
      ? `${Math.round(cor.h[s] * 100)} % hazard, ${Math.round(cor.u[s] * 100)} % unseen on arc`
      : 'no obstacle, little hazard, mostly seen';
  const drv = nearestArc(c.curvatures, cor.drv);
  const ref = cor.ref ? cor.ref[s] : null;
  let refWord = null;
  if (ref !== null) {
    if (ref === v) refWord = { cls: 'ok', text: 'matches reference' };
    else if (ref === BLOCKED && v === CLEAR) refWord = { cls: 'bad', text: 'reference blocked: missed (false clear)' };
    else if (ref === BLOCKED) refWord = { cls: 'warn', text: 'reference blocked: live says caution' };
    else if (v === BLOCKED) refWord = { cls: 'warn', text: `reference ${c.verdicts[ref]}: live over-blocks` };
    else refWord = { cls: 'warn', text: `reference ${c.verdicts[ref]}` };
  }
  let drvWord = null;
  if (drv >= 0 && cor.pick >= 0) {
    const gap = Math.abs(drv - cor.pick);
    drvWord = gap <= 1 ? { cls: 'ok', text: 'candidate ≈ driver' } : { cls: 'warn', text: `candidate ${gap} arcs from driver` };
  }
  return { v, why, pick: cor.pick, drv, ref, refWord, drvWord, driverBlocked: drv >= 0 && cor.v[drv] === BLOCKED };
}

const MODE_WORDS = { drive: 'drives sim arc', creep: 'CREEP: map not seen enough, 1.5 m/s', stop: 'STOP: every arc blocked' };

// sim only: virtual car speed, mode, the arc it drives (yellow). empty for real replays
function simRows(man, meta, name) {
  const sm = meta.sim;
  if (!man.synthetic || !sm) return '';
  const hit = sm.contact?.length ? `<div class="vrow"><span>Contact (truth)</span><b class="bad">${sm.contact.join(', ')}</b></div>` : '';
  return `<div class="vrow"><span>Virtual car</span><b>${sm.speed.toFixed(1)} m/s · ${MODE_WORDS[sm.mode] ?? sm.mode}</b></div>` +
    `<div class="vrow"><span>Sim drives (yellow)</span><b>${sm.choice >= 0 ? name(sm.choice) : '–'}</b></div>` + hit;
}

// big on-map card + side panel rows
export function verdictHtml(man, meta) {
  const c = man.corridor;
  if (!c || !meta.corridor) return { card: '', rows: '<span class="muted small">No corridor data in this replay. Re-export.</span>' };
  const f = story(man, meta), color = VERDICT_COLORS[f.v];
  const name = (a) => (a >= 0 ? arcName(c.curvatures, a) : '–');
  const badge = (w) => (w ? `<span class="tag ${w.cls}">${w.text}</span>` : '');
  const card = `<div class="vline"><span class="vdot" style="background:${color}"></span>` +
    `<span class="vword" style="color:${color}">STRAIGHT ${c.verdicts[f.v].toUpperCase()}</span></div>` +
    `<div class="vwhy">${f.why}</div>` +
    `<div class="vrow"><span>Candidate</span><b>${f.pick >= 0 ? name(f.pick) : 'NO CLEAR CANDIDATE'}</b></div>` +
    simRows(man, meta, name) +
    `<div class="vrow"><span>${man.synthetic ? 'Virtual car took' : 'Driver took'}</span><b>${f.drv >= 0 ? name(f.drv) : (meta.corridor.drv === undefined ? 'not in replay' : meta.corridor.drv === null ? 'end of slice' : 'sharper than arcs')}</b></div>` +
    `<div class="vtags">${badge(f.drvWord)}${badge(f.refWord)}</div>`;
  const cor = meta.corridor, row = (k, v) => `<span>${k}</span><b class="mono">${v}</b>`;
  let rows = row('Straight', f.v === BLOCKED ? `BLOCKED at ${cor.b[c.straight].toFixed(1)} m` : c.verdicts[f.v].toUpperCase());
  if (f.ref !== null) rows += row('Reference', c.verdicts[f.ref].toUpperCase());
  rows += row('Candidate', f.pick >= 0 ? `${name(f.pick)} · ${c.verdicts[cor.v[f.pick]]}` : 'none');
  if (f.pick >= 0) {
    rows += row('Clear length', `${cor.b[f.pick].toFixed(1)} m`);
    rows += row('Unseen / hazard', `${Math.round(cor.u[f.pick] * 100)} % / ${Math.round(cor.h[f.pick] * 100)} %`);
  }
  rows += row('Driver arc', f.drv >= 0 ? `${name(f.drv)} · ${c.verdicts[cor.v[f.drv]]}` : '–');
  const n = [0, 1, 2].map((x) => cor.v.filter((y) => y === x).length);
  rows += row('Clear / caution / block', `${n[0]} / ${n[1]} / ${n[2]}`);
  return { card, rows };
}

// timeline strip: row 1 live straight verdict, row 2 reference straight, row 3 candidate vs driver, ticks = events
export function drawStrip(cv, man, i, fit) {
  const g = cv.getContext('2d'), { dpr, W, H } = fit(cv);
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, W, H);
  const n = man.frames.length, x = (k) => (k / n) * W, w = Math.max(W / n, 1);
  const rows = [['live', 0], ['ref', 1], ['drv', 2]], rh = (H - 6) / rows.length;
  g.font = '10px system-ui, sans-serif';
  for (const [k, f] of man.frames.entries()) {
    const cor = f.corridor;
    if (!cor) continue;
    const s = man.corridor.straight;
    g.fillStyle = VERDICT_COLORS[cor.v[s]];
    g.fillRect(x(k), 6, w + 0.5, rh - 1);
    if (cor.ref) { g.fillStyle = VERDICT_COLORS[cor.ref[s]]; g.fillRect(x(k), 6 + rh, w + 0.5, rh - 1); }
    const d = nearestArc(man.corridor.curvatures, cor.drv);
    g.fillStyle = d < 0 || cor.pick < 0 ? '#30363d' : Math.abs(d - cor.pick) <= 1 ? '#39d0f0' : '#a371f7';
    g.fillRect(x(k), 6 + 2 * rh, w + 0.5, rh - 1);
  }
  g.fillStyle = '#e6edf3';
  for (const e of man.corridor?.events ?? []) g.fillRect(x(e), 0, Math.max(w, 1.5), 5);
  g.fillRect(x(i) - 1, 0, 2, H);
  return rows.length;
}
