// app: timeline, view state, input, HUD, inspector. reads replay files only, never calls python.
import { FrameStore, loadIndex, loadManifest, slotOf, value } from './data.js';
import { drawStrip, verdictHtml } from './story.js';
import {
  CLASS_COLORS, FILLS, PATCH_FILL, VERDICT_COLORS, arcWorld, bandImage, legendHtml, nearGround, pointColor, screenToWorld,
  worldToScreen,
} from './view.js';

const REPLAY_ROOT = '../outputs/replay/';
const DEFAULT_FOLDERS = ['08_2400-2699', '08_1500-1799'];  // first one present wins when no ?replay=
const GHOST_HOLD = 10;  // ghost clears stay visible this many scans, fading
const BUDGET = { map: 65, label: 30, corridor: 5, total: 100 };  // ms p95 budgets, docs/INTERFACES.md
const PREFETCH = 24;
const $ = (id) => document.getElementById(id);

const st = {
  i: 0, playing: false, speed: 1, fill: 'hazard', mode: 'ego',
  s: 8, panX: 0, panY: 0, anchor: [0, 0], pin: null, M: null,
  shown: null, dirty: true, fps: 0, W: 0, H: 0,
};
let man, store;

// full screen message over map
function status(msg) {
  $('status').hidden = !msg;
  $('status').textContent = msg ?? '';
}

// replay folder from ?replay= else first default folder listed in scene index, always ending in /
function replayBase(scenes) {
  let p = new URLSearchParams(location.search).get('replay');
  if (!p) {
    const have = new Set((scenes ?? []).map((s) => s.folder));
    const pick = DEFAULT_FOLDERS.find((f) => have.has(f)) ?? scenes?.[0]?.folder ?? DEFAULT_FOLDERS[1];
    p = REPLAY_ROOT + pick;
  }
  return new URL(p.endsWith('/') ? p : p + '/', location.href);
}

// scene menu: every folder in index.json; picking one reloads with ?replay=
function fillScenes(scenes, base) {
  const sel = $('scene');
  const here = base.href;
  const list = scenes ?? [];
  const opts = list.map((s) => {
    const url = new URL(REPLAY_ROOT + s.folder + '/', location.href).href;
    return `<option value="${REPLAY_ROOT + s.folder}/" ${url === here ? 'selected' : ''}>${s.synthetic && !s.scene.startsWith('SIM') ? 'SIM · ' : ''}${s.scene}</option>`;
  });
  if (!list.some((s) => new URL(REPLAY_ROOT + s.folder + '/', location.href).href === here)) {
    opts.unshift(`<option selected>${man.scene ?? `seq ${man.sequence} ${man.scans[0]}-${man.scans[1]}`}</option>`);
  }
  sel.innerHTML = opts.join('');
  sel.onchange = (e) => { location.search = `?replay=${encodeURIComponent(e.target.value)}`; };
}

// real vs simulation badge + banner. sim replays carry manifest.synthetic = true
function fillBadge() {
  const sim = !!man.synthetic;
  $('badge').textContent = sim ? 'SIMULATION DEMO (NON-BENCHMARK)' : 'REAL KITTI REPLAY';
  $('badge').className = `badge ${sim ? 'sim' : 'real'}`;
  $('simbanner').hidden = !sim;
  if (sim) {
    $('cor-note').textContent = 'Synthetic scene. Corridor check (real code) marks arcs; virtual car drives the non-blocked arc ' +
      'nearest the route (yellow dashed, hidden when same as cyan pick). White boxes = sim truth. Not a result.';
  }
}

// view params for frame: car centred (heading up or world axes) or world fixed anchor
function viewOf(meta) {
  const [x, y, , yaw] = meta.pose;
  const base = { s: st.s, panX: st.panX, panY: st.panY };
  if (st.mode === 'world') return { ...base, cx: st.anchor[0], cy: st.anchor[1], phi: 0 };
  // heading up: turn world by pi/2 - yaw so car x axis points to screen top
  return { ...base, cx: x, cy: y, phi: st.mode === 'ego' ? Math.PI / 2 - yaw : 0 };
}

// colour context for band images of one frame
function fillCtx(fr) {
  const nan = Object.fromEntries(man.columns.map((c) => [c.name, c.nan]));
  fr.nearGround ??= nearGround(fr, nan.ground);
  return { nan, nearGround: fr.nearGround, flagInherited: man.flags.inherited };
}

// size canvas to css box x device pixel ratio
function fitCanvas(cv) {
  const dpr = window.devicePixelRatio || 1, W = cv.clientWidth, H = cv.clientHeight;
  if (cv.width !== Math.round(W * dpr) || cv.height !== Math.round(H * dpr)) {
    cv.width = Math.round(W * dpr);
    cv.height = Math.round(H * dpr);
  }
  return { dpr, W, H };
}

// draw map + overlays for shown frame
function draw() {
  const fr = st.shown;
  if (!fr) return;
  const cv = $('map'), g = cv.getContext('2d'), { dpr, W, H } = fitCanvas(cv);
  st.W = W;
  st.H = H;
  g.setTransform(1, 0, 0, 1, 0, 0);
  g.fillStyle = '#0d1117';
  g.fillRect(0, 0, cv.width, cv.height);
  const M = worldToScreen(viewOf(fr.meta), W, H);
  st.M = M;
  g.setTransform(...M.map((v) => v * dpr));
  g.imageSmoothingEnabled = false;  // pixel sharp: 10 cm vs 50 cm cells stay visible
  const imgs = (fr.images[st.fill] ??= fr.bands.map((b) => bandImage(b, st.fill, fillCtx(fr))));
  const [cx, cy] = fr.meta.center;
  // each band clipped to own ring: every spot shows the cell FoveatedLayers.lookup returns there.
  // base rings far to near, then refined-patch overlays on top (overlay wins where it has a cell)
  const order = fr.bands.map((b, k) => k).sort((a, b) => (fr.bands[a].overlay - fr.bands[b].overlay) || (b - a));
  for (const k of order) {
    const b = fr.bands[k];
    g.save();
    g.beginPath();
    g.arc(cx, cy, b.r_max, 0, 2 * Math.PI);
    if (b.r_min > 0) g.arc(cx, cy, b.r_min, 0, 2 * Math.PI);
    g.clip('evenodd');
    g.drawImage(imgs[k], b.x_min, b.y_min, b.nx * b.res, b.ny * b.res);
    g.restore();
  }
  const lw = (px) => px / st.s;  // world units per screen px (uniform scale)
  if ($('refined').checked) {
    const pimgs = (fr.images.patch ??= fr.bands.map((b) => (b.overlay ? bandImage(b, PATCH_FILL, fillCtx(fr)) : null)));
    g.globalAlpha = 0.65;
    fr.bands.forEach((b, k) => { if (pimgs[k]) g.drawImage(pimgs[k], b.x_min, b.y_min, b.nx * b.res, b.ny * b.res); });
    g.globalAlpha = 1;
  }
  if ($('points').checked && fr.points) drawPoints(g, fr, lw);
  if ($('ghosts').checked) drawGhosts(g, fr.meta, lw);

  if ($('path').checked) {
    const P = man.frames.map((f) => f.pose);
    g.lineJoin = 'round';
    for (const [from, to, color] of [[0, P.length, 'rgba(88,166,255,0.35)'], [0, fr.meta.i + 1, '#58a6ff']]) {
      g.beginPath();
      P.slice(from, to).forEach(([x, y], k) => (k ? g.lineTo(x, y) : g.moveTo(x, y)));
      g.strokeStyle = color;
      g.lineWidth = lw(2);
      g.stroke();
    }
  }
  if ($('rings').checked) {
    g.setLineDash([lw(6), lw(4)]);
    for (const b of man.bands.filter((b) => !b.overlay)) {
      g.beginPath();
      g.arc(cx, cy, b.r_max, 0, 2 * Math.PI);
      g.strokeStyle = 'rgba(230,237,243,0.7)';
      g.lineWidth = lw(1.25);
      g.stroke();
    }
    g.setLineDash([]);
  }
  if (man.sim) drawSimTruth(g, fr.meta, lw);
  if ($('corridor').checked && man.corridor && fr.meta.corridor) drawCorridor(g, fr.meta, lw);
  if ($('car').checked) {
    const [x, y, , yaw] = fr.meta.pose, [x0, x1, y0, y1] = man.ego_box_velo;
    const co = Math.cos(yaw), si = Math.sin(yaw);
    // ego box corners: velodyne frame -> world (turn by yaw, shift by sensor xy)
    const w = (u, v) => [x + co * u - si * v, y + si * u + co * v];
    g.beginPath();
    [[x0, y0], [x1, y0], [x1, y1], [x0, y1]].map(([u, v]) => w(u, v)).forEach(([u, v], k) => (k ? g.lineTo(u, v) : g.moveTo(u, v)));
    g.closePath();
    g.fillStyle = '#e6edf3';
    g.fill();
    g.beginPath();
    g.moveTo(...w(x1, 0));
    g.lineTo(...w(x1 + 2.5, 0));
    g.strokeStyle = '#e6edf3';
    g.lineWidth = lw(2);
    g.stroke();
  }

  // screen space: ring labels + pin
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.font = '12px system-ui, sans-serif';
  if ($('rings').checked) {
    const cu = M[0] * cx + M[2] * cy + M[4], cv2 = M[1] * cx + M[3] * cy + M[5];
    man.bands.filter((b) => !b.overlay).forEach((b) => {
      const zone = Math.abs(b.r_max - man.zone_m) < 1e-6 ? ' · model zone edge' : '';
      const t = `${b.r_max} m · inside: ${Math.round(b.resolution * 100)} cm cells${zone}`;
      g.fillStyle = 'rgba(13,17,23,0.75)';
      const tw = g.measureText(t).width;
      g.fillRect(cu - tw / 2 - 4, cv2 - b.r_max * st.s - 16, tw + 8, 16);
      g.fillStyle = '#e6edf3';
      g.textAlign = 'center';
      g.fillText(t, cu, cv2 - b.r_max * st.s - 4);
    });
  }
  if (st.pin) {
    const [px, py] = st.pin, u = M[0] * px + M[2] * py + M[4], v = M[1] * px + M[3] * py + M[5];
    g.beginPath();
    g.arc(u, v, 7, 0, 2 * Math.PI);
    g.strokeStyle = '#ffffff';
    g.lineWidth = 2;
    g.stroke();
    g.beginPath();
    g.arc(u, v, 9, 0, 2 * Math.PI);
    g.strokeStyle = '#000000';
    g.lineWidth = 1;
    g.stroke();
  }
  if (st.shown.meta.i !== st.i) {
    g.fillStyle = '#e6edf3';
    g.textAlign = 'right';
    g.fillText('loading frame…', W - 12, 20);
  }
}

// current scan points as dots, coloured by height over terrain. world = sensor xy + stored offset
function drawPoints(g, fr, lw) {
  const p = fr.points, [sx, sy] = fr.meta.pose, r = lw(1.6);
  const groups = new Map();
  for (let j = 0; j < p.n; j++) {
    const h = p.h[j] === -128 ? NaN : p.h[j] * 0.02, c = pointColor(h);
    let a = groups.get(c);
    if (!a) groups.set(c, (a = []));
    a.push(j);
  }
  for (const [c, idx] of groups) {
    g.fillStyle = c;
    g.globalAlpha = c === '#9da7b3' ? 0.45 : 0.95;
    for (const j of idx) g.fillRect(sx + p.dx[j] * 0.01 - r / 2, sy + p.dy[j] * 0.01 - r / 2, r, r);
  }
  g.globalAlpha = 1;
}

// cells ghost clearing emptied in last GHOST_HOLD scans: magenta squares, fading with age
function drawGhosts(g, meta, lw) {
  const side = Math.max(0.3, lw(3));
  for (let a = GHOST_HOLD - 1; a >= 0; a--) {
    const f = man.frames[meta.i - a];
    if (!f?.cleared?.length) continue;
    const [sx, sy] = f.pose;
    g.globalAlpha = 0.9 * (1 - a / GHOST_HOLD);
    g.fillStyle = '#ff4fd8';
    for (const [dx, dy] of f.cleared) g.fillRect(sx + dx - side / 2, sy + dy - side / 2, side, side);
  }
  g.globalAlpha = 1;
}

// sim only: road edges + footway edge from synthetic centreline, truth boxes (walls faint, obstacles + movers bold white)
function drawSimTruth(g, meta, lw) {
  const sim = man.sim, R = sim.route;
  const edge = (d) => R.map(([x, y], k) => {
    const [x2, y2] = R[Math.min(k + 1, R.length - 1)], [x1, y1] = R[Math.max(k - 1, 0)];
    const h = Math.atan2(y2 - y1, x2 - x1);
    return [x - d * Math.sin(h), y + d * Math.cos(h)];
  });
  g.setLineDash([]);
  for (const [d, color] of [[sim.road_half_m, '#c9d1d9'], [-sim.road_half_m, '#c9d1d9'], [sim.road_half_m + 3, '#6e7681'], [-sim.road_half_m - 3, '#6e7681']]) {
    g.beginPath();
    edge(d).forEach(([x, y], k) => (k ? g.lineTo(x, y) : g.moveTo(x, y)));
    g.strokeStyle = color;
    g.lineWidth = lw(1);
    g.stroke();
  }
  const box = ([cx, cy, yaw, L, W, , kind], bold) => {
    const c = Math.cos(yaw), s = Math.sin(yaw);
    const pts = [[L / 2, W / 2], [-L / 2, W / 2], [-L / 2, -W / 2], [L / 2, -W / 2]].map(([u, v]) => [cx + c * u - s * v, cy + s * u + c * v]);
    g.beginPath();
    pts.forEach(([x, y], k) => (k ? g.lineTo(x, y) : g.moveTo(x, y)));
    g.closePath();
    g.strokeStyle = bold ? '#ffffff' : 'rgba(201,209,217,0.35)';
    g.lineWidth = lw(bold ? 2 : 1);
    g.stroke();
  };
  for (const b of sim.boxes) box(b, b[6] !== 'wall');
  for (const b of meta.sim?.movers ?? []) box(b, true);
}

// total pipeline ms of frame (corridor only in replays that have it)
const totalMs = (f) => f.map_ms + f.label_ms + (f.corridor_ms ?? 0);

// world-space corridor overlay: driven path next 20 m, 15 arcs coloured by verdict, picked arc in cyan
function drawCorridor(g, meta, lw) {
  const c = man.corridor, cor = meta.corridor, step = 0.5;
  // path the driver really took next (poses), up to arc length along path
  const P = man.frames.map((f) => f.pose);
  g.beginPath();
  let along = 0;
  g.moveTo(P[meta.i][0], P[meta.i][1]);
  for (let j = meta.i + 1; j < P.length && along < c.length_m; j++) {
    along += Math.hypot(P[j][0] - P[j - 1][0], P[j][1] - P[j - 1][1]);
    g.lineTo(P[j][0], P[j][1]);
  }
  g.setLineDash([lw(6), lw(5)]);
  g.strokeStyle = '#ffffff';
  g.lineWidth = lw(2);
  g.stroke();
  g.setLineDash([]);
  c.curvatures.forEach((k, a) => {
    if (a === cor.pick) return;
    const pts = arcWorld(k, c.length_m, step, meta.pose), cut = Math.round(cor.b[a] / step);
    const color = VERDICT_COLORS[cor.v[a]];
    // solid up to first obstacle cluster, faded after
    for (const [from, to, alpha] of [[0, cut + 1, 0.95], [cut, pts.length, 0.2]]) {
      if (to - from < 2) continue;
      g.beginPath();
      pts.slice(from, to).forEach(([x, y], j) => (j ? g.lineTo(x, y) : g.moveTo(x, y)));
      g.globalAlpha = alpha;
      g.strokeStyle = '#0d1117';  // dark halo keeps arc readable over red / green cells
      g.lineWidth = lw(6);
      g.stroke();
      g.strokeStyle = color;
      g.lineWidth = lw(3);
      g.stroke();
    }
    g.globalAlpha = 1;
    if (cor.v[a] === 2 && cut < pts.length) {
      const [x, y] = pts[cut], r = lw(6);
      g.beginPath();
      g.moveTo(x - r, y - r); g.lineTo(x + r, y + r);
      g.moveTo(x - r, y + r); g.lineTo(x + r, y - r);
      g.strokeStyle = '#f85149';
      g.lineWidth = lw(3);
      g.stroke();
    }
  });
  if (cor.pick >= 0) {
    const pts = arcWorld(c.curvatures[cor.pick], c.length_m, step, meta.pose);
    const line = () => { g.beginPath(); pts.forEach(([x, y], j) => (j ? g.lineTo(x, y) : g.moveTo(x, y))); };
    line();
    g.globalAlpha = 0.25;
    g.strokeStyle = '#39d0f0';
    g.lineWidth = 2 * c.half_width_m;  // footprint band, world metres
    g.lineCap = 'round';
    g.stroke();
    g.globalAlpha = 1;
    line();
    g.strokeStyle = '#0d1117';
    g.lineWidth = lw(7);
    g.stroke();
    g.strokeStyle = '#39d0f0';
    g.lineWidth = lw(4);
    g.stroke();
    g.lineCap = 'butt';
    // arrow head at tip, along last segment
    const [x1, y1] = pts.at(-1), [x0, y0] = pts.at(-2), ang = Math.atan2(y1 - y0, x1 - x0), h = lw(14);
    g.beginPath();
    g.moveTo(x1, y1);
    g.lineTo(x1 - h * Math.cos(ang - 0.45), y1 - h * Math.sin(ang - 0.45));
    g.lineTo(x1 - h * Math.cos(ang + 0.45), y1 - h * Math.sin(ang + 0.45));
    g.closePath();
    g.fillStyle = '#39d0f0';
    g.fill();
  }
  const ch = meta.sim?.choice;
  if (man.synthetic && ch >= 0 && ch !== cor.pick) {
    const pts = arcWorld(c.curvatures[ch], c.length_m, step, meta.pose);
    g.beginPath();
    pts.forEach(([x, y], j) => (j ? g.lineTo(x, y) : g.moveTo(x, y)));
    g.setLineDash([lw(8), lw(4)]);
    g.strokeStyle = '#ffd33d';
    g.lineWidth = lw(4);
    g.stroke();
    g.setLineDash([]);
  }
}

// corridor card on map + side rows
function corridorHud(meta) {
  const { card, rows } = verdictHtml(man, meta);
  $('verdict').innerHTML = $('corridor').checked ? card : '';
  $('h-corridor').innerHTML = rows;
}

// pipeline ms bars against per-part budgets
function msBars(meta) {
  const parts = [['Map', meta.map_ms, BUDGET.map], ['Labeler', meta.label_ms, BUDGET.label],
    ['Corridor', meta.corridor_ms, BUDGET.corridor], ['Total', totalMs(meta), BUDGET.total]];
  $('h-bars').innerHTML = parts.map(([k, v, b]) => {
    if (v === undefined) return `<span>${k}</span><span class="bar"></span><b class="mono">–</b>`;
    const pct = Math.min(100, (v / b) * 100);
    return `<span>${k}</span><span class="bar" title="budget ${b} ms p95"><i class="${v > b ? 'over' : ''}" style="width:${pct}%"></i></span>` +
      `<b class="mono ${v > b ? 'over' : ''}">${v.toFixed(1)} / ${b} ms</b>`;
  }).join('');
}

// fill event bookmark menu from manifest
function fillEvents() {
  const sel = $('events'), ev = man.corridor?.events ?? [];
  sel.innerHTML = `<option value="">${ev.length ? `${ev.length} corridor events` : 'no events'}</option>` +
    ev.map((i) => `<option value="${i}">frame ${i + 1} · scan ${man.frames[i].scan}</option>`).join('');
  sel.onchange = (e) => { if (e.target.value !== '') { setPlaying(false); seek(+e.target.value); focusCorridor(); } };
}

// zoom so corridor zone (arc length + margin) fills view; used on event jumps
function focusCorridor() {
  const cv = $('map');
  st.s = Math.min(cv.clientWidth, cv.clientHeight) / (2 * (man.corridor.length_m + 5));
  st.panX = 0;
  st.panY = cv.clientHeight * 0.25;  // car below centre so arcs ahead get the room
  st.dirty = true;
}

// jump to next / previous corridor event
function stepEvent(dir) {
  const ev = man.corridor?.events ?? [];
  const next = dir > 0 ? ev.find((i) => i > st.i) : [...ev].reverse().find((i) => i < st.i);
  if (next !== undefined) { setPlaying(false); seek(next); $('events').value = String(next); focusCorridor(); }
}

// cell under world xy: refined-patch overlay if it has an observed cell there, else base band by range
// from sensor (same rule as FoveatedLayers.lookup), then lattice cell
function cellAt(fr, x, y) {
  const [cx, cy] = fr.meta.center, r = Math.hypot(x - cx, y - cy);
  for (const [k, b] of fr.bands.entries()) {
    if (!b.overlay || r < b.r_min || r >= b.r_max) continue;
    const ix = Math.floor((x - b.x_min) / b.res), iy = Math.floor((y - b.y_min) / b.res);
    if (ix < 0 || iy < 0 || ix >= b.nx || iy >= b.ny) continue;
    const j = slotOf(b, ix * b.ny + iy);
    if (j >= 0) return { r, k, b, ix, iy, j };
  }
  const k = fr.bands.findIndex((b) => !b.overlay && r >= b.r_min && r < b.r_max);
  if (k < 0) return { r, k };
  const b = fr.bands[k], ix = Math.floor((x - b.x_min) / b.res), iy = Math.floor((y - b.y_min) / b.res);
  if (ix < 0 || iy < 0 || ix >= b.nx || iy >= b.ny) return { r, k };
  return { r, k, b, ix, iy, j: slotOf(b, ix * b.ny + iy) };
}

// inspector rows for pinned spot
function inspect() {
  const el = $('inspect'), fr = st.shown;
  if (!st.pin || !fr) return;
  const c = cellAt(fr, ...st.pin);
  const row = (k, v) => `<span>${k}</span><b class="mono">${v}</b>`;
  const m = (v, d = 2) => (Number.isNaN(v) ? '–' : `${v.toFixed(d)} m`);
  const cm = (v) => (Number.isNaN(v) ? '–' : `${Math.round(v * 100)} cm`);
  let h = row('Range from sensor', m(c.r, 1));
  if (!c.b) {
    el.innerHTML = h + row('Map', 'outside map rings');
    return;
  }
  const b = c.b, res = b.res;
  h += row('Ring', `${b.r_min}–${b.r_max} m, ${Math.round(res * 100)} cm cells${b.overlay ? ' (refined hazard patch)' : ''}`);
  h += row('Cell centre (world)', `${(b.x_min + (c.ix + 0.5) * res).toFixed(2)}, ${(b.y_min + (c.iy + 0.5) * res).toFixed(2)}`);
  if (c.j < 0) {
    el.innerHTML = h + row('State', 'unobserved — no data') +
      '<span class="small muted" style="grid-column:1/-1">Not the "unknown" class: no scan has filled this cell yet, or it decayed away.</span>';
    return;
  }
  const v = (n) => value(man, b, n, c.j);
  const lab = b.cols.label[c.j], fl = b.cols.flags[c.j];
  h += row('Label', `<span class="chip" style="background:${CLASS_COLORS[lab]}"></span>${man.classes[lab]}`);
  h += row('Label source', fl & man.flags.model ? `model (inside ${man.zone_m} m)` : 'threshold rule');
  h += row('z max (world)', m(v('z_max')));
  h += row('Height over terrain', cm(v('z_max') - v('terrain')));
  h += row('Ground (world)', m(v('ground')));
  h += row('Ground over terrain', cm(v('ground') - v('terrain')));
  fr.nearGround ??= nearGround(fr, man.col.ground.nan);
  h += row('Ground vs road near car', cm(v('ground') - fr.nearGround * man.col.ground.scale));
  h += row('Local terrain (world)', m(v('terrain')));
  h += row('Roughness (z std)', cm(v('rough')));
  h += row('Scans that saw cell', v('n_obs'));
  h += row('Confidence weight', v('weight').toFixed(2));
  h += row('Obstacle log-odds', v('logodds').toFixed(2));
  h += row('Inherited from coarse ring', fl & man.flags.inherited ? 'yes (placeholder)' : 'no');
  el.innerHTML = h;
}

// HUD numbers for frame meta
function hud(meta) {
  const tot = totalMs(meta);
  $('frameno').textContent = `${meta.i + 1}/${man.frames.length} · scan ${meta.scan} · ${(meta.t - man.frames[0].t).toFixed(1)} s`;
  msBars(meta, tot);
  corridorHud(meta);
  $('h-ghost').textContent = meta.cleared ? meta.cleared.length.toLocaleString() : '–';
  $('h-refined').textContent = meta.refined !== undefined ? meta.refined.toLocaleString() : '–';
  $('h-store').textContent = `${meta.store_mb.toFixed(2)} MB`;
  $('h-render').textContent = `${meta.render_mb.toFixed(2)} MB`;
  $('h-labelled').textContent = meta.labelled.toLocaleString();
  $('h-occ').textContent = meta.occupied.toLocaleString();
  $('h-bands').innerHTML = meta.bands.map((b, k) => {
    const s = man.bands[k];
    const what = s.overlay ? `Refined patch cells ${s.r_min}–${s.r_max} m` : `Observed cells ${s.r_min}–${s.r_max} m`;
    return `<span>${what} (${Math.round(s.resolution * 100)} cm)</span><b class="mono">${b.n_obs.toLocaleString()}</b>`;
  }).join('');
  $('h-classes').innerHTML = meta.classes.map((n, k) =>
    `<span><span class="chip" style="background:${CLASS_COLORS[k]}"></span>${man.classes[k]} cells</span><b class="mono">${n.toLocaleString()}</b>`).join('');
  $('scrub').value = meta.i;
  spark();
  if (man.corridor) drawStrip($('strip'), man, meta.i, fitCanvas);
  view3d?.setFrame(meta.i);
}

// sparkline of total ms per scan, 100 ms budget line, current frame marker
function spark() {
  const cv = $('spark'), g = cv.getContext('2d'), { dpr, W, H } = fitCanvas(cv);
  const tot = man.frames.map(totalMs), hi = Math.max(120, ...tot);
  const x = (k) => (k / (tot.length - 1)) * W, y = (v) => H - (v / hi) * (H - 4);
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, W, H);
  g.beginPath();
  tot.forEach((v, k) => (k ? g.lineTo(x(k), y(v)) : g.moveTo(x(k), y(v))));
  g.strokeStyle = '#58a6ff';
  g.lineWidth = 1;
  g.stroke();
  g.setLineDash([4, 3]);
  g.beginPath();
  g.moveTo(0, y(100));
  g.lineTo(W, y(100));
  g.strokeStyle = '#f85149';
  g.stroke();
  g.setLineDash([]);
  g.fillStyle = '#f85149';
  g.font = '10px system-ui, sans-serif';
  g.textAlign = 'left';
  g.fillText('100 ms', 2, y(100) - 2);
  g.fillStyle = '#e6edf3';
  g.fillRect(x(st.i) - 1, 0, 2, H);
}

// jump to frame i; draw when decoded
function seek(i) {
  st.i = Math.max(0, Math.min(man.frames.length - 1, i));
  hud(man.frames[st.i]);
  store.prefetch(st.i, PREFETCH);
  const ready = store.peek(st.i);
  if (ready) show(ready);
  else store.get(st.i).then((fr) => { if (fr.meta.i === st.i) show(fr); }).catch((e) => status(e.message));
  st.dirty = true;
}

// make decoded frame the drawn one
function show(fr) {
  st.shown = fr;
  st.dirty = true;
  inspect();
  view3d?.show(fr);
}

// 2D map tab or 3D chase tab. 3D module + vendored three.js load on first use only
let view3d = null;
async function setTab(tab) {
  st.tab = tab;
  $('tab2d').setAttribute('aria-selected', String(tab === '2d'));
  $('tab3d').setAttribute('aria-selected', String(tab === '3d'));
  $('map').hidden = tab !== '2d';
  $('three').hidden = tab !== '3d';
  document.querySelectorAll('.only2d').forEach((el) => { el.hidden = tab !== '2d'; });
  if (tab === '3d') {
    if (!view3d) {
      try {
        const { View3D } = await import('./view3d.js');
        view3d = new View3D($('three'), man, { $, fill: () => st.fill, fillCtx });
      } catch (e) {
        status(`3D view failed to load: ${e.message}`);
        return;
      }
    }
    view3d.active = true;
    if (st.shown) view3d.show(st.shown);
  } else if (view3d) view3d.active = false;
  st.dirty = true;
}

// play / pause
function setPlaying(on) {
  st.playing = on && st.i < man.frames.length - 1;
  $('play').textContent = st.playing ? '❚❚' : '▶︎';
  st.fpsN = 0;
  st.fpsT = performance.now();
}

// scale so outer ring fits view
function resetView() {
  const cv = $('map');
  st.s = Math.min(cv.clientWidth, cv.clientHeight) / (2 * (Math.max(...man.bands.map((b) => b.r_max)) + 3));
  st.panX = st.panY = 0;
  if (st.mode === 'world') st.anchor = man.frames[st.i].pose.slice(0, 2);
  st.dirty = true;
}

// animation loop: advance by real scan times x speed, wait when next frame not decoded yet
let last = performance.now(), acc = 0;
function tick(now) {
  const dt = Math.min((now - last) / 1000, 0.25);
  last = now;
  if (st.playing) {
    acc += dt * st.speed;
    while (st.playing) {
      if (st.i >= man.frames.length - 1) { setPlaying(false); break; }
      const step = man.frames[st.i + 1].t - man.frames[st.i].t;
      if (acc < step) break;
      if (!store.peek(st.i + 1)) { acc = step; store.prefetch(st.i + 1, PREFETCH); break; }
      acc -= step;
      seek(st.i + 1);
      st.fpsN++;
    }
    const span = (now - st.fpsT) / 1000;
    if (span > 1) {
      $('h-fps').textContent = (st.fpsN / span).toFixed(1);
      st.fpsN = 0;
      st.fpsT = now;
    }
  } else acc = 0;
  if (st.dirty) {
    st.dirty = false;
    draw();
  }
  requestAnimationFrame(tick);
}

// mouse / touch: drag pans, click pins cell, wheel zooms around cursor
function bindInput() {
  const cv = $('map');
  let down = null;
  cv.addEventListener('pointerdown', (e) => {
    down = { x: e.clientX, y: e.clientY, moved: false };
    cv.setPointerCapture(e.pointerId);
  });
  cv.addEventListener('pointermove', (e) => {
    if (!down) return;
    const dx = e.clientX - down.x, dy = e.clientY - down.y;
    if (!down.moved && Math.hypot(dx, dy) < 4) return;
    down.moved = true;
    cv.classList.add('dragging');
    st.panX += dx;
    st.panY += dy;
    down.x = e.clientX;
    down.y = e.clientY;
    st.dirty = true;
  });
  cv.addEventListener('pointerup', (e) => {
    cv.classList.remove('dragging');
    if (down && !down.moved && st.M) {
      const r = cv.getBoundingClientRect();
      st.pin = screenToWorld(st.M, e.clientX - r.left, e.clientY - r.top);
      inspect();
      st.dirty = true;
    }
    down = null;
  });
  cv.addEventListener('wheel', (e) => {
    e.preventDefault();
    const r = cv.getBoundingClientRect(), u = e.clientX - r.left, v = e.clientY - r.top;
    const s = Math.max(1, Math.min(400, st.s * Math.exp(-e.deltaY * 0.0015))), k = s / st.s;
    // keep world point under cursor fixed
    st.panX = u - st.W / 2 - k * (u - st.W / 2 - st.panX);
    st.panY = v - st.H / 2 - k * (v - st.H / 2 - st.panY);
    st.s = s;
    st.dirty = true;
  }, { passive: false });
  window.addEventListener('resize', () => { st.dirty = true; spark(); if (man.corridor) drawStrip($('strip'), man, st.i, fitCanvas); });

  $('play').onclick = () => setPlaying(!st.playing);
  $('prev').onclick = () => { setPlaying(false); seek(st.i - 1); };
  $('next').onclick = () => { setPlaying(false); seek(st.i + 1); };
  $('scrub').oninput = (e) => seek(+e.target.value);
  $('speed').onchange = (e) => { st.speed = +e.target.value; };
  $('reset').onclick = resetView;
  $('fill').onchange = (e) => {
    st.fill = e.target.value;
    $('legend').innerHTML = legendHtml(st.fill, man.classes);
    st.dirty = true;
    view3d?.redraw();
  };
  $('viewmode').onchange = (e) => {
    st.mode = e.target.value;
    st.anchor = man.frames[st.i].pose.slice(0, 2);
    st.panX = st.panY = 0;
    st.dirty = true;
  };
  for (const id of ['rings', 'path', 'car', 'corridor', 'points', 'ghosts', 'refined']) {
    $(id).onchange = () => { st.dirty = true; corridorHud(man.frames[st.i]); view3d?.redraw(); };
  }
  $('strip').onclick = (e) => {
    const r = e.target.getBoundingClientRect();
    setPlaying(false);
    seek(Math.floor(((e.clientX - r.left) / r.width) * man.frames.length));
  };
  $('tab2d').onclick = () => setTab('2d');
  $('tab3d').onclick = () => setTab('3d');
  $('spark').onclick = (e) => {
    const r = e.target.getBoundingClientRect();
    setPlaying(false);
    seek(Math.round(((e.clientX - r.left) / r.width) * (man.frames.length - 1)));
  };

  window.addEventListener('keydown', (e) => {
    if (e.target.tagName === 'SELECT' || e.target.tagName === 'INPUT' && e.target.type !== 'range') return;
    const step = e.shiftKey ? 10 : 1;
    const fills = Object.keys(FILLS), modes = ['ego', 'follow', 'world'];
    if (e.key === ' ') { e.preventDefault(); setPlaying(!st.playing); }
    else if (e.key === 'ArrowLeft') { e.preventDefault(); setPlaying(false); seek(st.i - step); }
    else if (e.key === 'ArrowRight') { e.preventDefault(); setPlaying(false); seek(st.i + step); }
    else if (e.key === 'Home') seek(0);
    else if (e.key === 'End') seek(man.frames.length - 1);
    else if (e.key >= '1' && e.key <= String(fills.length)) { $('fill').value = fills[+e.key - 1]; $('fill').onchange({ target: $('fill') }); }
    else if (e.key === 'v' || e.key === 'V') {
      $('viewmode').value = modes[(modes.indexOf(st.mode) + 1) % modes.length];
      $('viewmode').onchange({ target: $('viewmode') });
    } else if (e.key === 'r' || e.key === 'R') resetView();
    else if (e.key === 't' || e.key === 'T') setTab(st.tab === '3d' ? '2d' : '3d');
    else if (e.key === 'n' || e.key === 'N') stepEvent(1);
    else if (e.key === 'p' || e.key === 'P') stepEvent(-1);
  });
}

// boot: checks, manifest, first frame
async function main() {
  if (location.protocol === 'file:') {
    status('Browsers block file:// data loads.\nFrom repo root run:\n  python -m http.server 8000 --bind 127.0.0.1\nthen open http://127.0.0.1:8000/viewer/');
    return;
  }
  if (!('DecompressionStream' in window)) {
    status('This browser lacks DecompressionStream (gzip).\nUse current Chrome, Edge, Firefox or Safari.');
    return;
  }
  const scenes = await loadIndex(new URL(REPLAY_ROOT, location.href));
  const base = replayBase(scenes);
  try {
    man = await loadManifest(base);
  } catch (e) {
    status(`${e.message}\n\nMake replay first, from repo root:\n  python -m scripts.export_replay\n` +
      `or point viewer at another replay: ?replay=<folder>/`);
    return;
  }
  store = new FrameStore(base, man);
  fillScenes(scenes, base);
  fillBadge();
  $('scene').title = man.synthetic ? 'synthetic scene' : `seq ${man.sequence} · scans ${man.scans[0]}–${man.scans[1]} · model trained on ${man.train}`;
  $('points').checked = !!man.synthetic;  // sim scenes: points on, real replays: off (busy over labels)
  $('h-cpu').textContent = `Pipeline ms measured during export on ${man.timing.cpu}. ${man.timing.note}`;
  $('h-summary').textContent = `Whole replay (scans 2+): median ${man.timing.total_ms_median} ms, p95 ${man.timing.total_ms_p95} ms, ` +
    `${(man.timing.over_100ms_share * 100).toFixed(1)}% of scans over 100 ms budget.`;
  $('scrub').max = man.frames.length - 1;
  fillEvents();
  $('legend').innerHTML = legendHtml(st.fill, man.classes);
  bindInput();
  resetView();
  seek(0);
  requestAnimationFrame(tick);
}

main();
