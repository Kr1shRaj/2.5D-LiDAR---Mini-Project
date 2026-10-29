// app: timeline, view state, input, HUD, inspector. reads replay files only, never calls python.
import { FrameStore, loadManifest, slotOf, value } from './data.js';
import { CLASS_COLORS, FILLS, bandImage, legendHtml, nearGround, screenToWorld, worldToScreen } from './view.js';

const DEFAULT_REPLAY = '../outputs/replay/08_1500-1799/';
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

// replay folder from ?replay=, always ending in /
function replayBase() {
  const p = new URLSearchParams(location.search).get('replay') ?? DEFAULT_REPLAY;
  return new URL(p.endsWith('/') ? p : p + '/', location.href);
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
  // each band clipped to own ring: every spot shows the cell FoveatedLayers.lookup returns there
  for (let k = fr.bands.length - 1; k >= 0; k--) {
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
    for (const b of man.bands) {
      g.beginPath();
      g.arc(cx, cy, b.r_max, 0, 2 * Math.PI);
      g.strokeStyle = 'rgba(230,237,243,0.7)';
      g.lineWidth = lw(1.25);
      g.stroke();
    }
    g.setLineDash([]);
  }
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
    man.bands.forEach((b) => {
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

// cell under world xy: band by range from sensor (same rule as FoveatedLayers.lookup), then lattice cell
function cellAt(fr, x, y) {
  const [cx, cy] = fr.meta.center, r = Math.hypot(x - cx, y - cy);
  const k = fr.bands.findIndex((b) => r >= b.r_min && r < b.r_max);
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
  h += row('Ring', `${b.r_min}–${b.r_max} m, ${Math.round(res * 100)} cm cells`);
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
  const tot = meta.map_ms + meta.label_ms;
  $('frameno').textContent = `${meta.i + 1}/${man.frames.length} · scan ${meta.scan} · ${(meta.t - man.frames[0].t).toFixed(1)} s`;
  $('h-map').textContent = `${meta.map_ms.toFixed(1)} ms`;
  $('h-label').textContent = `${meta.label_ms.toFixed(1)} ms`;
  $('h-total').textContent = `${tot.toFixed(1)} ms`;
  $('h-total').classList.toggle('over', tot > 100);
  $('h-store').textContent = `${meta.store_mb.toFixed(2)} MB`;
  $('h-render').textContent = `${meta.render_mb.toFixed(2)} MB`;
  $('h-labelled').textContent = meta.labelled.toLocaleString();
  $('h-occ').textContent = meta.occupied.toLocaleString();
  $('h-bands').innerHTML = meta.bands.map((b, k) => {
    const s = man.bands[k];
    return `<span>Observed cells ${s.r_min}–${s.r_max} m (${Math.round(s.resolution * 100)} cm)</span><b class="mono">${b.n_obs.toLocaleString()}</b>`;
  }).join('');
  $('h-classes').innerHTML = meta.classes.map((n, k) =>
    `<span><span class="chip" style="background:${CLASS_COLORS[k]}"></span>${man.classes[k]} cells</span><b class="mono">${n.toLocaleString()}</b>`).join('');
  $('scrub').value = meta.i;
  spark();
}

// sparkline of total ms per scan, 100 ms budget line, current frame marker
function spark() {
  const cv = $('spark'), g = cv.getContext('2d'), { dpr, W, H } = fitCanvas(cv);
  const tot = man.frames.map((f) => f.map_ms + f.label_ms), hi = Math.max(120, ...tot);
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
  st.s = Math.min(cv.clientWidth, cv.clientHeight) / (2 * (man.bands.at(-1).r_max + 3));
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
  window.addEventListener('resize', () => { st.dirty = true; spark(); });

  $('play').onclick = () => setPlaying(!st.playing);
  $('prev').onclick = () => { setPlaying(false); seek(st.i - 1); };
  $('next').onclick = () => { setPlaying(false); seek(st.i + 1); };
  $('scrub').oninput = (e) => seek(+e.target.value);
  $('speed').onchange = (e) => { st.speed = +e.target.value; };
  $('reset').onclick = resetView;
  $('fill').onchange = (e) => { st.fill = e.target.value; $('legend').innerHTML = legendHtml(st.fill, man.classes); st.dirty = true; };
  $('viewmode').onchange = (e) => {
    st.mode = e.target.value;
    st.anchor = man.frames[st.i].pose.slice(0, 2);
    st.panX = st.panY = 0;
    st.dirty = true;
  };
  for (const id of ['rings', 'path', 'car']) $(id).onchange = () => { st.dirty = true; };
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
  const base = replayBase();
  try {
    man = await loadManifest(base);
  } catch (e) {
    status(`${e.message}\n\nMake replay first, from repo root:\n  python -m scripts.export_replay\n` +
      `or point viewer at another replay: ?replay=<folder>/`);
    return;
  }
  store = new FrameStore(base, man);
  $('slice').textContent = `seq ${man.sequence} · scans ${man.scans[0]}–${man.scans[1]} · model trained on ${man.train}`;
  $('h-cpu').textContent = `Pipeline ms measured during export on ${man.timing.cpu}. ${man.timing.note}`;
  $('h-summary').textContent = `Whole replay (scans 2+): median ${man.timing.total_ms_median} ms, p95 ${man.timing.total_ms_p95} ms, ` +
    `${(man.timing.over_100ms_share * 100).toFixed(1)}% of scans over 100 ms budget.`;
  $('scrub').max = man.frames.length - 1;
  $('legend').innerHTML = legendHtml(st.fill, man.classes);
  bindInput();
  resetView();
  seek(0);
  requestAnimationFrame(tick);
}

main();
