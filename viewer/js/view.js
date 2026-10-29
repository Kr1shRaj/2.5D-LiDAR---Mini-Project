// drawing: world <-> screen transform, colour maps, band images, overlays.
// world frame = manifest.frame (KITTI velodyne at sequence scan 0), metres, z up.

// world xy -> canvas px as affine [a, b, c, d, e, f] (canvas setTransform order):
//   u = a x + c y + e,  v = b x + d y + f
// world turned by phi, scaled by s px/m, then y flipped (world y up -> screen v down). ONLY place y flips.
export function worldToScreen({ cx, cy, phi, s, panX, panY }, W, H) {
  const co = Math.cos(phi), si = Math.sin(phi);
  const a = s * co, c = -s * si, b = -s * si, d = -s * co;
  return [a, b, c, d, W / 2 + panX - a * cx - c * cy, H / 2 + panY - b * cx - d * cy];
}

// canvas px -> world xy, inverse of worldToScreen matrix
export function screenToWorld(m, u, v) {
  const [a, b, c, d, e, f] = m, det = a * d - b * c;
  return [(d * (u - e) - c * (v - f)) / det, (-b * (u - e) + a * (v - f)) / det];
}

// hex -> [r, g, b]
const rgb = (h) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16));
// [r, g, b] -> packed ABGR for Uint32 view of ImageData (little endian)
const pack = ([r, g, b], a = 255) => ((a << 24) | (b << 16) | (g << 8) | r) >>> 0;

// 256 entry lookup table from evenly spaced hex stops
function lut(stops) {
  const c = stops.map(rgb), out = new Uint32Array(256);
  for (let i = 0; i < 256; i++) {
    const p = (i / 255) * (c.length - 1), k = Math.min(Math.floor(p), c.length - 2), t = p - k;
    out[i] = pack(c[k].map((v, j) => Math.round(v + (c[k + 1][j] - v) * t)));
  }
  return { out, stops };
}

export const CLASS_COLORS = ['#3fb950', '#f0883e', '#f85149', '#8b949e'];  // traversable, hazardous, obstacle, unknown
const CLASS_PACKED = CLASS_COLORS.map((h) => pack(rgb(h)));
const NO_VALUE = pack(rgb('#30363d'));  // observed cell but this layer has no value (NaN)
const INHERITED = pack(rgb('#d2a8ff'));
const VIRIDIS = lut(['#440154', '#482878', '#3e4989', '#31688e', '#26828e', '#1f9e89', '#35b779', '#6ece58', '#b5de2b', '#fde725']);
const DIVERGE = lut(['#2166ac', '#67a9cf', '#d1e5f0', '#f7f7f7', '#fddbc7', '#ef8a62', '#b2182b']);
const MAGMA = lut(['#2c115f', '#721f81', '#b73779', '#f1605d', '#feb078', '#fcfdbf']);

// fill modes: colour of one observed cell from raw int columns. ranges in cm (heights) or raw weight units.
export const FILLS = {
  hazard: {
    title: 'Candidate hazard labels',
    cell: (cols, j) => CLASS_PACKED[cols.label[j]] ?? NO_VALUE,
  },
  height: {
    title: 'Height of z max over local terrain', lo: -20, hi: 150, unit: 'cm', ramp: VIRIDIS, mark: 30,
    cell: (cols, j, ctx, f) => {
      const z = cols.z_max[j], t = cols.terrain[j];
      return z === ctx.nan.z_max || t === ctx.nan.terrain ? NO_VALUE : f.ramp.out[ramp(z - t, f)];
    },
  },
  ground: {
    title: 'Ground elevation vs road near car (0 = median ground within 10 m)', lo: -100, hi: 100, unit: 'cm',
    ramp: DIVERGE,
    cell: (cols, j, ctx, f) => {
      const g = cols.ground[j];
      return g === ctx.nan.ground ? NO_VALUE : f.ramp.out[ramp(g - ctx.nearGround, f)];
    },
  },
  confidence: {
    title: 'Decayed confidence weight (purple = inherited from coarse ring)', lo: 0, hi: 3000, unit: 'weight', scale: 0.01,
    ramp: MAGMA,
    cell: (cols, j, ctx, f) => (cols.flags[j] & ctx.flagInherited ? INHERITED : f.ramp.out[ramp(cols.weight[j], f)]),
  },
};

// value -> lut index, clamped
const ramp = (v, f) => Math.max(0, Math.min(255, Math.round(((v - f.lo) / (f.hi - f.lo)) * 255)));

// canvas of one band, pixel (ix, iy) = lattice cell, unobserved = transparent. cached per frame + fill.
export function bandImage(band, fill, ctx) {
  const cv = document.createElement('canvas');
  cv.width = band.nx;
  cv.height = band.ny;
  const g = cv.getContext('2d');
  const img = g.createImageData(band.nx, band.ny);
  const px = new Uint32Array(img.data.buffer);
  const f = FILLS[fill], { mask, cols, ny, nx } = band;
  let j = 0;
  for (let b = 0; b < mask.length; b++) {
    const m = mask[b];
    if (!m) continue;
    for (let bit = 0; bit < 8; bit++) {
      if (!((m >> bit) & 1)) continue;
      const flat = b * 8 + bit, ix = (flat / ny) | 0, iy = flat - ix * ny;
      // image row = iy: drawn under y-flip transform, row 0 lands at y_min (bottom)
      px[iy * nx + ix] = f.cell(cols, j++, ctx, f);
    }
  }
  g.putImageData(img, 0, 0);
  return cv;
}

// legend html for fill mode
export function legendHtml(fill, classes) {
  const f = FILLS[fill];
  if (fill === 'hazard') {
    return `<b>${f.title}</b>` + classes.map((c, k) =>
      `<div class="row"><span class="chip" style="background:${CLASS_COLORS[k]}"></span>${c}</div>`).join('') +
      `<div class="row"><span class="chip" style="border:1px solid #30363d"></span>unobserved (no data)</div>`;
  }
  const sc = f.scale ?? 1, fmt = (v) => `${+(v * sc).toFixed(2)}`;
  const mark = f.mark !== undefined ? `<div class="small muted">obstacle threshold ${f.mark} cm = ${
    Math.round(((f.mark - f.lo) / (f.hi - f.lo)) * 100)}% along bar</div>` : '';
  return `<b>${f.title}</b><div class="ramp" style="background:linear-gradient(90deg,${f.ramp.stops.join(',')})"></div>` +
    `<div class="ticks small"><span>≤ ${fmt(f.lo)}</span><span>${fmt((f.lo + f.hi) / 2)}</span><span>≥ ${fmt(f.hi)} ${f.unit}</span></div>` +
    mark + `<div class="row small"><span class="chip" style="background:#30363d"></span>observed, no value for this layer</div>`;
}

// median ground (raw cm) of inner ring: road level near car, read from data, no fixed sensor height
export function nearGround(fr, nan) {
  const g = fr.bands[0].cols.ground.filter((v) => v !== nan).sort();
  return g.length ? g[g.length >> 1] : 0;
}
