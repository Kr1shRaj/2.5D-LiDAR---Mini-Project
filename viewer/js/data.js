// replay data: manifest, frame fetch, gunzip, decode. no drawing here.
// frame layout = src/viz/replay.py. typed arrays read little endian (every browser CPU is little endian).

export const FORMAT = 'fovreplay-1';
const DTYPES = { '<i2': Int16Array, '<u2': Uint16Array, '<i1': Int8Array, '<u1': Uint8Array };
const POP = new Uint8Array(256).map((_, b) => { let n = 0; for (let v = b; v; v >>= 1) n += v & 1; return n; });

// bytes rounded up to next multiple of 4
const pad4 = (n) => (n + 3) & ~3;

// fetch + check replay manifest
export async function loadManifest(base) {
  const r = await fetch(new URL('manifest.json', base));
  if (!r.ok) throw new Error(`no manifest at ${r.url} (HTTP ${r.status})`);
  const m = await r.json();
  if (m.format !== FORMAT) throw new Error(`manifest format ${m.format}, viewer reads ${FORMAT}`);
  m.col = Object.fromEntries(m.columns.map((c) => [c.name, c]));
  return m;
}

// gzip bytes -> ArrayBuffer via browser DecompressionStream
async function gunzip(buf) {
  const s = new Blob([buf]).stream().pipeThrough(new DecompressionStream('gzip'));
  return new Response(s).arrayBuffer();
}

// frame bytes -> bands near to far: lattice + observed mask + raw int columns + per byte rank
export function decodeFrame(buf, meta, manifest) {
  let at = 0;
  const bands = meta.bands.map((h, k) => {
    const nb = Math.ceil((h.nx * h.ny) / 8);
    const mask = new Uint8Array(buf, at, nb);
    at += pad4(nb);
    const cols = {};
    for (const c of manifest.columns) {
      const T = DTYPES[c.dtype];
      cols[c.name] = new T(buf, at, h.n_obs);
      at += pad4(h.n_obs * T.BYTES_PER_ELEMENT);
    }
    // rank[b] = set bits before byte b, so flat id -> column slot is O(1)
    const rank = new Uint32Array(nb + 1);
    for (let b = 0; b < nb; b++) rank[b + 1] = rank[b] + POP[mask[b]];
    if (rank[nb] !== h.n_obs) throw new Error(`band ${k}: mask has ${rank[nb]} cells, manifest says ${h.n_obs}`);
    const spec = manifest.bands[k];
    return { ...h, res: spec.resolution, r_min: spec.r_min, r_max: spec.r_max, mask, cols, rank };
  });
  if (at !== buf.byteLength) throw new Error(`frame ${meta.i}: ${buf.byteLength} bytes, layout says ${at}`);
  return { meta, bands, images: {} };
}

// column slot of dense flat id ix * ny + iy, -1 = cell not observed
export function slotOf(band, flat) {
  const b = flat >> 3, bit = flat & 7, m = band.mask[b];
  if (!((m >> bit) & 1)) return -1;
  return band.rank[b] + POP[m & ((1 << bit) - 1)];
}

// one column value in real units, NaN for sentinel
export function value(manifest, band, name, j) {
  const c = manifest.col[name], q = band.cols[name][j];
  return q === c.nan ? NaN : q * c.scale;
}

// fetch + decode frames on demand. keeps last `keep` decoded frames.
export class FrameStore {
  constructor(base, manifest, keep = 60) {
    this.base = base;
    this.m = manifest;
    this.keep = keep;
    this.done = new Map();
    this.pending = new Map();
  }

  // decoded frame or undefined, no fetch
  peek(i) {
    const f = this.done.get(i);
    if (f) { this.done.delete(i); this.done.set(i, f); }
    return f;
  }

  // promise of decoded frame i
  get(i) {
    const f = this.peek(i);
    if (f) return Promise.resolve(f);
    if (this.pending.has(i)) return this.pending.get(i);
    const meta = this.m.frames[i];
    const p = fetch(new URL(meta.file, this.base))
      .then((r) => { if (!r.ok) throw new Error(`frame ${i}: HTTP ${r.status}`); return r.arrayBuffer(); })
      .then(gunzip)
      .then((buf) => {
        const fr = decodeFrame(buf, meta, this.m);
        this.done.set(i, fr);
        while (this.done.size > this.keep) this.done.delete(this.done.keys().next().value);
        return fr;
      })
      .finally(() => this.pending.delete(i));
    this.pending.set(i, p);
    return p;
  }

  // warm next n frames
  prefetch(i, n) {
    for (let k = i; k < Math.min(i + n, this.m.frames.length); k++) {
      if (!this.done.has(k) && !this.pending.has(k)) this.get(k).catch(() => {});
    }
  }
}
