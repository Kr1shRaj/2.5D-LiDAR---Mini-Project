// 3D chase view: map cells as boxes at true cell size, current scan points, corridor arcs, car. three.js vendored.
// world frame = manifest.frame (z up). scene origin moved to current sensor xy each frame so float32 stays precise.
import * as THREE from '../vendor/three.module.min.js';
import { FILLS, PATCH_FILL, VERDICT_COLORS, arcWorld, pointColor } from './view.js';

const MAX_CELLS = 200000;
const MAX_POINTS = 60000;
const MIN_COLUMN = 0.03;  // flat cells still get 3 cm box so they read as a surface
const GHOST_HOLD = 10;

// packed ABGR (view.js lut format) -> three colour
function unpack(c, out) {
  return out.setRGB((c & 255) / 255, ((c >> 8) & 255) / 255, ((c >> 16) & 255) / 255, THREE.SRGBColorSpace);
}

export class View3D {
  constructor(el, man, hooks) {
    this.el = el;
    this.man = man;
    this.h = hooks;  // { $, fill(), fillCtx(fr) }
    this.active = true;
    this.fr = null;
    this.orbit = { yaw: 0, pitch: 0.42, dist: 22 };  // camera behind car: yaw relative to heading, pitch above horizon
    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.renderer.setClearColor(0x0d1117);
    el.appendChild(this.renderer.domElement);
    this.scene = new THREE.Scene();
    this.scene.fog = new THREE.Fog(0x0d1117, 45, 110);
    this.camera = new THREE.PerspectiveCamera(55, 1, 0.1, 400);
    this.camera.up.set(0, 0, 1);
    this.scene.add(new THREE.HemisphereLight(0xdde6f0, 0x20242a, 1.4));
    const sun = new THREE.DirectionalLight(0xffffff, 1.6);
    sun.position.set(30, -20, 60);
    this.scene.add(sun);

    const box = new THREE.BoxGeometry(1, 1, 1);
    box.translate(0, 0, 0.5);  // unit box sits on z = 0: scale z = column height
    this.cells = new THREE.InstancedMesh(box, new THREE.MeshLambertMaterial(), MAX_CELLS);
    this.cells.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    this.cells.frustumCulled = false;
    this.scene.add(this.cells);

    this.ptGeo = new THREE.BufferGeometry();
    this.ptGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(MAX_POINTS * 3), 3));
    this.ptGeo.setAttribute('color', new THREE.BufferAttribute(new Float32Array(MAX_POINTS * 3), 3));
    this.points = new THREE.Points(this.ptGeo, new THREE.PointsMaterial({ size: 0.07, vertexColors: true }));
    this.points.frustumCulled = false;
    this.scene.add(this.points);

    this.overlay = new THREE.Group();  // rings, path, arcs, car, ghosts: rebuilt per frame (few objects)
    this.scene.add(this.overlay);
    this.bindInput();
    new ResizeObserver(() => this.resize()).observe(el);
    this.resize();
    const loop = () => { if (this.active) this.render(); requestAnimationFrame(loop); };
    requestAnimationFrame(loop);
  }

  resize() {
    const w = this.el.clientWidth || 1, h = this.el.clientHeight || 1;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  // drag orbits around car, wheel zooms, double click resets chase camera
  bindInput() {
    const cv = this.renderer.domElement;
    let down = null;
    cv.addEventListener('pointerdown', (e) => { down = [e.clientX, e.clientY]; cv.setPointerCapture(e.pointerId); });
    cv.addEventListener('pointermove', (e) => {
      if (!down) return;
      this.orbit.yaw -= (e.clientX - down[0]) * 0.006;
      this.orbit.pitch = Math.max(0.05, Math.min(1.5, this.orbit.pitch + (e.clientY - down[1]) * 0.005));
      down = [e.clientX, e.clientY];
    });
    cv.addEventListener('pointerup', () => { down = null; });
    cv.addEventListener('wheel', (e) => {
      e.preventDefault();
      this.orbit.dist = Math.max(4, Math.min(120, this.orbit.dist * Math.exp(e.deltaY * 0.0015)));
    }, { passive: false });
    cv.addEventListener('dblclick', () => this.resetView());
  }

  resetView() { this.orbit = { yaw: 0, pitch: 0.42, dist: 22 }; }

  // new decoded frame: rebuild cells + points + overlays
  show(fr) {
    this.fr = fr;
    this.redraw();
  }

  setFrame() {}

  redraw() {
    const fr = this.fr;
    if (!fr) return;
    const ctx = this.h.fillCtx(fr), man = this.man, $ = this.h.$;
    const [sx, sy, sz, yaw] = fr.meta.pose;
    this.origin = [sx, sy];
    const g0 = fr.nearGround * man.col.ground.scale;  // road level near car, world z, from data
    this.road = g0;
    this.sensorZ = sz;
    this.yaw = yaw;
    this.buildCells(fr, ctx, $('refined').checked);
    this.buildPoints(fr, $('points').checked);
    this.buildOverlay(fr, g0);
  }

  // one box per observed cell. lookup rule as 2D: base cell inside own ring; overlay patch cell drawn instead of base
  buildCells(fr, ctx, tintRefined) {
    const f = FILLS[this.h.fill()], m = new THREE.Matrix4(), col = new THREE.Color(), tint = new THREE.Color(0x39d0f0);
    const [ox, oy] = this.origin, [cx, cy] = fr.meta.center, nan = ctx.nan, sc = 0.01;
    const covered = new Set();  // base 50 cm cells hidden under observed refined patch cells (key = ix,iy of base)
    const over = fr.bands.findIndex((b) => b.overlay), baseFar = fr.bands.findIndex((b, k) => !b.overlay && b.r_min === fr.bands[over]?.r_min);
    let n = 0;
    const order = fr.bands.map((b, k) => k).sort((a, b) => fr.bands[b].overlay - fr.bands[a].overlay);  // overlay first
    for (const k of order) {
      const b = fr.bands[k], { mask, cols, ny } = b;
      let j = 0;
      for (let byte = 0; byte < mask.length; byte++) {
        const mb = mask[byte];
        if (!mb) continue;
        for (let bit = 0; bit < 8; bit++) {
          if (!((mb >> bit) & 1)) continue;
          const flat = byte * 8 + bit, ix = (flat / ny) | 0, iy = flat - ix * ny, slot = j++;
          const x = b.x_min + (ix + 0.5) * b.res, y = b.y_min + (iy + 0.5) * b.res, r = Math.hypot(x - cx, y - cy);
          if (r < b.r_min || r >= b.r_max) continue;
          if (b.overlay && baseFar >= 0) {
            const B = fr.bands[baseFar];
            covered.add(`${Math.floor((x - B.x_min) / B.res)},${Math.floor((y - B.y_min) / B.res)}`);
          } else if (k === baseFar && covered.has(`${ix},${iy}`)) continue;
          if (n >= MAX_CELLS) break;
          const zt = cols.z_max[slot], gr = cols.ground[slot];
          if (zt === nan.z_max) continue;
          const bottom = gr === nan.ground ? zt * sc - MIN_COLUMN : Math.min(gr * sc, zt * sc - MIN_COLUMN);
          const hgt = Math.max(zt * sc - bottom, MIN_COLUMN);
          m.makeScale(b.res * 0.96, b.res * 0.96, hgt);
          m.setPosition(x - ox, y - oy, bottom);
          this.cells.setMatrixAt(n, m);
          unpack(f.cell(cols, slot, ctx, f), col);
          if (b.overlay && tintRefined) col.lerp(tint, 0.55);
          this.cells.setColorAt(n, col);
          n++;
        }
      }
    }
    this.cells.count = n;
    this.cells.instanceMatrix.needsUpdate = true;
    if (this.cells.instanceColor) this.cells.instanceColor.needsUpdate = true;
  }

  // current scan points, every 4th, coloured by height over terrain
  buildPoints(fr, on) {
    const p = fr.points;
    if (!on || !p) { this.ptGeo.setDrawRange(0, 0); return; }
    const pos = this.ptGeo.attributes.position.array, rgb = this.ptGeo.attributes.color.array, c = new THREE.Color();
    const n = Math.min(p.n, MAX_POINTS), [sx, sy] = fr.meta.pose, [ox, oy] = this.origin, cache = new Map();
    for (let j = 0; j < n; j++) {
      pos[3 * j] = sx + p.dx[j] * 0.01 - ox;
      pos[3 * j + 1] = sy + p.dy[j] * 0.01 - oy;
      pos[3 * j + 2] = p.z[j] * 0.01;
      const hex = pointColor(p.h[j] === -128 ? NaN : p.h[j] * 0.02);
      if (!cache.has(hex)) cache.set(hex, c.set(hex).toArray());
      const a = cache.get(hex);
      rgb[3 * j] = a[0]; rgb[3 * j + 1] = a[1]; rgb[3 * j + 2] = a[2];
    }
    this.ptGeo.setDrawRange(0, n);
    this.ptGeo.attributes.position.needsUpdate = true;
    this.ptGeo.attributes.color.needsUpdate = true;
  }

  // rings, driven path, corridor arcs + picked footprint, car box, ghost clears
  buildOverlay(fr, g0) {
    const grp = this.overlay, man = this.man, $ = this.h.$, [ox, oy] = this.origin, z = g0 + 0.06;
    for (const o of [...grp.children]) { grp.remove(o); o.geometry?.dispose(); o.material?.dispose(); }
    const line = (pts, color, opts = {}) => {
      const geo = new THREE.BufferGeometry().setFromPoints(pts.map(([x, y, zz]) => new THREE.Vector3(x - ox, y - oy, zz ?? z)));
      const mat = opts.dashed ? new THREE.LineDashedMaterial({ color, dashSize: 0.8, gapSize: 0.6, transparent: true, opacity: opts.opacity ?? 1 })
        : new THREE.LineBasicMaterial({ color, transparent: true, opacity: opts.opacity ?? 1 });
      const l = new THREE.Line(geo, mat);
      if (opts.dashed) l.computeLineDistances();
      grp.add(l);
    };
    // flat ribbon along polyline, width w metres: reads as a lane on the ground, unlike 1 px lines
    const ribbon = (pts, w, color, opacity) => {
      const v = [];
      for (let i = 0; i < pts.length - 1; i++) {
        const [x0, y0] = pts[i], [x1, y1] = pts[i + 1], dx = x1 - x0, dy = y1 - y0, L = Math.hypot(dx, dy) || 1;
        const nx = (-dy / L) * w / 2, ny = (dx / L) * w / 2;
        const a = [x0 + nx - ox, y0 + ny - oy, z], b = [x0 - nx - ox, y0 - ny - oy, z];
        const c = [x1 + nx - ox, y1 + ny - oy, z], d = [x1 - nx - ox, y1 - ny - oy, z];
        v.push(...a, ...b, ...c, ...b, ...d, ...c);
      }
      const geo = new THREE.BufferGeometry();
      geo.setAttribute('position', new THREE.Float32BufferAttribute(v, 3));
      grp.add(new THREE.Mesh(geo, new THREE.MeshBasicMaterial({ color, transparent: true, opacity, side: THREE.DoubleSide, depthWrite: false })));
    };
    const [cx, cy] = fr.meta.center;
    if ($('rings').checked) {
      for (const b of man.bands.filter((b) => !b.overlay)) {
        const pts = [];
        for (let a = 0; a <= 128; a++) pts.push([cx + b.r_max * Math.cos((a / 128) * 2 * Math.PI), cy + b.r_max * Math.sin((a / 128) * 2 * Math.PI)]);
        line(pts, 0xe6edf3, { opacity: 0.5 });
      }
    }
    const P = man.frames.map((f) => f.pose), dz = (j) => P[j][2] - this.sensorZ;  // path follows pose height changes
    if ($('path').checked) line(P.map(([x, y], j) => [x, y, z + dz(j)]), 0x58a6ff, { opacity: 0.6 });
    const c = man.corridor, cor = fr.meta.corridor;
    if ($('corridor').checked && c && cor) {
      const drv = [];
      let along = 0;
      for (let j = fr.meta.i; j < P.length && along < c.length_m; j++) {
        if (j > fr.meta.i) along += Math.hypot(P[j][0] - P[j - 1][0], P[j][1] - P[j - 1][1]);
        drv.push([P[j][0], P[j][1], z + 0.03 + dz(j)]);
      }
      if (drv.length > 1) line(drv, 0xffffff, { dashed: true });
      c.curvatures.forEach((k, a) => {
        if (a === cor.pick) return;
        const pts = arcWorld(k, c.length_m, 0.5, fr.meta.pose), cut = Math.round(cor.b[a] / 0.5) + 1;
        line(pts.slice(0, cut), new THREE.Color(VERDICT_COLORS[cor.v[a]]).getHex(), { opacity: 0.95 });
        if (cor.v[a] === 2 && cut - 1 < pts.length) {
          const [x, y] = pts[cut - 1], s = 0.5;
          line([[x - s, y - s], [x + s, y + s]], 0xf85149);
          line([[x - s, y + s], [x + s, y - s]], 0xf85149);
        }
      });
      if (cor.pick >= 0) {
        const pts = arcWorld(c.curvatures[cor.pick], c.length_m, 0.5, fr.meta.pose);
        ribbon(pts, 2 * c.half_width_m, 0x39d0f0, 0.28);
        line(pts, 0x39d0f0);
      }
    }
    if ($('ghosts').checked) {
      const v = [];
      for (let a = 0; a < GHOST_HOLD; a++) {
        const f = man.frames[fr.meta.i - a];
        if (!f?.cleared) continue;
        for (const [dx, dy] of f.cleared) v.push(f.pose[0] + dx - ox, f.pose[1] + dy - oy, z + 0.4 * (1 - a / GHOST_HOLD));
      }
      if (v.length) {
        const geo = new THREE.BufferGeometry();
        geo.setAttribute('position', new THREE.Float32BufferAttribute(v, 3));
        grp.add(new THREE.Points(geo, new THREE.PointsMaterial({ color: 0xff4fd8, size: 0.35 })));
      }
    }
    if ($('car').checked) {
      const [x0, x1, y0, y1] = man.ego_box_velo, hgt = Math.max(this.sensorZ - this.road, 1.2);
      const geo = new THREE.BoxGeometry(x1 - x0, y1 - y0, hgt);
      const car = new THREE.Mesh(geo, new THREE.MeshLambertMaterial({ color: 0xe6edf3 }));
      const [sx, sy, , yaw] = fr.meta.pose, mx = (x0 + x1) / 2, my = (y0 + y1) / 2;
      car.position.set(sx + Math.cos(yaw) * mx - Math.sin(yaw) * my - ox, sy + Math.sin(yaw) * mx + Math.cos(yaw) * my - oy, this.road + hgt / 2);
      car.rotation.z = yaw;
      grp.add(car);
    }
  }

  // chase camera: behind car along heading (+ user orbit), looking ahead of car
  render() {
    if (!this.fr) return;
    const o = this.orbit, h = this.yaw + Math.PI + o.yaw, lz = this.road + 1.0;
    const ax = Math.cos(this.yaw) * 8, ay = Math.sin(this.yaw) * 8;  // look 8 m ahead of sensor
    this.camera.position.set(Math.cos(h) * Math.cos(o.pitch) * o.dist, Math.sin(h) * Math.cos(o.pitch) * o.dist,
      lz + Math.sin(o.pitch) * o.dist);
    this.camera.lookAt(ax, ay, lz);
    this.renderer.render(this.scene, this.camera);
  }
}
