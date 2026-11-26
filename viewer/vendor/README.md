# Vendored JS

Third-party files copied in whole at one pinned version. No CDN at runtime, no npm, no build step.
Never edit by hand; to upgrade, replace the whole file and update this table.

| File | Package | Version | Source | sha256 | Licence |
|---|---|---|---|---|---|
| `three.module.min.js` | three | 0.160.0 | npm tarball `three-0.160.0.tgz`, `build/three.module.min.js` | `3e690ac7d180b0aadf0891bea39eec643e29e2d3e75c99b18689518665f69ba6` | MIT, `three.LICENSE` |

Used only by `viewer/js/view3d.js` (3D chase tab), loaded on first use of that tab.
