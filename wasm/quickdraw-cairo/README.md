# Quick Draw Cairo WASM rasterizer

This directory contains the small C bridge used to render the app's hand-drawn
vector strokes with the real Cairo image backend compiled to WebAssembly.

## Why this exists

The Quick, Draw! training bitmaps were not made by shrinking a browser canvas.
Google's published preprocessing uses simplified 256x256 vector drawings and a
Cairo 28x28 rasterizer. The app records pointer input as strokes, follows the
published simplification steps, and uses this bridge for the final raster step.

Published vector simplification:

1. translate the drawing so its minimum x/y are at the top-left origin;
2. uniformly scale it so the largest extent reaches 255;
3. resample strokes at 1-pixel spacing;
4. apply Ramer-Douglas-Peucker simplification with epsilon 2.0.

Published Cairo raster settings reproduced by `quickdraw_cairo_module.c`:

- image surface: ARGB32, 28x28;
- antialias: `CAIRO_ANTIALIAS_BEST`;
- line cap/join: round;
- line diameter: 16 in the original 256x256 coordinate space;
- padding: 16;
- black background, white stroke;
- center using the simplified drawing bounding-box maximum before rasterizing.

The public Quick Draw raster function can be found in the discussion at:
https://github.com/googlecreativelab/quickdraw-dataset/issues/19

The dataset simplification description is documented at:
https://github.com/googlecreativelab/quickdraw-dataset

## Reproducible WASM build

`.github/workflows/build-quickdraw-cairo-wasm.yml` builds and vendors:

- pixman 0.46.4;
- Cairo 1.18.4;
- Emscripten 4.0.10;
- this C bridge.

Only the single-threaded Cairo image backend is built. Fontconfig, FreeType,
HarfBuzz, PNG, PDF/PS/SVG, X11/XCB and other unrelated backends are disabled.
The generated files are:

- `vendor/quickdraw-cairo.js`
- `vendor/quickdraw-cairo.wasm`

The workflow also instantiates the generated module in Node and renders a test
stroke before committing the generated runtime.

## Third-party licenses

Cairo is available under either LGPL-2.1 or MPL-1.1. This project uses the
MPL-1.1 option for the distributed Cairo build. See Cairo's upstream source and
license files for the complete terms:

- https://gitlab.freedesktop.org/cairo/cairo
- https://cairographics.org/

pixman is distributed under an MIT-style license. Upstream source:

- https://gitlab.freedesktop.org/pixman/pixman

The generated WASM is built from the exact upstream release versions listed
above by the checked-in GitHub Actions workflow, so the corresponding source and
build instructions remain available alongside the binary.
