// Quick, Draw! 28x28 rasterizer backed by the real Cairo image backend.
//
// The rendering constants and transform reproduce the public function used to
// generate the official Quick Draw numpy bitmaps:
//   CAIRO_ANTIALIAS_BEST, ROUND caps/joins, line diameter 16, padding 16,
//   original 256x256 coordinate space, and a centered bounding box.
//
// JavaScript performs the documented vector simplification first (top-left
// alignment, uniform 0..255 scaling, 1px resampling, RDP epsilon=2). This file
// only owns the Cairo rasterization step so the model sees the same kind of
// grayscale pixels as the training bitmaps.

#include <cairo.h>
#include <emscripten/emscripten.h>
#include <stdint.h>
#include <stddef.h>

#define QD_SIDE 28
#define QD_RASTER_LEN (QD_SIDE * QD_SIDE)
#define QD_MAX_STROKES 256
#define QD_MAX_POINTS 16384

typedef struct {
  double x;
  double y;
} qd_point_t;

static qd_point_t qd_points[QD_MAX_POINTS];
static int qd_stroke_start[QD_MAX_STROKES];
static int qd_stroke_length[QD_MAX_STROKES];
static int qd_stroke_count = 0;
static int qd_point_count = 0;
static uint8_t qd_raster[QD_RASTER_LEN];
static int qd_last_status = 0;

EMSCRIPTEN_KEEPALIVE
void qd_reset(void) {
  qd_stroke_count = 0;
  qd_point_count = 0;
  qd_last_status = 0;
}

EMSCRIPTEN_KEEPALIVE
int qd_begin_stroke(void) {
  if (qd_stroke_count >= QD_MAX_STROKES) {
    qd_last_status = -1;
    return -1;
  }
  qd_stroke_start[qd_stroke_count] = qd_point_count;
  qd_stroke_length[qd_stroke_count] = 0;
  qd_stroke_count++;
  return qd_stroke_count - 1;
}

EMSCRIPTEN_KEEPALIVE
int qd_add_point(double x, double y) {
  if (qd_stroke_count <= 0) {
    qd_last_status = -2;
    return -2;
  }
  if (qd_point_count >= QD_MAX_POINTS) {
    qd_last_status = -3;
    return -3;
  }
  qd_points[qd_point_count].x = x;
  qd_points[qd_point_count].y = y;
  qd_point_count++;
  qd_stroke_length[qd_stroke_count - 1]++;
  return 0;
}

EMSCRIPTEN_KEEPALIVE
int qd_render(void) {
  if (qd_stroke_count <= 0 || qd_point_count <= 0) {
    qd_last_status = -4;
    return -4;
  }

  double max_x = 0.0;
  double max_y = 0.0;
  for (int i = 0; i < qd_point_count; i++) {
    if (qd_points[i].x > max_x) max_x = qd_points[i].x;
    if (qd_points[i].y > max_y) max_y = qd_points[i].y;
  }

  cairo_surface_t *surface = cairo_image_surface_create(CAIRO_FORMAT_ARGB32, QD_SIDE, QD_SIDE);
  if (cairo_surface_status(surface) != CAIRO_STATUS_SUCCESS) {
    qd_last_status = -5;
    cairo_surface_destroy(surface);
    return -5;
  }

  cairo_t *ctx = cairo_create(surface);
  if (cairo_status(ctx) != CAIRO_STATUS_SUCCESS) {
    qd_last_status = -6;
    cairo_destroy(ctx);
    cairo_surface_destroy(surface);
    return -6;
  }

  // These values intentionally match the published Quick Draw rasterizer.
  const double original_side = 256.0;
  const double line_diameter = 16.0;
  const double padding = 16.0;
  const double total_padding = padding * 2.0 + line_diameter;
  const double new_scale = (double)QD_SIDE / (original_side + total_padding);

  cairo_set_antialias(ctx, CAIRO_ANTIALIAS_BEST);
  cairo_set_line_cap(ctx, CAIRO_LINE_CAP_ROUND);
  cairo_set_line_join(ctx, CAIRO_LINE_JOIN_ROUND);
  cairo_set_line_width(ctx, line_diameter);

  cairo_scale(ctx, new_scale, new_scale);
  cairo_translate(ctx, total_padding / 2.0, total_padding / 2.0);

  // Official bitmaps are white strokes on a black background.
  cairo_set_source_rgb(ctx, 0.0, 0.0, 0.0);
  cairo_paint(ctx);

  // Simplified vectors are top-left aligned. The bitmap generator recenters
  // them using their maximum x/y extent inside the original 256x256 space.
  const double offset_x = (original_side - max_x) / 2.0;
  const double offset_y = (original_side - max_y) / 2.0;

  cairo_set_source_rgb(ctx, 1.0, 1.0, 1.0);
  for (int s = 0; s < qd_stroke_count; s++) {
    const int start = qd_stroke_start[s];
    const int length = qd_stroke_length[s];
    if (length <= 0) continue;

    const qd_point_t first = qd_points[start];
    cairo_move_to(ctx, first.x + offset_x, first.y + offset_y);

    // The published Python function includes the first point in the line_to
    // loop as well, so retain that exact path construction here.
    for (int i = 0; i < length; i++) {
      const qd_point_t p = qd_points[start + i];
      cairo_line_to(ctx, p.x + offset_x, p.y + offset_y);
    }
    cairo_stroke(ctx);
  }

  cairo_surface_flush(surface);
  unsigned char *pixels = cairo_image_surface_get_data(surface);
  const int stride = cairo_image_surface_get_stride(surface);
  if (!pixels) {
    qd_last_status = -7;
    cairo_destroy(ctx);
    cairo_surface_destroy(surface);
    return -7;
  }

  // WASM is little-endian; CAIRO_FORMAT_ARGB32 therefore stores BGRA bytes.
  // The image is grayscale, so B == G == R and the first byte is sufficient.
  for (int y = 0; y < QD_SIDE; y++) {
    for (int x = 0; x < QD_SIDE; x++) {
      qd_raster[y * QD_SIDE + x] = pixels[y * stride + x * 4];
    }
  }

  cairo_destroy(ctx);
  cairo_surface_destroy(surface);
  qd_last_status = 0;
  return 0;
}

EMSCRIPTEN_KEEPALIVE
uintptr_t qd_raster_ptr(void) {
  return (uintptr_t)qd_raster;
}

EMSCRIPTEN_KEEPALIVE
int qd_raster_length(void) {
  return QD_RASTER_LEN;
}

EMSCRIPTEN_KEEPALIVE
int qd_get_last_status(void) {
  return qd_last_status;
}

EMSCRIPTEN_KEEPALIVE
int qd_get_point_count(void) {
  return qd_point_count;
}

EMSCRIPTEN_KEEPALIVE
const char *qd_cairo_version(void) {
  return cairo_version_string();
}
