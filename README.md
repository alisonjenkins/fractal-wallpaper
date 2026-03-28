# fractal-wallpaper

A fast, SIMD-accelerated fractal wallpaper generator written in Rust. Produces unique, visually rich wallpapers at any resolution with 9 fractal types, 12+ color palettes, and a color-theory random palette generator.

Default resolution is 5440x1440 (ultra-wide dual monitor), but any resolution is supported via `--width` and `--height`.

## Quick Start

```bash
# Build (requires Rust toolchain)
cargo build --release

# Generate a random Mandelbrot wallpaper
./target/release/fractal-wallpaper mandelbrot

# Generate all 9 fractal types at once
./target/release/fractal-wallpaper all

# Pick a specific palette and fractal
./target/release/fractal-wallpaper flame -p sakura -o my_wallpaper.png
```

## Fractal Types

| Type | Flag | Description |
|------|------|-------------|
| Mandelbrot | `mandelbrot` | Classic escape-time fractal with SIMD acceleration and cardioid skip |
| Julia | `julia` | Julia set with randomized constants from the Mandelbrot boundary |
| Burning Ship | `burning-ship` | Absolute-value variant producing gothic architectural structures |
| Newton | `newton` | Newton's method on z^3-1, showing root basin boundaries |
| Tricorn | `tricorn` | Conjugate Mandelbrot (Mandelbar) with 3-fold symmetry |
| Phoenix | `phoenix` | Julia variant with memory — feathery organic structures |
| Flame | `flame` | IFS with 10 nonlinear variations — wispy, fairy-like tendrils |
| Buddhabrot | `buddhabrot` | Escaping Mandelbrot orbit traces — ghostly nebula imagery |
| Strange Attractor | `strange-attractor` | Clifford/De Jong attractors — flowing silk-thread patterns |

Generate all types at once with `all`:

```bash
./target/release/fractal-wallpaper all
```

## Palettes

### Preset Palettes

| Palette | Flag | Description |
|---------|------|-------------|
| Twilight | `twilight` | Deep purples through warm peach to lavender |
| Ocean | `ocean` | Dark navy through cyan to white |
| Fire | `fire` | Black through red-orange to yellow-white |
| Neon | `neon` | Vivid cyan, green, yellow, magenta on black |
| Frost | `frost` | Icy white-blue gradient to deep navy |
| Earth | `earth` | Brown-green natural tones |
| Sakura | `sakura` | Deep rose through bubblegum pink to white |
| Catppuccin Mocha | `catppuccin-mocha` | Catppuccin dark theme accent colors |
| Catppuccin Macchiato | `catppuccin-macchiato` | Catppuccin medium-dark accent colors |
| Catppuccin Frappe | `catppuccin-frappe` | Catppuccin medium accent colors |
| Catppuccin Latte | `catppuccin-latte` | Catppuccin light theme accent colors |
| Random | `random` | Color-theory generated palette (unique each run) |

```bash
# Use a specific palette
./target/release/fractal-wallpaper mandelbrot -p catppuccin-mocha

# Random palette (uses complementary, analogous, triadic, split-complementary, or tetradic harmony)
./target/release/fractal-wallpaper flame -p random
```

### Custom Palette Files

Load your own palette from a JSON file with `--palette-file`. The file should contain a JSON array of `[R, G, B]` arrays (2-12 color anchors, values 0-255):

```json
[
  [10, 0, 30],
  [80, 0, 120],
  [200, 50, 180],
  [255, 200, 230],
  [255, 255, 255],
  [200, 50, 180],
  [80, 0, 120],
  [10, 0, 30]
]
```

```bash
./target/release/fractal-wallpaper mandelbrot --palette-file my_palette.json
```

The output filename will use the palette file's name (e.g. `fractal_mandelbrot_my_palette_5440x1440.png`).

## CLI Reference

```
fractal-wallpaper [OPTIONS] [FRACTAL]
```

| Flag | Description | Default |
|------|-------------|---------|
| `[FRACTAL]` | Fractal type (see table above), or `all` | `mandelbrot` |
| `-p, --palette` | Color palette preset | Random from all presets |
| `-o, --output` | Output file path | Auto-generated from type/palette/resolution |
| `-s, --seed` | Random seed for reproducibility | System time |
| `-m, --max-iter` | Maximum iterations for escape-time fractals | `1500` |
| `--width` | Image width in pixels | `5440` |
| `--height` | Image height in pixels | `1440` |
| `--samples` | Sample count for histogram fractals (flame/buddhabrot/attractor) | Type-dependent |
| `--supersample` | Anti-aliasing factor (2 = render at 2x, downsample) | `1` |
| `--palette-file` | Load custom palette from JSON file | — |
| `--save-params` | Save fractal parameters to JSON for reproduction | — |
| `--load-params` | Load parameters from a previously saved JSON file | — |

## Examples

```bash
# 1080p Mandelbrot with sakura palette
./target/release/fractal-wallpaper mandelbrot -p sakura --width 1920 --height 1080

# 4K flame fractal with anti-aliasing
./target/release/fractal-wallpaper flame --width 3840 --height 2160 --supersample 2

# High-quality Buddhabrot with 200M samples
./target/release/fractal-wallpaper buddhabrot --samples 200000000 -p catppuccin-mocha

# Save parameters for a fractal you like
./target/release/fractal-wallpaper julia -p neon --save-params favorite.json -o favorite.png

# Reproduce it later at a different resolution
./target/release/fractal-wallpaper --load-params favorite.json --width 3840 --height 2160

# Reproducible output with a fixed seed
./target/release/fractal-wallpaper mandelbrot --seed 42 -p twilight
```

## How It Works

Each run automatically finds visually interesting parameters:

- **Escape-time fractals** (Mandelbrot, Julia, Burning Ship, Tricorn, Phoenix): Binary search finds the fractal boundary, then viewport scoring evaluates zoom levels for visual richness — checking that detail spans the full width with good inside/outside ratio.
- **Flame fractals**: Random IFS transform sets are probed at low resolution and scored for fill rate, density variance, and spatial spread.
- **Buddhabrot**: Metropolis-Hastings sampling concentrates orbit traces near the Mandelbrot boundary where escaping orbits produce the most detail.
- **Strange attractors**: Random Clifford/De Jong parameters are scored for coverage, spread, and wide aspect ratio to fill ultra-wide frames.

## Performance

- **SIMD**: Mandelbrot, Julia, and Burning Ship use `f64x4` (AVX2) to process 4 pixels simultaneously with mask-based iteration counting
- **Parallelism**: All fractal types use Rayon for multi-core parallelism (per-row for escape-time, per-batch for histogram)
- **Cardioid skip**: Mandelbrot skips iteration for points inside the main cardioid and period-2 bulb
- **Metropolis-Hastings**: Buddhabrot focuses sampling on the boundary (~4x faster than uniform random)
- **Native CPU**: Builds with `-C target-cpu=native` for AVX2 auto-vectorization

Typical generation times at 5440x1440 on a modern CPU:

| Type | Time |
|------|------|
| Mandelbrot | ~0.25s |
| Julia | ~0.09s |
| Burning Ship | ~0.17s |
| Newton | ~0.02s |
| Flame (100M samples) | ~0.8s |
| Buddhabrot (50M samples) | ~2.5s |
| Strange Attractor (20M samples) | ~0.2s |
