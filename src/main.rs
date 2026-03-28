use clap::{Parser, ValueEnum};
use image::{Rgb, RgbImage};
use rayon::prelude::*;
use std::path::PathBuf;
use std::time::Instant;

const WIDTH: u32 = 5440;
const HEIGHT: u32 = 1440;

// ── Simple PRNG (xoshiro256**) ──────────────────────────────────────────────

struct Rng {
    s: [u64; 4],
}

impl Rng {
    fn new(seed: u64) -> Self {
        let mut z = seed;
        let mut s = [0u64; 4];
        for slot in &mut s {
            z = z.wrapping_add(0x9e3779b97f4a7c15);
            let mut x = z;
            x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
            *slot = x ^ (x >> 31);
        }
        Self { s }
    }

    fn next_u64(&mut self) -> u64 {
        let result = (self.s[1].wrapping_mul(5)).rotate_left(7).wrapping_mul(9);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }

    fn f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + self.f64() * (hi - lo)
    }

    fn choose<T: Copy>(&mut self, items: &[T]) -> T {
        let idx = (self.next_u64() as usize) % items.len();
        items[idx]
    }
}

// ── CLI ─────────────────────────────────────────────────────────────────────

#[derive(Parser)]
#[command(name = "fractal-wallpaper", about = "Fractal wallpaper generator (5440×1440)")]
struct Cli {
    /// Fractal type to generate
    #[arg(default_value = "mandelbrot")]
    fractal: FractalType,

    /// Color palette (random if not specified)
    #[arg(short, long)]
    palette: Option<Palette>,

    /// Maximum iterations
    #[arg(short, long, default_value_t = 1500)]
    max_iter: u32,

    /// Output file path
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// Random seed (uses system time if not specified)
    #[arg(short, long)]
    seed: Option<u64>,
}

#[derive(Clone, Copy, ValueEnum, PartialEq)]
enum FractalType {
    Mandelbrot,
    Julia,
    BurningShip,
    Newton,
    All,
}

#[derive(Clone, Copy, ValueEnum)]
enum Palette {
    Twilight,
    Ocean,
    Fire,
    Neon,
    Frost,
    Earth,
}

const ALL_PALETTES: [Palette; 6] = [
    Palette::Twilight, Palette::Ocean, Palette::Fire,
    Palette::Neon, Palette::Frost, Palette::Earth,
];

impl std::fmt::Display for FractalType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Mandelbrot => write!(f, "mandelbrot"),
            Self::Julia => write!(f, "julia"),
            Self::BurningShip => write!(f, "burning-ship"),
            Self::Newton => write!(f, "newton"),
            Self::All => write!(f, "all"),
        }
    }
}

impl std::fmt::Display for Palette {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Twilight => write!(f, "twilight"),
            Self::Ocean => write!(f, "ocean"),
            Self::Fire => write!(f, "fire"),
            Self::Neon => write!(f, "neon"),
            Self::Frost => write!(f, "frost"),
            Self::Earth => write!(f, "earth"),
        }
    }
}

// ── Color Palettes ──────────────────────────────────────────────────────────

fn palette_anchors(palette: Palette) -> Vec<[u8; 3]> {
    match palette {
        Palette::Twilight => vec![
            [10, 2, 30], [40, 5, 80], [90, 20, 140], [160, 50, 180],
            [220, 100, 160], [255, 160, 120], [255, 220, 180], [255, 255, 240],
            [200, 180, 255], [120, 100, 200], [60, 40, 140], [20, 10, 60],
        ],
        Palette::Ocean => vec![
            [2, 5, 20], [5, 20, 60], [10, 50, 120], [20, 100, 160],
            [40, 160, 190], [100, 210, 220], [180, 240, 240], [240, 255, 255],
            [100, 200, 230], [30, 120, 180], [10, 60, 130], [2, 20, 60],
        ],
        Palette::Fire => vec![
            [10, 0, 0], [40, 5, 0], [100, 15, 0], [180, 40, 0],
            [240, 80, 0], [255, 140, 20], [255, 200, 60], [255, 255, 160],
            [255, 220, 80], [220, 120, 10], [150, 50, 0], [60, 10, 0],
        ],
        Palette::Neon => vec![
            [0, 0, 0], [20, 0, 40], [60, 0, 100], [0, 80, 180],
            [0, 200, 200], [0, 255, 150], [150, 255, 0], [255, 255, 0],
            [255, 100, 200], [200, 0, 255], [100, 0, 200], [30, 0, 60],
        ],
        Palette::Frost => vec![
            [240, 248, 255], [200, 220, 240], [160, 200, 235], [120, 180, 230],
            [80, 150, 220], [50, 120, 210], [30, 80, 180], [15, 50, 140],
            [5, 25, 80], [2, 10, 40], [20, 40, 90], [80, 120, 180],
        ],
        Palette::Earth => vec![
            [15, 10, 5], [40, 25, 10], [80, 50, 20], [130, 80, 30],
            [180, 120, 50], [210, 170, 80], [230, 210, 140], [245, 240, 200],
            [180, 200, 130], [100, 150, 80], [50, 100, 50], [20, 50, 20],
        ],
    }
}

fn build_colormap(palette: Palette, n: usize) -> Vec<[u8; 3]> {
    let anchors = palette_anchors(palette);
    let num = anchors.len();
    let xs: Vec<f64> = (0..num).map(|i| i as f64 / (num - 1) as f64).collect();
    (0..n)
        .map(|i| {
            let t = i as f64 / (n - 1) as f64;
            let mut seg = 0;
            for j in 0..num - 1 {
                if t >= xs[j] && t <= xs[j + 1] {
                    seg = j;
                    break;
                }
            }
            let local_t = if xs[seg + 1] - xs[seg] > 0.0 {
                (t - xs[seg]) / (xs[seg + 1] - xs[seg])
            } else {
                0.0
            };
            let a = &anchors[seg];
            let b = &anchors[seg + 1];
            [
                (a[0] as f64 + (b[0] as f64 - a[0] as f64) * local_t) as u8,
                (a[1] as f64 + (b[1] as f64 - a[1] as f64) * local_t) as u8,
                (a[2] as f64 + (b[2] as f64 - a[2] as f64) * local_t) as u8,
            ]
        })
        .collect()
}

// ── Single-point iteration functions (for probing) ──────────────────────────

fn iterate_mandelbrot(cr: f64, ci: f64, max_iter: u32) -> f64 {
    let mut zr = 0.0;
    let mut zi = 0.0;
    let mut i = 0u32;
    while i < max_iter {
        let zr2 = zr * zr;
        let zi2 = zi * zi;
        if zr2 + zi2 > 65536.0 {
            break;
        }
        zi = 2.0 * zr * zi + ci;
        zr = zr2 - zi2 + cr;
        i += 1;
    }
    if i < max_iter {
        let mag = (zr * zr + zi * zi).sqrt();
        i as f64 + 1.0 - mag.ln().ln() / std::f64::consts::LN_2
    } else {
        0.0
    }
}

fn iterate_julia(zr0: f64, zi0: f64, cr: f64, ci: f64, max_iter: u32) -> f64 {
    let mut zr = zr0;
    let mut zi = zi0;
    let mut i = 0u32;
    while i < max_iter {
        let zr2 = zr * zr;
        let zi2 = zi * zi;
        if zr2 + zi2 > 65536.0 {
            break;
        }
        let new_zr = zr2 - zi2 + cr;
        zi = 2.0 * zr * zi + ci;
        zr = new_zr;
        i += 1;
    }
    if i < max_iter {
        let mag = (zr * zr + zi * zi).sqrt();
        i as f64 + 1.0 - mag.ln().ln() / std::f64::consts::LN_2
    } else {
        0.0
    }
}

fn iterate_burning_ship(cr: f64, ci: f64, max_iter: u32) -> f64 {
    let mut zr: f64 = 0.0;
    let mut zi: f64 = 0.0;
    let mut i = 0u32;
    while i < max_iter {
        let azr = zr.abs();
        let azi = zi.abs();
        let zr2 = azr * azr;
        let zi2 = azi * azi;
        if zr2 + zi2 > 65536.0 {
            break;
        }
        zi = 2.0 * azr * azi + ci;
        zr = zr2 - zi2 + cr;
        i += 1;
    }
    if i < max_iter {
        let mag = (zr * zr + zi * zi).sqrt();
        i as f64 + 1.0 - mag.ln().ln() / std::f64::consts::LN_2
    } else {
        0.0
    }
}

// ── Boundary-finding: locate interesting points ─────────────────────────────

/// Find a point on the boundary of the Mandelbrot set by binary search.
/// Start from a point known to be outside, search toward a point inside.
fn find_boundary_point(
    outside: (f64, f64),
    inside: (f64, f64),
    max_iter: u32,
) -> (f64, f64) {
    let mut or = outside.0;
    let mut oi = outside.1;
    let mut ir = inside.0;
    let mut ii = inside.1;

    for _ in 0..64 {
        let mr = (or + ir) / 2.0;
        let mi = (oi + ii) / 2.0;
        let val = iterate_mandelbrot(mr, mi, max_iter);
        if val == 0.0 {
            // Inside set — move inside point to midpoint
            ir = mr;
            ii = mi;
        } else {
            // Outside set — move outside point to midpoint
            or = mr;
            oi = mi;
        }
    }
    ((or + ir) / 2.0, (oi + ii) / 2.0)
}

/// Probe a viewport with spatial awareness.
/// Returns a score from 0.0 (boring) to 1.0 (visually rich).
/// Checks that detail is spread across the full width, not clustered.
fn score_viewport(
    center: (f64, f64),
    zoom: f64,
    max_iter: u32,
    iterate_fn: &dyn Fn(f64, f64, u32) -> f64,
) -> f64 {
    let aspect = WIDTH as f64 / HEIGHT as f64;
    let x_range = 3.5 / zoom;
    let y_range = x_range / aspect;
    let x_min = center.0 - x_range / 2.0;
    let y_min = center.1 - y_range / 2.0;

    let sx = 48; // More horizontal samples for ultra-wide
    let sy = 16;
    let mut values = vec![0.0f64; sx * sy];

    for row in 0..sy {
        for col in 0..sx {
            let x = x_min + x_range * (col as f64 + 0.5) / sx as f64;
            let y = y_min + y_range * (row as f64 + 0.5) / sy as f64;
            values[row * sx + col] = iterate_fn(x, y, max_iter);
        }
    }

    let total = values.len() as f64;
    let inside_count = values.iter().filter(|&&v| v == 0.0).count() as f64;
    let inside_frac = inside_count / total;

    // ── Check 1: inside fraction (5-35% is ideal) ───────────────────────
    let inside_score = if inside_frac < 0.01 || inside_frac > 0.60 {
        0.05  // Almost all black or all exterior — very boring
    } else if inside_frac >= 0.05 && inside_frac <= 0.35 {
        1.0
    } else {
        0.4
    };

    // ── Check 2: detail spread across the width ─────────────────────────
    // Split the viewport into 4 horizontal quarters.
    // Each quarter must have a mix of inside/outside to score well.
    let quarter_w = sx / 4;
    let mut quarters_active = 0u32;
    for q in 0..4 {
        let mut q_inside = 0;
        let mut q_total = 0;
        for row in 0..sy {
            for col in (q * quarter_w)..((q + 1) * quarter_w) {
                q_total += 1;
                if values[row * sx + col] == 0.0 {
                    q_inside += 1;
                }
            }
        }
        let q_frac = q_inside as f64 / q_total as f64;
        // Quarter is "active" if it has a meaningful mix (not >90% one thing)
        if q_frac > 0.05 && q_frac < 0.90 {
            quarters_active += 1;
        }
    }
    // Want at least 3 of 4 quarters to have detail
    let spread_score = match quarters_active {
        4 => 1.0,
        3 => 0.7,
        2 => 0.3,
        _ => 0.05,
    };

    // ── Check 3: iteration diversity (color richness) ───────────────────
    let escaped: Vec<f64> = values.iter().copied().filter(|&v| v > 0.0).collect();
    let diversity_score = if escaped.is_empty() {
        0.0
    } else {
        let mean = escaped.iter().sum::<f64>() / escaped.len() as f64;
        let variance = escaped.iter()
            .map(|&v| (v - mean) * (v - mean))
            .sum::<f64>() / escaped.len() as f64;
        let std_dev = variance.sqrt();
        // Normalize: std_dev of ~10% of max_iter is excellent
        (std_dev / max_iter as f64 * 10.0).min(1.0)
    };

    // ── Check 4: mean iteration depth ───────────────────────────────────
    let depth_score = if escaped.is_empty() {
        0.0
    } else {
        let mean = escaped.iter().sum::<f64>() / escaped.len() as f64;
        (mean / max_iter as f64 * 5.0).min(1.0)
    };

    inside_score * 0.25 + spread_score * 0.35 + diversity_score * 0.25 + depth_score * 0.15
}

// ── Parameter Generation with Quality Search ────────────────────────────────

struct FractalParams {
    center: (f64, f64),
    zoom: f64,
    julia_c: Option<(f64, f64)>,
    color_offset: f64,
}

fn find_interesting_mandelbrot(rng: &mut Rng, max_iter: u32) -> FractalParams {
    let mut best_params = FractalParams {
        center: (-0.75, 0.0),
        zoom: 1.0,
        julia_c: None,
        color_offset: rng.f64(),
    };
    let mut best_score = 0.0f64;

    for _ in 0..80 {
        // Strategy: pick a random angle, shoot a ray from origin,
        // find where it crosses the set boundary, then zoom in there.
        let angle = rng.range(0.0, std::f64::consts::TAU);
        let ray_len = 2.5;
        let outside = (ray_len * angle.cos(), ray_len * angle.sin());
        let inside_pt = (-0.1, 0.0);

        let boundary = find_boundary_point(outside, inside_pt, max_iter);

        // Try various zoom levels at this boundary point
        let zoom = 10.0f64.powf(rng.range(1.5, 4.0)); // ~30x to ~10000x

        let score = score_viewport(
            boundary, zoom, max_iter,
            &|x, y, mi| iterate_mandelbrot(x, y, mi),
        );

        if score > best_score {
            best_score = score;
            best_params = FractalParams {
                center: boundary,
                zoom,
                julia_c: None,
                color_offset: rng.f64(),
            };
        }

        if best_score > 0.80 {
            break;
        }
    }

    best_params
}

fn find_interesting_julia(rng: &mut Rng, max_iter: u32) -> FractalParams {
    let mut best_params = FractalParams {
        center: (0.0, 0.0),
        zoom: 1.0,
        julia_c: Some((-0.7269, 0.1889)),
        color_offset: rng.f64(),
    };
    let mut best_score = 0.0f64;

    for _ in 0..50 {
        // Pick c from the Mandelbrot boundary — these produce connected,
        // detailed Julia sets
        let angle = rng.range(0.0, std::f64::consts::TAU);
        let outside = (2.0 * angle.cos(), 2.0 * angle.sin());
        let inside = (-0.1, 0.0);
        let c = find_boundary_point(outside, inside, 200);

        // Small perturbation toward outside for more detail
        let perturb = rng.range(0.001, 0.02);
        let c = (
            c.0 + perturb * (c.0 - inside.0),
            c.1 + perturb * (c.1 - inside.1),
        );

        let zoom = 10.0f64.powf(rng.range(0.0, 1.5));

        let score = score_viewport(
            (0.0, 0.0), zoom, max_iter,
            &|x, y, mi| iterate_julia(x, y, c.0, c.1, mi),
        );

        if score > best_score {
            best_score = score;
            best_params = FractalParams {
                center: (0.0, 0.0),
                zoom,
                julia_c: Some(c),
                color_offset: rng.f64(),
            };
        }

        if best_score > 0.80 {
            break;
        }
    }

    best_params
}

fn find_interesting_burning_ship(rng: &mut Rng, max_iter: u32) -> FractalParams {
    let mut best_params = FractalParams {
        center: (-1.75, -0.04),
        zoom: 30.0,
        julia_c: None,
        color_offset: rng.f64(),
    };
    let mut best_score = 0.0f64;

    // The Burning Ship's interesting features are its "masts" and "hull".
    // Curated regions known to have dramatic architectural structures,
    // plus boundary search from known inside points.
    let curated: [(f64, f64, f64, f64); 10] = [
        // (center_x, center_y, min_zoom, max_zoom)
        (-1.755, -0.028, 20.0, 200.0),    // Main mast
        (-1.762, -0.028, 50.0, 300.0),     // Upper mast detail
        (-1.78, -0.008, 10.0, 80.0),       // Hull bow
        (-1.74, -0.03, 15.0, 100.0),       // Rigging
        (-1.77, -0.01, 20.0, 150.0),       // Mast base
        (-1.76, -0.035, 30.0, 200.0),      // Between masts
        (-1.7, -0.05, 3.0, 20.0),          // Wide ship view
        (-1.755, -0.02, 40.0, 250.0),      // Mast tip
        (-1.765, -0.015, 30.0, 180.0),     // Hull detail
        (-0.515, -0.565, 20.0, 150.0),     // Satellite ship
    ];

    for _ in 0..80 {
        let (cx, cy, min_z, max_z) = rng.choose(&curated);

        // Jitter the center slightly (scaled to zoom)
        let zoom = 10.0f64.powf(rng.range(min_z.log10(), max_z.log10()));
        let jitter = 0.5 / zoom;
        let center = (
            cx + rng.range(-jitter, jitter),
            cy + rng.range(-jitter, jitter),
        );

        let score = score_viewport(
            center, zoom, max_iter,
            &|x, y, mi| iterate_burning_ship(x, y, mi),
        );

        if score > best_score {
            best_score = score;
            best_params = FractalParams {
                center,
                zoom,
                julia_c: None,
                color_offset: rng.f64(),
            };
        }

        if best_score > 0.80 {
            break;
        }
    }

    best_params
}

fn find_interesting_newton(rng: &mut Rng, _max_iter: u32) -> FractalParams {
    // Newton fractals are interesting everywhere near root boundaries.
    // Zoom into the area around the origin where all three basins meet.
    let center = (rng.range(-0.3, 0.3), rng.range(-0.3, 0.3));
    let zoom = 10.0f64.powf(rng.range(0.3, 2.0));

    FractalParams {
        center,
        zoom,
        julia_c: None,
        color_offset: rng.f64(),
    }
}

// ── Fractal Computation (full image) ────────────────────────────────────────

fn compute_mandelbrot(
    width: u32, height: u32,
    center: (f64, f64), zoom: f64, max_iter: u32,
) -> Vec<f64> {
    let w = width as usize;
    let h = height as usize;
    let aspect = width as f64 / height as f64;
    let x_range = 3.5 / zoom;
    let y_range = x_range / aspect;
    let x_min = center.0 - x_range / 2.0;
    let y_min = center.1 - y_range / 2.0;
    let x_step = x_range / width as f64;
    let y_step = y_range / height as f64;

    let mut result = vec![0.0f64; w * h];
    result
        .par_chunks_mut(w)
        .enumerate()
        .for_each(|(py, row)| {
            let ci = y_min + py as f64 * y_step;
            for px in 0..w {
                let cr = x_min + px as f64 * x_step;
                row[px] = iterate_mandelbrot(cr, ci, max_iter);
            }
        });
    result
}

fn compute_julia(
    width: u32, height: u32,
    c: (f64, f64), zoom: f64, max_iter: u32,
) -> Vec<f64> {
    let w = width as usize;
    let h = height as usize;
    let aspect = width as f64 / height as f64;
    let x_range = 3.5 / zoom;
    let y_range = x_range / aspect;
    let x_min = -x_range / 2.0;
    let y_min = -y_range / 2.0;
    let x_step = x_range / width as f64;
    let y_step = y_range / height as f64;

    let mut result = vec![0.0f64; w * h];
    result
        .par_chunks_mut(w)
        .enumerate()
        .for_each(|(py, row)| {
            for px in 0..w {
                let zr = x_min + px as f64 * x_step;
                let zi = y_min + py as f64 * y_step;
                row[px] = iterate_julia(zr, zi, c.0, c.1, max_iter);
            }
        });
    result
}

fn compute_burning_ship(
    width: u32, height: u32,
    center: (f64, f64), zoom: f64, max_iter: u32,
) -> Vec<f64> {
    let w = width as usize;
    let h = height as usize;
    let aspect = width as f64 / height as f64;
    let x_range = 3.5 / zoom;
    let y_range = x_range / aspect;
    let x_min = center.0 - x_range / 2.0;
    let y_min = center.1 - y_range / 2.0;
    let x_step = x_range / width as f64;
    let y_step = y_range / height as f64;

    let mut result = vec![0.0f64; w * h];
    result
        .par_chunks_mut(w)
        .enumerate()
        .for_each(|(py, row)| {
            let ci = y_min + py as f64 * y_step;
            for px in 0..w {
                let cr = x_min + px as f64 * x_step;
                row[px] = iterate_burning_ship(cr, ci, max_iter);
            }
        });
    result
}

fn compute_newton(
    width: u32, height: u32,
    center: (f64, f64), zoom: f64, max_iter: u32,
) -> Vec<f64> {
    let w = width as usize;
    let h = height as usize;
    let aspect = width as f64 / height as f64;
    let x_range = 4.0 / zoom;
    let y_range = x_range / aspect;
    let x_min = center.0 - x_range / 2.0;
    let y_min = center.1 - y_range / 2.0;
    let x_step = x_range / width as f64;
    let y_step = y_range / height as f64;

    let roots: [(f64, f64); 3] = [
        (1.0, 0.0),
        (-0.5, 0.866_025_403_784_438_6),
        (-0.5, -0.866_025_403_784_438_6),
    ];
    let tol = 1e-6;

    let mut result = vec![0.0f64; w * h];
    result
        .par_chunks_mut(w)
        .enumerate()
        .for_each(|(py, row)| {
            for px in 0..w {
                let mut zr = x_min + px as f64 * x_step;
                let mut zi = y_min + py as f64 * y_step;
                let mut root_id: i32 = -1;
                let mut shade = 0u32;

                for i in 0..max_iter {
                    let zr2 = zr * zr;
                    let zi2 = zi * zi;
                    let z2r = zr2 - zi2;
                    let z2i = 2.0 * zr * zi;
                    let z3r = z2r * zr - z2i * zi;
                    let z3i = z2r * zi + z2i * zr;
                    let dr = 3.0 * z2r;
                    let di = 3.0 * z2i;
                    let denom = dr * dr + di * di;
                    if denom < 1e-24 {
                        break;
                    }
                    let fr = z3r - 1.0;
                    let fi = z3i;
                    let qr = (fr * dr + fi * di) / denom;
                    let qi = (fi * dr - fr * di) / denom;
                    zr -= qr;
                    zi -= qi;

                    for (ri, root) in roots.iter().enumerate() {
                        let dr = zr - root.0;
                        let di = zi - root.1;
                        if dr * dr + di * di < tol {
                            root_id = ri as i32;
                            shade = i;
                            break;
                        }
                    }
                    if root_id >= 0 {
                        break;
                    }
                }

                row[px] = if root_id >= 0 {
                    (root_id as f64 / 3.0
                        + (1.0 - shade as f64 / max_iter as f64) * 0.25)
                        * max_iter as f64
                } else {
                    0.0
                };
            }
        });
    result
}

// ── Rendering ───────────────────────────────────────────────────────────────

fn render(data: &[f64], width: u32, height: u32, palette: Palette, color_offset: f64) -> RgbImage {
    let cmap = build_colormap(palette, 2048);
    let cmap_len = cmap.len();
    let offset = (color_offset * cmap_len as f64) as usize;

    let d_max = data
        .iter()
        .copied()
        .filter(|&v| v > 0.0)
        .fold(0.0f64, f64::max)
        .max(1.0);

    let pixels: Vec<[u8; 3]> = data
        .par_iter()
        .map(|&val| {
            if val == 0.0 {
                [0, 0, 0]
            } else {
                let normed = val / d_max;
                let idx = ((normed * 12.0 * cmap_len as f64) as usize + offset) % cmap_len;
                cmap[idx]
            }
        })
        .collect();

    let mut img = RgbImage::new(width, height);
    for (i, rgb) in pixels.into_iter().enumerate() {
        let x = (i % width as usize) as u32;
        let y = (i / width as usize) as u32;
        img.put_pixel(x, y, Rgb(rgb));
    }
    img
}

// ── Main ────────────────────────────────────────────────────────────────────

fn generate(
    fractal: FractalType,
    palette: Palette,
    max_iter: u32,
    output: Option<PathBuf>,
    rng: &mut Rng,
) {
    let t_search = Instant::now();
    let params = match fractal {
        FractalType::Mandelbrot => find_interesting_mandelbrot(rng, max_iter),
        FractalType::Julia => find_interesting_julia(rng, max_iter),
        FractalType::BurningShip => find_interesting_burning_ship(rng, max_iter),
        FractalType::Newton => find_interesting_newton(rng, max_iter),
        FractalType::All => unreachable!(),
    };
    let search_time = t_search.elapsed();

    let out_path = output.unwrap_or_else(|| {
        PathBuf::from(format!("fractal_{fractal}_{palette}_{WIDTH}x{HEIGHT}.png"))
    });

    println!(
        "Generating {fractal} ({WIDTH}x{HEIGHT}, palette={palette}, max_iter={max_iter})"
    );
    println!(
        "  Found interesting view in {:.2}s: center=({:.10}, {:.10}), zoom={:.1}",
        search_time.as_secs_f64(), params.center.0, params.center.1, params.zoom
    );
    if let Some(c) = params.julia_c {
        println!("  julia c=({:.10}, {:.10})", c.0, c.1);
    }

    let t0 = Instant::now();
    let data = match fractal {
        FractalType::Mandelbrot => {
            compute_mandelbrot(WIDTH, HEIGHT, params.center, params.zoom, max_iter)
        }
        FractalType::Julia => {
            let c = params.julia_c.unwrap();
            compute_julia(WIDTH, HEIGHT, c, params.zoom, max_iter)
        }
        FractalType::BurningShip => {
            compute_burning_ship(WIDTH, HEIGHT, params.center, params.zoom, max_iter)
        }
        FractalType::Newton => {
            compute_newton(WIDTH, HEIGHT, params.center, params.zoom, max_iter.min(500))
        }
        FractalType::All => unreachable!(),
    };
    let t1 = Instant::now();
    println!("  Computed in {:.2}s", (t1 - t0).as_secs_f64());

    let img = render(&data, WIDTH, HEIGHT, palette, params.color_offset);
    img.save(&out_path).expect("Failed to save image");
    let t2 = Instant::now();
    println!(
        "  Saved to {} ({:.2}s)",
        out_path.display(),
        (t2 - t1).as_secs_f64()
    );
}

fn main() {
    let cli = Cli::parse();

    let seed = cli.seed.unwrap_or_else(|| {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64
    });
    println!("Seed: {seed}");
    let mut rng = Rng::new(seed);

    match cli.fractal {
        FractalType::All => {
            let types = [
                FractalType::Mandelbrot,
                FractalType::Julia,
                FractalType::BurningShip,
                FractalType::Newton,
            ];
            for ft in types {
                let pal = cli.palette.unwrap_or_else(|| rng.choose(&ALL_PALETTES));
                generate(ft, pal, cli.max_iter, None, &mut rng);
            }
        }
        ft => {
            let pal = cli.palette.unwrap_or_else(|| rng.choose(&ALL_PALETTES));
            generate(ft, pal, cli.max_iter, cli.output, &mut rng);
        }
    }
}
