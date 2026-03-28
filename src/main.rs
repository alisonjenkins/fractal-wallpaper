use clap::{Parser, ValueEnum};
use image::{Rgb, RgbImage};
use rayon::prelude::*;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
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

    /// Number of samples for histogram-based fractals (flame, buddhabrot, attractor)
    #[arg(long)]
    samples: Option<u64>,
}

#[derive(Clone, Copy, ValueEnum, PartialEq)]
enum FractalType {
    Mandelbrot,
    Julia,
    BurningShip,
    Newton,
    Flame,
    Buddhabrot,
    StrangeAttractor,
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
            Self::Flame => write!(f, "flame"),
            Self::Buddhabrot => write!(f, "buddhabrot"),
            Self::StrangeAttractor => write!(f, "strange-attractor"),
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

// ── Histogram (for density-based fractals) ──────────────────────────────────

struct Histogram {
    bins: Vec<AtomicU64>,
    width: u32,
    height: u32,
}

impl Histogram {
    fn new(width: u32, height: u32) -> Self {
        let n = (width as usize) * (height as usize);
        let mut bins = Vec::with_capacity(n);
        for _ in 0..n {
            bins.push(AtomicU64::new(0));
        }
        Self { bins, width, height }
    }

    fn increment(&self, x: u32, y: u32) {
        if x < self.width && y < self.height {
            let idx = y as usize * self.width as usize + x as usize;
            self.bins[idx].fetch_add(1, Ordering::Relaxed);
        }
    }

    fn to_vec_f64(&self) -> Vec<f64> {
        self.bins
            .iter()
            .map(|b| b.load(Ordering::Relaxed) as f64)
            .collect()
    }
}

fn tone_map_log(histogram: &[f64]) -> Vec<f64> {
    histogram
        .iter()
        .map(|&v| if v > 0.0 { v.ln_1p() } else { 0.0 })
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
    attractor_params: Option<(f64, f64, f64, f64)>,
    attractor_type: Option<AttractorType>,
    buddhabrot_iters: Option<(u32, u32, u32)>,
    flame_transforms: Option<Vec<FlameTransform>>,
    samples: u64,
}

#[derive(Clone, Copy)]
enum AttractorType {
    Clifford,
    DeJong,
}

#[derive(Clone)]
#[allow(dead_code)]
struct FlameTransform {
    a: f64, b: f64, c: f64, d: f64, e: f64, f: f64,
    variations: [f64; 10],
    weight: f64,
    color: f64,
}

impl Default for FractalParams {
    fn default() -> Self {
        Self {
            center: (0.0, 0.0),
            zoom: 1.0,
            julia_c: None,
            color_offset: 0.0,
            attractor_params: None,
            attractor_type: None,
            buddhabrot_iters: None,
            flame_transforms: None,
            samples: 0,
        }
    }
}

fn find_interesting_mandelbrot(rng: &mut Rng, max_iter: u32) -> FractalParams {
    let mut best_params = FractalParams {
        center: (-0.75, 0.0),
        color_offset: rng.f64(),
        ..Default::default()
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
                color_offset: rng.f64(),
                ..Default::default()
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
        julia_c: Some((-0.7269, 0.1889)),
        color_offset: rng.f64(),
        ..Default::default()
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
                zoom,
                julia_c: Some(c),
                color_offset: rng.f64(),
                ..Default::default()
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
        color_offset: rng.f64(),
        ..Default::default()
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
                color_offset: rng.f64(),
                ..Default::default()
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
        color_offset: rng.f64(),
        ..Default::default()
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

// ── Strange Attractor ───────────────────────────────────────────────────────

fn compute_attractor(
    width: u32, height: u32,
    a: f64, b: f64, c: f64, d: f64,
    at: AttractorType,
    samples: u64,
    rng_seed: u64,
) -> Vec<f64> {
    // Pre-compute bounding box with a short orbit
    let mut x = 0.1f64;
    let mut y = 0.1f64;
    let mut x_min = f64::MAX;
    let mut x_max = f64::MIN;
    let mut y_min = f64::MAX;
    let mut y_max = f64::MIN;

    for _ in 0..200_000 {
        let (xn, yn) = match at {
            AttractorType::Clifford => (
                (a * y).sin() + c * (a * x).cos(),
                (b * x).sin() + d * (b * y).cos(),
            ),
            AttractorType::DeJong => (
                (a * y).sin() - (b * x).cos(),
                (c * x).sin() - (d * y).cos(),
            ),
        };
        x = xn;
        y = yn;
        if x.is_finite() && y.is_finite() {
            x_min = x_min.min(x);
            x_max = x_max.max(x);
            y_min = y_min.min(y);
            y_max = y_max.max(y);
        }
    }

    let margin = 0.05;
    let x_range = (x_max - x_min) * (1.0 + margin);
    let y_range = (y_max - y_min) * (1.0 + margin);
    let x_center = (x_min + x_max) / 2.0;
    let y_center = (y_min + y_max) / 2.0;

    let aspect = width as f64 / height as f64;
    let (view_w, view_h) = if x_range / y_range > aspect {
        (x_range, x_range / aspect)
    } else {
        (y_range * aspect, y_range)
    };
    let vx_min = x_center - view_w / 2.0;
    let vy_min = y_center - view_h / 2.0;

    let histogram = Histogram::new(width, height);
    let num_threads = rayon::current_num_threads().max(1);
    let samples_per_thread = samples / num_threads as u64;

    (0..num_threads).into_par_iter().for_each(|tid| {
        let mut rng = Rng::new(rng_seed.wrapping_add((tid as u64).wrapping_mul(0x9e3779b97f4a7c15)));
        let mut x = rng.range(-0.1, 0.1);
        let mut y = rng.range(-0.1, 0.1);

        for i in 0..samples_per_thread {
            let (xn, yn) = match at {
                AttractorType::Clifford => (
                    (a * y).sin() + c * (a * x).cos(),
                    (b * x).sin() + d * (b * y).cos(),
                ),
                AttractorType::DeJong => (
                    (a * y).sin() - (b * x).cos(),
                    (c * x).sin() - (d * y).cos(),
                ),
            };
            x = xn;
            y = yn;

            if i < 100 { continue; }

            let px = ((x - vx_min) / view_w * width as f64) as i64;
            let py = ((y - vy_min) / view_h * height as f64) as i64;
            if px >= 0 && px < width as i64 && py >= 0 && py < height as i64 {
                histogram.increment(px as u32, py as u32);
            }
        }
    });

    tone_map_log(&histogram.to_vec_f64())
}

fn score_attractor_params(a: f64, b: f64, c: f64, d: f64, at: AttractorType) -> f64 {
    let probe_w: u32 = 272;
    let probe_h: u32 = 72;
    let probe_samples: u64 = 200_000;

    let mut x = 0.1f64;
    let mut y = 0.1f64;
    let mut x_min = f64::MAX;
    let mut x_max = f64::MIN;
    let mut y_min = f64::MAX;
    let mut y_max = f64::MIN;

    for i in 0..probe_samples {
        let (xn, yn) = match at {
            AttractorType::Clifford => (
                (a * y).sin() + c * (a * x).cos(),
                (b * x).sin() + d * (b * y).cos(),
            ),
            AttractorType::DeJong => (
                (a * y).sin() - (b * x).cos(),
                (c * x).sin() - (d * y).cos(),
            ),
        };
        x = xn;
        y = yn;

        if !x.is_finite() || !y.is_finite() || x.abs() > 1e10 || y.abs() > 1e10 {
            return 0.0; // Divergent
        }

        if i > 100 {
            x_min = x_min.min(x);
            x_max = x_max.max(x);
            y_min = y_min.min(y);
            y_max = y_max.max(y);
        }
    }

    let bbox_w = x_max - x_min;
    let bbox_h = y_max - y_min;
    if bbox_w < 0.5 || bbox_h < 0.5 {
        return 0.0; // Fixed point or tiny orbit
    }

    // Accumulate into a small probe histogram
    let aspect = probe_w as f64 / probe_h as f64;
    let (vw, vh) = if bbox_w / bbox_h > aspect {
        (bbox_w * 1.05, bbox_w * 1.05 / aspect)
    } else {
        (bbox_h * 1.05 * aspect, bbox_h * 1.05)
    };
    let vx = (x_min + x_max) / 2.0 - vw / 2.0;
    let vy = (y_min + y_max) / 2.0 - vh / 2.0;

    let mut histogram = vec![0u32; probe_w as usize * probe_h as usize];
    x = 0.1;
    y = 0.1;
    for i in 0..probe_samples {
        let (xn, yn) = match at {
            AttractorType::Clifford => (
                (a * y).sin() + c * (a * x).cos(),
                (b * x).sin() + d * (b * y).cos(),
            ),
            AttractorType::DeJong => (
                (a * y).sin() - (b * x).cos(),
                (c * x).sin() - (d * y).cos(),
            ),
        };
        x = xn;
        y = yn;
        if i < 100 { continue; }

        let px = ((x - vx) / vw * probe_w as f64) as i64;
        let py = ((y - vy) / vh * probe_h as f64) as i64;
        if px >= 0 && px < probe_w as i64 && py >= 0 && py < probe_h as i64 {
            histogram[py as usize * probe_w as usize + px as usize] += 1;
        }
    }

    let total = histogram.len();
    let filled = histogram.iter().filter(|&&v| v > 0).count();
    let fill_rate = filled as f64 / total as f64;

    // Want 5-40% fill
    let fill_score = if fill_rate < 0.03 || fill_rate > 0.60 {
        0.1
    } else if fill_rate >= 0.05 && fill_rate <= 0.40 {
        1.0
    } else {
        0.5
    };

    // Check spread across 4 horizontal quarters
    let qw = probe_w as usize / 4;
    let mut quarters_active = 0u32;
    for q in 0..4 {
        let mut q_filled = 0;
        let mut q_total = 0;
        for row in 0..probe_h as usize {
            for col in (q * qw)..((q + 1) * qw) {
                q_total += 1;
                if histogram[row * probe_w as usize + col] > 0 {
                    q_filled += 1;
                }
            }
        }
        if q_filled as f64 / q_total as f64 > 0.02 {
            quarters_active += 1;
        }
    }
    let spread_score = match quarters_active {
        4 => 1.0,
        3 => 0.6,
        _ => 0.1,
    };

    fill_score * 0.5 + spread_score * 0.5
}

fn find_interesting_attractor(rng: &mut Rng) -> FractalParams {
    let mut best_params = FractalParams {
        attractor_params: Some((1.7, 1.7, 0.6, 1.2)),
        attractor_type: Some(AttractorType::Clifford),
        samples: 20_000_000,
        color_offset: rng.f64(),
        ..Default::default()
    };
    let mut best_score = 0.0f64;

    for _ in 0..80 {
        let at = if rng.f64() < 0.5 {
            AttractorType::Clifford
        } else {
            AttractorType::DeJong
        };
        let a = rng.range(-3.0, 3.0);
        let b = rng.range(-3.0, 3.0);
        let c = rng.range(-3.0, 3.0);
        let d = rng.range(-3.0, 3.0);

        let score = score_attractor_params(a, b, c, d, at);

        if score > best_score {
            best_score = score;
            best_params = FractalParams {
                attractor_params: Some((a, b, c, d)),
                attractor_type: Some(at),
                samples: 20_000_000,
                color_offset: rng.f64(),
                ..Default::default()
            };
        }

        if best_score > 0.85 {
            break;
        }
    }

    best_params
}

// ── Buddhabrot ──────────────────────────────────────────────────────────────

fn compute_buddhabrot(
    width: u32, height: u32,
    samples: u64,
    r_iter: u32, g_iter: u32, b_iter: u32,
    rng_seed: u64,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let hist_r = Histogram::new(width, height);
    let hist_g = Histogram::new(width, height);
    let hist_b = Histogram::new(width, height);

    let num_threads = rayon::current_num_threads().max(1);
    let samples_per_thread = samples / num_threads as u64;
    let max_channel_iter = r_iter.max(g_iter).max(b_iter);

    // The viewable region: centered on the Mandelbrot set, stretched for ultra-wide
    let view_x_min = -2.5;
    let view_x_max = 1.5;
    let view_y_range = (view_x_max - view_x_min) / (width as f64 / height as f64);
    let view_y_min = -view_y_range / 2.0;

    (0..num_threads).into_par_iter().for_each(|tid| {
        let mut rng = Rng::new(rng_seed.wrapping_add((tid as u64).wrapping_mul(0x9e3779b97f4a7c15)));
        let mut orbit = Vec::with_capacity(max_channel_iter as usize);

        for _ in 0..samples_per_thread {
            let cr = rng.range(-2.0, 1.0);
            let ci = rng.range(-1.5, 1.5);

            // Check if point escapes
            let mut zr = 0.0;
            let mut zi = 0.0;
            let mut escaped = false;

            orbit.clear();
            for i in 0..max_channel_iter {
                let zr2 = zr * zr;
                let zi2 = zi * zi;
                if zr2 + zi2 > 4.0 {
                    escaped = true;
                    let _ = i;
                    break;
                }
                orbit.push((zr, zi));
                zi = 2.0 * zr * zi + ci;
                zr = zr2 - zi2 + cr;
            }

            if !escaped { continue; }

            // Accumulate orbit into histograms
            for (i, &(ozr, ozi)) in orbit.iter().enumerate() {
                let px = ((ozr - view_x_min) / (view_x_max - view_x_min) * width as f64) as i64;
                let py = ((ozi - view_y_min) / view_y_range * height as f64) as i64;
                if px >= 0 && px < width as i64 && py >= 0 && py < height as i64 {
                    let ui = i as u32;
                    if ui < b_iter {
                        hist_b.increment(px as u32, py as u32);
                    }
                    if ui < g_iter {
                        hist_g.increment(px as u32, py as u32);
                    }
                    if ui < r_iter {
                        hist_r.increment(px as u32, py as u32);
                    }
                }
            }
        }
    });

    (hist_r.to_vec_f64(), hist_g.to_vec_f64(), hist_b.to_vec_f64())
}

fn render_buddhabrot(
    r_data: &[f64], g_data: &[f64], b_data: &[f64],
    width: u32, height: u32,
) -> RgbImage {
    // Use percentile-based normalization to avoid outlier domination,
    // then gamma correction for brightness
    let gamma = 0.4;

    // Find 99.5th percentile for each channel to normalize against
    let percentile_max = |data: &[f64]| -> f64 {
        let mut sorted: Vec<f64> = data.iter().copied().filter(|&v| v > 0.0).collect();
        if sorted.is_empty() { return 1.0; }
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let idx = (sorted.len() as f64 * 0.995) as usize;
        sorted[idx.min(sorted.len() - 1)].max(1.0)
    };
    let r_norm = percentile_max(r_data);
    let g_norm = percentile_max(g_data);
    let b_norm = percentile_max(b_data);

    let mut img = RgbImage::new(width, height);
    for (i, ((r, g), b)) in r_data.iter().zip(g_data).zip(b_data).enumerate() {
        let x = (i % width as usize) as u32;
        let y = (i / width as usize) as u32;
        if *r == 0.0 && *g == 0.0 && *b == 0.0 {
            img.put_pixel(x, y, Rgb([0, 0, 0]));
        } else {
            img.put_pixel(x, y, Rgb([
                ((*r / r_norm).min(1.0).powf(gamma) * 255.0) as u8,
                ((*g / g_norm).min(1.0).powf(gamma) * 255.0) as u8,
                ((*b / b_norm).min(1.0).powf(gamma) * 255.0) as u8,
            ]));
        }
    }
    img
}

fn find_interesting_buddhabrot(rng: &mut Rng) -> FractalParams {
    // Buddhabrot always looks good — the main thing to vary is the iteration triple
    let triples: [(u32, u32, u32); 6] = [
        (5000, 500, 50),    // Classic nebula (red deep, blue shallow)
        (2000, 200, 20),    // Softer, faster
        (10000, 1000, 100), // Ultra-detailed
        (500, 5000, 50),    // Green dominant
        (50, 500, 5000),    // Blue dominant
        (1000, 100, 1000),  // Purple (red+blue)
    ];

    let triple = rng.choose(&triples);

    FractalParams {
        buddhabrot_iters: Some(triple),
        samples: 50_000_000,
        color_offset: rng.f64(),
        ..Default::default()
    }
}

// ── Flame Fractals ──────────────────────────────────────────────────────────

fn apply_variations(x: f64, y: f64, weights: &[f64; 10]) -> (f64, f64) {
    let r2 = x * x + y * y;
    let r = r2.sqrt();
    let theta = y.atan2(x);
    let mut vx = 0.0;
    let mut vy = 0.0;

    // V0: Linear
    if weights[0] != 0.0 { vx += weights[0] * x; vy += weights[0] * y; }
    // V1: Sinusoidal
    if weights[1] != 0.0 { vx += weights[1] * x.sin(); vy += weights[1] * y.sin(); }
    // V2: Spherical
    if weights[2] != 0.0 {
        let s = if r2 > 1e-10 { 1.0 / r2 } else { 1.0 };
        vx += weights[2] * x * s;
        vy += weights[2] * y * s;
    }
    // V3: Swirl
    if weights[3] != 0.0 {
        let sr = r2.sin();
        let cr = r2.cos();
        vx += weights[3] * (x * sr - y * cr);
        vy += weights[3] * (x * cr + y * sr);
    }
    // V4: Horseshoe
    if weights[4] != 0.0 {
        let inv_r = if r > 1e-10 { 1.0 / r } else { 1.0 };
        vx += weights[4] * inv_r * (x - y) * (x + y);
        vy += weights[4] * inv_r * 2.0 * x * y;
    }
    // V5: Polar
    if weights[5] != 0.0 {
        vx += weights[5] * theta / std::f64::consts::PI;
        vy += weights[5] * (r - 1.0);
    }
    // V6: Handkerchief
    if weights[6] != 0.0 {
        vx += weights[6] * r * (theta + r).sin();
        vy += weights[6] * r * (theta - r).cos();
    }
    // V7: Heart
    if weights[7] != 0.0 {
        vx += weights[7] * r * (theta * r).sin();
        vy += weights[7] * -r * (theta * r).cos();
    }
    // V8: Disc
    if weights[8] != 0.0 {
        let t = theta / std::f64::consts::PI;
        let pr = std::f64::consts::PI * r;
        vx += weights[8] * t * pr.sin();
        vy += weights[8] * t * pr.cos();
    }
    // V9: Spiral
    if weights[9] != 0.0 {
        let inv_r = if r > 1e-10 { 1.0 / r } else { 1.0 };
        vx += weights[9] * inv_r * (theta.cos() + r.sin());
        vy += weights[9] * inv_r * (theta.sin() - r.cos());
    }

    (vx, vy)
}

fn compute_flame(
    width: u32, height: u32,
    transforms: &[FlameTransform],
    samples: u64,
    rng_seed: u64,
) -> Vec<f64> {
    // Pre-compute cumulative weights
    let total_weight: f64 = transforms.iter().map(|t| t.weight).sum();
    let cum_weights: Vec<f64> = transforms
        .iter()
        .scan(0.0, |acc, t| { *acc += t.weight / total_weight; Some(*acc) })
        .collect();

    let histogram = Histogram::new(width, height);
    let num_threads = rayon::current_num_threads().max(1);
    let samples_per_thread = samples / num_threads as u64;

    (0..num_threads).into_par_iter().for_each(|tid| {
        let mut rng = Rng::new(rng_seed.wrapping_add((tid as u64).wrapping_mul(0x9e3779b97f4a7c15)));
        let mut x = rng.range(-1.0, 1.0);
        let mut y = rng.range(-1.0, 1.0);

        for i in 0..samples_per_thread {
            // Select transform by weight
            let r = rng.f64();
            let mut chosen = 0;
            for (j, &cw) in cum_weights.iter().enumerate() {
                if r <= cw {
                    chosen = j;
                    break;
                }
            }
            let t = &transforms[chosen];

            // Apply affine transform
            let ax = t.a * x + t.b * y + t.c;
            let ay = t.d * x + t.e * y + t.f;

            // Apply variation functions
            let (vx, vy) = apply_variations(ax, ay, &t.variations);
            x = vx;
            y = vy;

            // Divergence guard
            if !x.is_finite() || !y.is_finite() || x.abs() > 1e10 || y.abs() > 1e10 {
                x = rng.range(-1.0, 1.0);
                y = rng.range(-1.0, 1.0);
                continue;
            }

            if i < 20 { continue; } // Warmup

            // Map to pixel — flame coordinates typically in [-2, 2]
            let px = ((x + 2.5) / 5.0 * width as f64) as i64;
            let py = ((y + 1.5) / 3.0 * height as f64) as i64;
            if px >= 0 && px < width as i64 && py >= 0 && py < height as i64 {
                histogram.increment(px as u32, py as u32);
            }
        }
    });

    tone_map_log(&histogram.to_vec_f64())
}

fn random_flame_transform(rng: &mut Rng) -> FlameTransform {
    let mut variations = [0.0f64; 10];
    // Pick 1-3 active variations
    let n_active = rng.choose(&[1, 1, 2, 2, 3]);
    for _ in 0..n_active {
        let idx = (rng.next_u64() as usize) % 10;
        variations[idx] = rng.range(0.2, 1.0);
    }
    // Normalize variation weights
    let sum: f64 = variations.iter().sum();
    if sum > 0.0 {
        for v in &mut variations {
            *v /= sum;
        }
    } else {
        variations[0] = 1.0; // Fallback to linear
    }

    FlameTransform {
        a: rng.range(-1.0, 1.0),
        b: rng.range(-1.0, 1.0),
        c: rng.range(-0.5, 0.5),
        d: rng.range(-1.0, 1.0),
        e: rng.range(-1.0, 1.0),
        f: rng.range(-0.5, 0.5),
        variations,
        weight: rng.range(0.2, 1.0),
        color: rng.f64(),
    }
}

fn score_flame_params(transforms: &[FlameTransform], rng_seed: u64) -> f64 {
    let probe_w: u32 = 272;
    let probe_h: u32 = 72;
    let probe_samples: u64 = 500_000;

    let total_weight: f64 = transforms.iter().map(|t| t.weight).sum();
    let cum_weights: Vec<f64> = transforms
        .iter()
        .scan(0.0, |acc, t| { *acc += t.weight / total_weight; Some(*acc) })
        .collect();

    let mut histogram = vec![0u32; probe_w as usize * probe_h as usize];
    let mut rng = Rng::new(rng_seed);
    let mut x = rng.range(-1.0, 1.0);
    let mut y = rng.range(-1.0, 1.0);

    for i in 0..probe_samples {
        let r = rng.f64();
        let mut chosen = 0;
        for (j, &cw) in cum_weights.iter().enumerate() {
            if r <= cw { chosen = j; break; }
        }
        let t = &transforms[chosen];
        let ax = t.a * x + t.b * y + t.c;
        let ay = t.d * x + t.e * y + t.f;
        let (vx, vy) = apply_variations(ax, ay, &t.variations);
        x = vx;
        y = vy;

        if !x.is_finite() || !y.is_finite() || x.abs() > 1e10 || y.abs() > 1e10 {
            return 0.0; // Divergent
        }

        if i < 20 { continue; }

        let px = ((x + 2.5) / 5.0 * probe_w as f64) as i64;
        let py = ((y + 1.5) / 3.0 * probe_h as f64) as i64;
        if px >= 0 && px < probe_w as i64 && py >= 0 && py < probe_h as i64 {
            histogram[py as usize * probe_w as usize + px as usize] += 1;
        }
    }

    let total = histogram.len();
    let filled = histogram.iter().filter(|&&v| v > 0).count();
    let fill_rate = filled as f64 / total as f64;

    let fill_score = if fill_rate < 0.02 || fill_rate > 0.60 {
        0.1
    } else if fill_rate >= 0.05 && fill_rate <= 0.40 {
        1.0
    } else {
        0.5
    };

    // Density variance — want structure, not uniform noise
    let vals: Vec<f64> = histogram.iter().filter(|&&v| v > 0).map(|&v| v as f64).collect();
    let density_score = if vals.is_empty() {
        0.0
    } else {
        let mean = vals.iter().sum::<f64>() / vals.len() as f64;
        let variance = vals.iter().map(|&v| (v - mean) * (v - mean)).sum::<f64>() / vals.len() as f64;
        let cv = variance.sqrt() / mean.max(1.0); // Coefficient of variation
        (cv / 3.0).min(1.0) // High CV = structured, not uniform
    };

    // Spread check
    let qw = probe_w as usize / 4;
    let mut quarters_active = 0u32;
    for q in 0..4 {
        let mut q_filled = 0;
        let mut q_total = 0;
        for row in 0..probe_h as usize {
            for col in (q * qw)..((q + 1) * qw) {
                q_total += 1;
                if histogram[row * probe_w as usize + col] > 0 {
                    q_filled += 1;
                }
            }
        }
        if q_filled as f64 / q_total as f64 > 0.01 {
            quarters_active += 1;
        }
    }
    let spread_score = match quarters_active {
        4 => 1.0,
        3 => 0.6,
        _ => 0.1,
    };

    fill_score * 0.35 + density_score * 0.35 + spread_score * 0.30
}

fn find_interesting_flame(rng: &mut Rng) -> FractalParams {
    let mut best_transforms = vec![random_flame_transform(rng), random_flame_transform(rng)];
    let mut best_score = 0.0f64;

    for _ in 0..30 {
        let n_transforms = rng.choose(&[2, 3, 3, 4, 5]);
        let transforms: Vec<FlameTransform> = (0..n_transforms)
            .map(|_| random_flame_transform(rng))
            .collect();

        let seed = rng.next_u64();
        let score = score_flame_params(&transforms, seed);

        if score > best_score {
            best_score = score;
            best_transforms = transforms;
        }

        if best_score > 0.80 {
            break;
        }
    }

    FractalParams {
        flame_transforms: Some(best_transforms),
        samples: 100_000_000,
        color_offset: rng.f64(),
        ..Default::default()
    }
}

// ── Rendering ───────────────────────────────────────────────────────────────

fn render(
    data: &[f64], width: u32, height: u32, palette: Palette,
    color_offset: f64, cycle_factor: f64,
) -> RgbImage {
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
                let idx = ((normed * cycle_factor * cmap_len as f64) as usize + offset) % cmap_len;
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
    samples: Option<u64>,
    output: Option<PathBuf>,
    rng: &mut Rng,
) {
    let seed = rng.next_u64();
    let t_search = Instant::now();
    let mut params = match fractal {
        FractalType::Mandelbrot => find_interesting_mandelbrot(rng, max_iter),
        FractalType::Julia => find_interesting_julia(rng, max_iter),
        FractalType::BurningShip => find_interesting_burning_ship(rng, max_iter),
        FractalType::Newton => find_interesting_newton(rng, max_iter),
        FractalType::Flame => find_interesting_flame(rng),
        FractalType::Buddhabrot => find_interesting_buddhabrot(rng),
        FractalType::StrangeAttractor => find_interesting_attractor(rng),
        FractalType::All => unreachable!(),
    };
    let search_time = t_search.elapsed();

    // Override samples if provided via CLI
    if let Some(s) = samples {
        params.samples = s;
    }

    let out_path = output.unwrap_or_else(|| {
        PathBuf::from(format!("fractal_{fractal}_{palette}_{WIDTH}x{HEIGHT}.png"))
    });

    println!(
        "Generating {fractal} ({WIDTH}x{HEIGHT}, palette={palette}, max_iter={max_iter})"
    );
    println!(
        "  Found interesting params in {:.2}s",
        search_time.as_secs_f64(),
    );

    let t0 = Instant::now();
    let img = match fractal {
        FractalType::Mandelbrot => {
            let data = compute_mandelbrot(WIDTH, HEIGHT, params.center, params.zoom, max_iter);
            render(&data, WIDTH, HEIGHT, palette, params.color_offset, 12.0)
        }
        FractalType::Julia => {
            let c = params.julia_c.unwrap();
            let data = compute_julia(WIDTH, HEIGHT, c, params.zoom, max_iter);
            render(&data, WIDTH, HEIGHT, palette, params.color_offset, 12.0)
        }
        FractalType::BurningShip => {
            let data = compute_burning_ship(WIDTH, HEIGHT, params.center, params.zoom, max_iter);
            render(&data, WIDTH, HEIGHT, palette, params.color_offset, 12.0)
        }
        FractalType::Newton => {
            let data = compute_newton(WIDTH, HEIGHT, params.center, params.zoom, max_iter.min(500));
            render(&data, WIDTH, HEIGHT, palette, params.color_offset, 12.0)
        }
        FractalType::StrangeAttractor => {
            let (a, b, c, d) = params.attractor_params.unwrap();
            let at = params.attractor_type.unwrap();
            let data = compute_attractor(WIDTH, HEIGHT, a, b, c, d, at, params.samples, seed);
            render(&data, WIDTH, HEIGHT, palette, params.color_offset, 2.0)
        }
        FractalType::Buddhabrot => {
            let (ri, gi, bi) = params.buddhabrot_iters.unwrap();
            let (r, g, b) = compute_buddhabrot(WIDTH, HEIGHT, params.samples, ri, gi, bi, seed);
            render_buddhabrot(&r, &g, &b, WIDTH, HEIGHT)
        }
        FractalType::Flame => {
            let transforms = params.flame_transforms.as_ref().unwrap();
            let data = compute_flame(WIDTH, HEIGHT, transforms, params.samples, seed);
            render(&data, WIDTH, HEIGHT, palette, params.color_offset, 2.0)
        }
        FractalType::All => unreachable!(),
    };
    let t1 = Instant::now();
    println!("  Computed in {:.2}s", (t1 - t0).as_secs_f64());

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
                FractalType::StrangeAttractor,
                FractalType::Buddhabrot,
                FractalType::Flame,
            ];
            for ft in types {
                let pal = cli.palette.unwrap_or_else(|| rng.choose(&ALL_PALETTES));
                generate(ft, pal, cli.max_iter, cli.samples, None, &mut rng);
            }
        }
        ft => {
            let pal = cli.palette.unwrap_or_else(|| rng.choose(&ALL_PALETTES));
            generate(ft, pal, cli.max_iter, cli.samples, cli.output, &mut rng);
        }
    }
}
