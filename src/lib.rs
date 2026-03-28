pub use image::{Rgb, RgbImage};

use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU64, Ordering};
use wide::{f64x4, CmpGt};

pub const DEFAULT_WIDTH: u32 = 5440;
pub const DEFAULT_HEIGHT: u32 = 1440;

// ── Simple PRNG (xoshiro256**) ──────────────────────────────────────────────

pub struct Rng {
    s: [u64; 4],
}

impl Rng {
    pub fn new(seed: u64) -> Self {
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

    pub fn next_u64(&mut self) -> u64 {
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

    pub fn f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    pub fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + self.f64() * (hi - lo)
    }

    pub fn choose<T: Copy>(&mut self, items: &[T]) -> T {
        let idx = (self.next_u64() as usize) % items.len();
        items[idx]
    }
}

#[derive(Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum FractalType {
    Mandelbrot,
    Julia,
    BurningShip,
    Newton,
    Flame,
    Buddhabrot,
    StrangeAttractor,
    Tricorn,
    Phoenix,
}

pub const ALL_FRACTAL_TYPES: &[FractalType] = &[
    FractalType::Mandelbrot,
    FractalType::Julia,
    FractalType::BurningShip,
    FractalType::Newton,
    FractalType::Tricorn,
    FractalType::Phoenix,
    FractalType::Flame,
    FractalType::Buddhabrot,
    FractalType::StrangeAttractor,
];

#[derive(Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum Palette {
    Twilight,
    Ocean,
    Fire,
    Neon,
    Frost,
    Earth,
    Sakura,
    CatppuccinMocha,
    CatppuccinMacchiato,
    CatppuccinFrappe,
    CatppuccinLatte,
    Random,
}

pub const ALL_PALETTES: &[Palette] = &[
    Palette::Twilight, Palette::Ocean, Palette::Fire,
    Palette::Neon, Palette::Frost, Palette::Earth, Palette::Sakura,
    Palette::CatppuccinMocha, Palette::CatppuccinMacchiato,
    Palette::CatppuccinFrappe, Palette::CatppuccinLatte,
    Palette::Random,
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
            Self::Tricorn => write!(f, "tricorn"),
            Self::Phoenix => write!(f, "phoenix"),
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
            Self::Sakura => write!(f, "sakura"),
            Self::CatppuccinMocha => write!(f, "catppuccin-mocha"),
            Self::CatppuccinMacchiato => write!(f, "catppuccin-macchiato"),
            Self::CatppuccinFrappe => write!(f, "catppuccin-frappe"),
            Self::CatppuccinLatte => write!(f, "catppuccin-latte"),
            Self::Random => write!(f, "random"),
        }
    }
}

// ── Color Theory Palette Generator ──────────────────────────────────────────

/// Convert HSL (h: 0-360, s: 0-1, l: 0-1) to RGB (0-255).
pub fn hsl_to_rgb(h: f64, s: f64, l: f64) -> [u8; 3] {
    let c = (1.0 - (2.0 * l - 1.0).abs()) * s;
    let h2 = h / 60.0;
    let x = c * (1.0 - (h2 % 2.0 - 1.0).abs());
    let (r1, g1, b1) = match h2 as u32 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    let m = l - c / 2.0;
    [
        ((r1 + m) * 255.0).clamp(0.0, 255.0) as u8,
        ((g1 + m) * 255.0).clamp(0.0, 255.0) as u8,
        ((b1 + m) * 255.0).clamp(0.0, 255.0) as u8,
    ]
}

/// Color harmony schemes for generating pleasing palettes.
#[derive(Clone, Copy)]
pub enum Harmony {
    Complementary,   // Base + opposite
    Analogous,       // Base + neighbors
    Triadic,         // Base + 120° + 240°
    SplitComplementary, // Base + opposite±30°
    Tetradic,        // Base + 90° + 180° + 270°
}

/// Generate a 12-anchor palette using color theory.
pub fn generate_random_palette(rng: &mut Rng) -> Vec<[u8; 3]> {
    let harmonies = [
        Harmony::Complementary,
        Harmony::Analogous,
        Harmony::Triadic,
        Harmony::SplitComplementary,
        Harmony::Tetradic,
    ];
    let harmony = rng.choose(&harmonies);

    // Pick a random base hue
    let base_hue = rng.range(0.0, 360.0);
    // Base saturation — avoid too desaturated
    let base_sat = rng.range(0.5, 1.0);

    // Generate key hues based on the harmony scheme
    let key_hues: Vec<f64> = match harmony {
        Harmony::Complementary => vec![base_hue, (base_hue + 180.0) % 360.0],
        Harmony::Analogous => vec![
            base_hue,
            (base_hue + 30.0) % 360.0,
            (base_hue + 330.0) % 360.0, // -30
        ],
        Harmony::Triadic => vec![
            base_hue,
            (base_hue + 120.0) % 360.0,
            (base_hue + 240.0) % 360.0,
        ],
        Harmony::SplitComplementary => vec![
            base_hue,
            (base_hue + 150.0) % 360.0,
            (base_hue + 210.0) % 360.0,
        ],
        Harmony::Tetradic => vec![
            base_hue,
            (base_hue + 90.0) % 360.0,
            (base_hue + 180.0) % 360.0,
            (base_hue + 270.0) % 360.0,
        ],
    };

    // Build 12 anchors by cycling through key hues with varying lightness
    // Pattern: dark → mid → bright → highlight → mid → dark (creates a smooth cycle)
    let lightness_curve = [
        0.08, 0.15, 0.25, 0.40, 0.55, 0.70, 0.85, 0.95,
        0.75, 0.50, 0.30, 0.12,
    ];
    let saturation_curve = [
        0.6, 0.8, 0.9, 1.0, 0.9, 0.7, 0.4, 0.2,
        0.6, 0.9, 1.0, 0.7,
    ];

    let mut anchors = Vec::with_capacity(12);
    for i in 0..12 {
        let hue_idx = i % key_hues.len();
        // Slight hue variation per anchor for richness
        let hue = (key_hues[hue_idx] + rng.range(-10.0, 10.0)).rem_euclid(360.0);
        let sat = (base_sat * saturation_curve[i] + rng.range(-0.05, 0.05)).clamp(0.0, 1.0);
        let lit = (lightness_curve[i] + rng.range(-0.03, 0.03)).clamp(0.02, 0.98);
        anchors.push(hsl_to_rgb(hue, sat, lit));
    }

    anchors
}

// ── Color Palettes ──────────────────────────────────────────────────────────

pub fn palette_anchors(palette: Palette) -> Vec<[u8; 3]> {
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
        Palette::Sakura => vec![
            [30, 5, 20], [80, 10, 50], [150, 30, 80], [200, 60, 120],
            [240, 110, 160], [255, 170, 200], [255, 220, 230], [255, 245, 245],
            [255, 200, 210], [230, 130, 170], [180, 60, 110], [100, 20, 60],
        ],
        // Catppuccin palettes — accent colors cycling through with base tones
        // Colors from https://catppuccin.com/palette
        Palette::CatppuccinMocha => vec![
            // Crust, Mantle, Base, Surface0 → accents → back to dark
            [17, 17, 27],    // Crust
            [24, 24, 37],    // Mantle
            [137, 180, 250], // Blue
            [116, 199, 236], // Sapphire
            [148, 226, 213], // Teal
            [166, 227, 161], // Green
            [249, 226, 175], // Yellow
            [250, 179, 135], // Peach
            [243, 139, 168], // Red
            [245, 194, 231], // Pink
            [203, 166, 247], // Mauve
            [30, 30, 46],    // Base
        ],
        Palette::CatppuccinMacchiato => vec![
            [24, 25, 38],    // Crust
            [30, 32, 48],    // Mantle
            [138, 173, 244], // Blue
            [125, 196, 228], // Sapphire
            [139, 213, 202], // Teal
            [166, 218, 149], // Green
            [238, 212, 159], // Yellow
            [245, 169, 127], // Peach
            [237, 135, 150], // Red
            [245, 189, 230], // Pink
            [198, 160, 246], // Mauve
            [36, 39, 58],    // Base
        ],
        Palette::CatppuccinFrappe => vec![
            [35, 38, 52],    // Crust
            [41, 44, 60],    // Mantle
            [140, 170, 238], // Blue
            [133, 193, 220], // Sapphire
            [129, 200, 190], // Teal
            [166, 209, 137], // Green
            [229, 200, 144], // Yellow
            [239, 159, 118], // Peach
            [231, 130, 132], // Red
            [244, 184, 228], // Pink
            [202, 158, 230], // Mauve
            [48, 52, 70],    // Base
        ],
        Palette::CatppuccinLatte => vec![
            [220, 224, 232], // Crust
            [230, 233, 239], // Mantle
            [30, 102, 245],  // Blue
            [32, 159, 181],  // Sapphire
            [23, 146, 153],  // Teal
            [64, 160, 43],   // Green
            [223, 142, 29],  // Yellow
            [254, 100, 11],  // Peach
            [210, 15, 57],   // Red
            [234, 118, 203], // Pink
            [136, 57, 239],  // Mauve
            [239, 241, 245], // Base
        ],
        Palette::Random => {
            // Fallback if Random reaches here unresolved — use a pleasant default
            vec![
                [10, 2, 30], [40, 5, 80], [90, 20, 140], [160, 50, 180],
                [220, 100, 160], [255, 160, 120], [255, 220, 180], [255, 255, 240],
                [200, 180, 255], [120, 100, 200], [60, 40, 140], [20, 10, 60],
            ]
        }
    }
}

pub fn build_colormap_from_anchors(anchors: &[[u8; 3]], n: usize) -> Vec<[u8; 3]> {
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

pub struct Histogram {
    pub bins: Vec<AtomicU64>,
    pub width: u32,
    pub height: u32,
}

impl Histogram {
    pub fn new(width: u32, height: u32) -> Self {
        let n = (width as usize) * (height as usize);
        let mut bins = Vec::with_capacity(n);
        for _ in 0..n {
            bins.push(AtomicU64::new(0));
        }
        Self { bins, width, height }
    }

    pub fn increment(&self, x: u32, y: u32) {
        if x < self.width && y < self.height {
            let idx = y as usize * self.width as usize + x as usize;
            self.bins[idx].fetch_add(1, Ordering::Relaxed);
        }
    }

    pub fn to_vec_f64(&self) -> Vec<f64> {
        self.bins
            .iter()
            .map(|b| b.load(Ordering::Relaxed) as f64)
            .collect()
    }
}

pub fn tone_map_log(histogram: &[f64]) -> Vec<f64> {
    histogram
        .iter()
        .map(|&v| if v > 0.0 { v.ln_1p() } else { 0.0 })
        .collect()
}

// ── Single-point iteration functions (for probing) ──────────────────────────

/// Check if point is inside the main cardioid or period-2 bulb.
/// These points never escape, so we can skip iteration entirely.
#[inline(always)]
pub fn in_cardioid_or_bulb(cr: f64, ci: f64) -> bool {
    let ci2 = ci * ci;
    // Main cardioid: |(c - 1/4)| < 1/2 * (1 - cos(theta))
    let q = (cr - 0.25) * (cr - 0.25) + ci2;
    if q * (q + (cr - 0.25)) <= 0.25 * ci2 {
        return true;
    }
    // Period-2 bulb: |(c + 1)| < 1/4
    if (cr + 1.0) * (cr + 1.0) + ci2 <= 0.0625 {
        return true;
    }
    false
}

pub fn iterate_mandelbrot(cr: f64, ci: f64, max_iter: u32) -> f64 {
    if in_cardioid_or_bulb(cr, ci) {
        return 0.0;
    }
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

pub fn iterate_julia(zr0: f64, zi0: f64, cr: f64, ci: f64, max_iter: u32) -> f64 {
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

pub fn iterate_burning_ship(cr: f64, ci: f64, max_iter: u32) -> f64 {
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

pub fn iterate_tricorn(cr: f64, ci: f64, max_iter: u32) -> f64 {
    // Tricorn/Mandelbar: z = conj(z)^2 + c
    let mut zr = 0.0;
    let mut zi = 0.0;
    let mut i = 0u32;
    while i < max_iter {
        let zr2 = zr * zr;
        let zi2 = zi * zi;
        if zr2 + zi2 > 65536.0 {
            break;
        }
        // conj(z)^2 = (zr - zi*i)^2 = zr^2 - zi^2 - 2*zr*zi*i
        let new_zr = zr2 - zi2 + cr;
        zi = -2.0 * zr * zi + ci; // Note: negative sign (conjugate)
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

pub fn iterate_phoenix(zr0: f64, zi0: f64, cr: f64, ci: f64, max_iter: u32) -> f64 {
    // Phoenix fractal: z_n+1 = z_n^2 + Re(c) + Im(c)*z_n-1
    let mut zr = zr0;
    let mut zi = zi0;
    let mut zr_prev = 0.0;
    let mut zi_prev = 0.0;
    let mut i = 0u32;
    while i < max_iter {
        let zr2 = zr * zr;
        let zi2 = zi * zi;
        if zr2 + zi2 > 65536.0 {
            break;
        }
        let new_zr = zr2 - zi2 + cr + ci * zr_prev;
        let new_zi = 2.0 * zr * zi + ci * zi_prev;
        zr_prev = zr;
        zi_prev = zi;
        zr = new_zr;
        zi = new_zi;
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
pub fn find_boundary_point(
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
    width: u32,
    height: u32,
) -> f64 {
    let aspect = width as f64 / height as f64;
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

#[derive(Serialize, Deserialize)]
pub struct FractalParams {
    pub center: (f64, f64),
    pub zoom: f64,
    pub julia_c: Option<(f64, f64)>,
    pub color_offset: f64,
    pub attractor_params: Option<(f64, f64, f64, f64)>,
    pub attractor_type: Option<AttractorType>,
    pub buddhabrot_iters: Option<(u32, u32, u32)>,
    pub flame_transforms: Option<Vec<FlameTransform>>,
    pub samples: u64,
    /// Pre-generated palette anchors for Random palette
    pub random_palette: Option<Vec<[u8; 3]>>,
}

/// Saved parameter file format — includes everything needed to reproduce a fractal.
#[derive(Serialize, Deserialize)]
pub struct SavedParams {
    pub fractal: FractalType,
    pub palette: Palette,
    pub max_iter: u32,
    pub seed: u64,
    pub params: FractalParams,
}

#[derive(Clone, Copy, Serialize, Deserialize)]
pub enum AttractorType {
    Clifford,
    DeJong,
}

#[derive(Clone, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct FlameTransform {
    pub a: f64, b: f64, c: f64, d: f64, e: f64, f: f64,
    pub variations: [f64; 10],
    pub weight: f64,
    pub color: f64,
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
            random_palette: None,
        }
    }
}

fn find_interesting_mandelbrot(rng: &mut Rng, max_iter: u32, width: u32, height: u32) -> FractalParams {
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
            width, height,
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

fn find_interesting_julia(rng: &mut Rng, max_iter: u32, width: u32, height: u32) -> FractalParams {
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
            width, height,
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

fn find_interesting_burning_ship(rng: &mut Rng, max_iter: u32, width: u32, height: u32) -> FractalParams {
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
            width, height,
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

/// SIMD smooth iteration for 4 Mandelbrot pixels at once.
/// Uses SIMD masks throughout the hot loop — no scalar extraction until the end.
fn iterate_mandelbrot_4x(cr: f64x4, ci: f64x4, max_iter: u32) -> [f64; 4] {
    // Pre-check cardioid/bulb for each lane
    let cr_arr: [f64; 4] = bytemuck::cast(cr);
    let ci_arr: [f64; 4] = bytemuck::cast(ci);
    let mut skip_mask = f64x4::ZERO; // All-ones for lanes to skip
    let mut any_active = false;
    for k in 0..4 {
        if in_cardioid_or_bulb(cr_arr[k], ci_arr[k]) {
            // Set this lane to all-ones (true mask)
            let mut arr = [0.0f64; 4];
            arr[k] = f64::from_bits(u64::MAX);
            skip_mask = skip_mask | f64x4::from(arr);
        } else {
            any_active = true;
        }
    }
    if !any_active { return [0.0; 4]; }

    let mut zr = f64x4::ZERO;
    let mut zi = f64x4::ZERO;
    // Track counts as f64x4 for pure SIMD increment
    let mut counts = f64x4::ZERO;
    let one = f64x4::splat(1.0);
    let threshold = f64x4::splat(65536.0);
    let two = f64x4::splat(2.0);
    // done_mask: all-ones for lanes that are finished (escaped or cardioid-skipped)
    let mut done_mask = skip_mask;

    for _ in 0..max_iter {
        let zr2 = zr * zr;
        let zi2 = zi * zi;
        let mag2 = zr2 + zi2;

        // Lanes that just escaped this iteration
        let just_escaped = mag2.cmp_gt(threshold);
        done_mask = done_mask | just_escaped;

        // Increment counts only for still-active lanes (not done)
        // active = NOT done_mask before this escape check
        // We want to count this iteration for lanes that haven't escaped yet
        // The increment must happen before we update done_mask, so use the
        // mask from before just_escaped was folded in.
        // Actually we already folded it in, so we need to use: NOT (done_mask)
        // But the lane that JUST escaped should still get its count incremented.
        // Solution: increment for lanes NOT in the old done_mask.
        // Recompute: active lanes are those not yet done BEFORE this iteration's escape.
        let active = !done_mask | just_escaped; // Was active: not done, OR just now escaped
        let not_skip = !skip_mask; // Never count cardioid-skipped lanes
        counts = counts + (active & not_skip).blend(one, f64x4::ZERO);

        // All lanes done?
        let done_bits: [u64; 4] = bytemuck::cast(done_mask);
        if done_bits[0] != 0 && done_bits[1] != 0 && done_bits[2] != 0 && done_bits[3] != 0 {
            break;
        }

        // z = z^2 + c for all lanes
        let new_zi = two * zr * zi + ci;
        let new_zr = zr2 - zi2 + cr;
        zr = new_zr;
        zi = new_zi;
    }

    // Extract final values for smooth coloring (scalar — only done once)
    let mag2_arr: [f64; 4] = bytemuck::cast(zr * zr + zi * zi);
    let counts_arr: [f64; 4] = bytemuck::cast(counts);
    let escaped_arr: [u64; 4] = bytemuck::cast(done_mask & !skip_mask);
    let mut result = [0.0f64; 4];
    for k in 0..4 {
        if escaped_arr[k] != 0 && counts_arr[k] > 0.0 {
            let mag = mag2_arr[k].sqrt();
            result[k] = counts_arr[k] + 1.0 - mag.ln().ln() / std::f64::consts::LN_2;
        }
    }
    result
}

pub fn compute_mandelbrot(
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
    let step4 = f64x4::from([0.0, 1.0, 2.0, 3.0]) * f64x4::splat(x_step);

    let mut result = vec![0.0f64; w * h];
    result
        .par_chunks_mut(w)
        .enumerate()
        .for_each(|(py, row)| {
            let ci = f64x4::splat(y_min + py as f64 * y_step);
            let mut px = 0;
            // SIMD: process 4 pixels at a time
            while px + 4 <= w {
                let base_cr = x_min + px as f64 * x_step;
                let cr = f64x4::splat(base_cr) + step4;
                let vals = iterate_mandelbrot_4x(cr, ci, max_iter);
                row[px..px + 4].copy_from_slice(&vals);
                px += 4;
            }
            // Scalar remainder
            while px < w {
                let cr = x_min + px as f64 * x_step;
                row[px] = iterate_mandelbrot(cr, y_min + py as f64 * y_step, max_iter);
                px += 1;
            }
        });
    result
}

/// SIMD smooth iteration for 4 Julia pixels at once.
/// Uses SIMD masks throughout — no scalar extraction in the hot loop.
fn iterate_julia_4x(zr0: f64x4, zi0: f64x4, cr: f64x4, ci: f64x4, max_iter: u32) -> [f64; 4] {
    let mut zr = zr0;
    let mut zi = zi0;
    let mut counts = f64x4::ZERO;
    let one = f64x4::splat(1.0);
    let threshold = f64x4::splat(65536.0);
    let two = f64x4::splat(2.0);
    let mut done_mask = f64x4::ZERO;

    for _ in 0..max_iter {
        let zr2 = zr * zr;
        let zi2 = zi * zi;
        let mag2 = zr2 + zi2;
        let just_escaped = mag2.cmp_gt(threshold);

        // Increment counts for lanes not yet done (including those escaping now)
        let active = !done_mask;
        counts = counts + active.blend(one, f64x4::ZERO);

        done_mask = done_mask | just_escaped;

        let done_bits: [u64; 4] = bytemuck::cast(done_mask);
        if done_bits[0] != 0 && done_bits[1] != 0 && done_bits[2] != 0 && done_bits[3] != 0 {
            break;
        }

        let new_zr = zr2 - zi2 + cr;
        let new_zi = two * zr * zi + ci;
        zr = new_zr;
        zi = new_zi;
    }

    let mag2_arr: [f64; 4] = bytemuck::cast(zr * zr + zi * zi);
    let counts_arr: [f64; 4] = bytemuck::cast(counts);
    let escaped_arr: [u64; 4] = bytemuck::cast(done_mask);
    let mut result = [0.0f64; 4];
    for k in 0..4 {
        if escaped_arr[k] != 0 && counts_arr[k] > 0.0 {
            let mag = mag2_arr[k].sqrt();
            result[k] = counts_arr[k] + 1.0 - mag.ln().ln() / std::f64::consts::LN_2;
        }
    }
    result
}

pub fn compute_julia(
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
    let step4 = f64x4::from([0.0, 1.0, 2.0, 3.0]) * f64x4::splat(x_step);
    let cr = f64x4::splat(c.0);
    let ci = f64x4::splat(c.1);

    let mut result = vec![0.0f64; w * h];
    result
        .par_chunks_mut(w)
        .enumerate()
        .for_each(|(py, row)| {
            let zi0 = f64x4::splat(y_min + py as f64 * y_step);
            let mut px = 0;
            while px + 4 <= w {
                let base_zr = x_min + px as f64 * x_step;
                let zr0 = f64x4::splat(base_zr) + step4;
                let vals = iterate_julia_4x(zr0, zi0, cr, ci, max_iter);
                row[px..px + 4].copy_from_slice(&vals);
                px += 4;
            }
            while px < w {
                let zr = x_min + px as f64 * x_step;
                let zi = y_min + py as f64 * y_step;
                row[px] = iterate_julia(zr, zi, c.0, c.1, max_iter);
                px += 1;
            }
        });
    result
}

/// SIMD smooth iteration for 4 Burning Ship pixels at once.
/// Uses SIMD masks throughout — no scalar extraction in the hot loop.
fn iterate_burning_ship_4x(cr: f64x4, ci: f64x4, max_iter: u32) -> [f64; 4] {
    let mut zr = f64x4::ZERO;
    let mut zi = f64x4::ZERO;
    let mut counts = f64x4::ZERO;
    let one = f64x4::splat(1.0);
    let threshold = f64x4::splat(65536.0);
    let two = f64x4::splat(2.0);
    let mut done_mask = f64x4::ZERO;

    for _ in 0..max_iter {
        let azr = zr.abs();
        let azi = zi.abs();
        let zr2 = azr * azr;
        let zi2 = azi * azi;
        let mag2 = zr2 + zi2;
        let just_escaped = mag2.cmp_gt(threshold);

        let active = !done_mask;
        counts = counts + active.blend(one, f64x4::ZERO);

        done_mask = done_mask | just_escaped;

        let done_bits: [u64; 4] = bytemuck::cast(done_mask);
        if done_bits[0] != 0 && done_bits[1] != 0 && done_bits[2] != 0 && done_bits[3] != 0 {
            break;
        }

        let new_zi = two * azr * azi + ci;
        let new_zr = zr2 - zi2 + cr;
        zr = new_zr;
        zi = new_zi;
    }

    let mag2_arr: [f64; 4] = bytemuck::cast(zr * zr + zi * zi);
    let counts_arr: [f64; 4] = bytemuck::cast(counts);
    let escaped_arr: [u64; 4] = bytemuck::cast(done_mask);
    let mut result = [0.0f64; 4];
    for k in 0..4 {
        if escaped_arr[k] != 0 && counts_arr[k] > 0.0 {
            let mag = mag2_arr[k].sqrt();
            result[k] = counts_arr[k] + 1.0 - mag.ln().ln() / std::f64::consts::LN_2;
        }
    }
    result
}

pub fn compute_burning_ship(
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
    let step4 = f64x4::from([0.0, 1.0, 2.0, 3.0]) * f64x4::splat(x_step);

    let mut result = vec![0.0f64; w * h];
    result
        .par_chunks_mut(w)
        .enumerate()
        .for_each(|(py, row)| {
            let ci = f64x4::splat(y_min + py as f64 * y_step);
            let mut px = 0;
            while px + 4 <= w {
                let base_cr = x_min + px as f64 * x_step;
                let cr = f64x4::splat(base_cr) + step4;
                let vals = iterate_burning_ship_4x(cr, ci, max_iter);
                row[px..px + 4].copy_from_slice(&vals);
                px += 4;
            }
            while px < w {
                let cr = x_min + px as f64 * x_step;
                row[px] = iterate_burning_ship(cr, y_min + py as f64 * y_step, max_iter);
                px += 1;
            }
        });
    result
}

pub fn compute_newton(
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

// ── Tricorn / Mandelbar ─────────────────────────────────────────────────────

pub fn compute_tricorn(
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
                row[px] = iterate_tricorn(cr, ci, max_iter);
            }
        });
    result
}

fn find_interesting_tricorn(rng: &mut Rng, max_iter: u32, width: u32, height: u32) -> FractalParams {
    // Tricorn boundary is similar to Mandelbrot but with 3-fold symmetry
    let mut best_params = FractalParams {
        center: (-0.3, 0.0),
        color_offset: rng.f64(),
        ..Default::default()
    };
    let mut best_score = 0.0f64;

    for _ in 0..80 {
        let angle = rng.range(0.0, std::f64::consts::TAU);
        let ray_len = 2.5;
        let outside = (ray_len * angle.cos(), ray_len * angle.sin());
        // Use a known inside point for Tricorn
        let inside_pt = (-0.2, 0.0);

        // Binary search for the Tricorn boundary
        let mut or = outside.0;
        let mut oi = outside.1;
        let mut ir = inside_pt.0;
        let mut ii = inside_pt.1;
        for _ in 0..64 {
            let mr = (or + ir) / 2.0;
            let mi = (oi + ii) / 2.0;
            if iterate_tricorn(mr, mi, max_iter) == 0.0 {
                ir = mr; ii = mi;
            } else {
                or = mr; oi = mi;
            }
        }
        let boundary = ((or + ir) / 2.0, (oi + ii) / 2.0);

        let zoom = 10.0f64.powf(rng.range(1.0, 3.5));
        let score = score_viewport(
            boundary, zoom, max_iter,
            &|x, y, mi| iterate_tricorn(x, y, mi),
            width, height,
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
        if best_score > 0.80 { break; }
    }
    best_params
}

// ── Phoenix Fractal ─────────────────────────────────────────────────────────

pub fn compute_phoenix(
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
                row[px] = iterate_phoenix(zr, zi, c.0, c.1, max_iter);
            }
        });
    result
}

fn find_interesting_phoenix(rng: &mut Rng, max_iter: u32, width: u32, height: u32) -> FractalParams {
    // Curated Phoenix constants known to produce interesting patterns
    let constants: [(f64, f64); 8] = [
        (0.5667, -0.5),
        (0.2, -0.5),
        (-0.5, 0.0),
        (0.56667, -0.5),
        (0.4, -0.3),
        (-0.4, 0.1),
        (0.3, -0.4),
        (0.56, -0.45),
    ];

    let mut best_params = FractalParams {
        julia_c: Some(constants[0]),
        color_offset: rng.f64(),
        ..Default::default()
    };
    let mut best_score = 0.0f64;

    for _ in 0..50 {
        let (cr, ci) = rng.choose(&constants);
        let c = (
            cr + rng.range(-0.05, 0.05),
            ci + rng.range(-0.05, 0.05),
        );
        let zoom = 10.0f64.powf(rng.range(0.0, 1.2));

        let score = score_viewport(
            (0.0, 0.0), zoom, max_iter,
            &|x, y, mi| iterate_phoenix(x, y, c.0, c.1, mi),
            width, height,
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
        if best_score > 0.80 { break; }
    }
    best_params
}

// ── Strange Attractor ───────────────────────────────────────────────────────

pub fn compute_attractor(
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

    // Prefer attractors whose aspect ratio is wider (fills ultra-wide better)
    let bbox_aspect = bbox_w / bbox_h;
    let aspect_score = if bbox_aspect > 2.0 {
        1.0 // Wide — great for ultra-wide displays
    } else if bbox_aspect > 1.0 {
        0.7
    } else {
        0.3 // Taller than wide — lots of wasted space on sides
    };

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
        _ => 0.05,
    };

    fill_score * 0.30 + spread_score * 0.40 + aspect_score * 0.30
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

pub fn compute_buddhabrot(
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

        // Metropolis-Hastings: start from a known escaping point near the boundary
        let mut cur_cr = -0.75;
        let mut cur_ci = 0.1;
        let step_size = 0.01;

        for sample_idx in 0..samples_per_thread {
            // Propose a new point (random walk from current, or uniform random every 100 steps)
            let (cr, ci) = if sample_idx % 100 == 0 {
                // Periodic uniform random to avoid getting stuck
                (rng.range(-2.0, 1.0), rng.range(-1.5, 1.5))
            } else {
                (
                    cur_cr + rng.range(-step_size, step_size),
                    cur_ci + rng.range(-step_size, step_size),
                )
            };

            // Skip points known to be inside the cardioid/bulb
            if in_cardioid_or_bulb(cr, ci) { continue; }

            // Check if point escapes
            let mut zr = 0.0;
            let mut zi = 0.0;
            let mut escaped = false;

            orbit.clear();
            for _ in 0..max_channel_iter {
                let zr2 = zr * zr;
                let zi2 = zi * zi;
                if zr2 + zi2 > 4.0 {
                    escaped = true;
                    break;
                }
                orbit.push((zr, zi));
                zi = 2.0 * zr * zi + ci;
                zr = zr2 - zi2 + cr;
            }

            if !escaped { continue; }

            // Accept the proposed point for future random walks
            cur_cr = cr;
            cur_ci = ci;

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

pub fn render_buddhabrot(
    r_data: &[f64], g_data: &[f64], b_data: &[f64],
    width: u32, height: u32,
    anchors: &[[u8; 3]], color_offset: f64,
) -> RgbImage {
    let cmap = build_colormap_from_anchors(anchors, 2048);
    let cmap_len = cmap.len();
    let offset = (color_offset * cmap_len as f64) as usize;
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
            let rn = (*r / r_norm).min(1.0);
            let gn = (*g / g_norm).min(1.0);
            let bn = (*b / b_norm).min(1.0);

            // Brightness from total density across all channels
            let brightness = ((rn + gn + bn) / 3.0).powf(gamma);

            // Color position from channel ratios — different iteration depths
            // produce different structural features, driving palette variation
            let total = rn + gn + bn;
            let color_pos = if total > 0.0 {
                // Weight channels to spread across palette:
                // deep iterations (r) -> low palette, shallow (b) -> high palette
                (rn * 0.0 + gn * 0.5 + bn * 1.0) / total
            } else {
                0.0
            };

            let color_idx = ((color_pos * cmap_len as f64) as usize + offset) % cmap_len;
            let base = cmap[color_idx];
            img.put_pixel(x, y, Rgb([
                (base[0] as f64 * brightness).min(255.0) as u8,
                (base[1] as f64 * brightness).min(255.0) as u8,
                (base[2] as f64 * brightness).min(255.0) as u8,
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

pub fn apply_variations(x: f64, y: f64, weights: &[f64; 10]) -> (f64, f64) {
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

/// Compute flame fractal, returning (density, color_index) per pixel.
/// Color index is the average transform color at each pixel.
pub fn compute_flame(
    width: u32, height: u32,
    transforms: &[FlameTransform],
    samples: u64,
    rng_seed: u64,
) -> (Vec<f64>, Vec<f64>) {
    // Pre-compute cumulative weights
    let total_weight: f64 = transforms.iter().map(|t| t.weight).sum();
    let cum_weights: Vec<f64> = transforms
        .iter()
        .scan(0.0, |acc, t| { *acc += t.weight / total_weight; Some(*acc) })
        .collect();

    let n = (width as usize) * (height as usize);
    let histogram = Histogram::new(width, height);
    // Accumulate color indices using atomic u64 (fixed-point: value * 1_000_000)
    let color_acc: Vec<AtomicU64> = (0..n).map(|_| AtomicU64::new(0)).collect();
    let num_threads = rayon::current_num_threads().max(1);
    let samples_per_thread = samples / num_threads as u64;

    (0..num_threads).into_par_iter().for_each(|tid| {
        let mut rng = Rng::new(rng_seed.wrapping_add((tid as u64).wrapping_mul(0x9e3779b97f4a7c15)));
        let mut x = rng.range(-1.0, 1.0);
        let mut y = rng.range(-1.0, 1.0);
        let mut color_idx = 0.5f64;

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

            // Blend color index with transform's color
            color_idx = (color_idx + t.color) / 2.0;

            // Divergence guard
            if !x.is_finite() || !y.is_finite() || x.abs() > 1e10 || y.abs() > 1e10 {
                x = rng.range(-1.0, 1.0);
                y = rng.range(-1.0, 1.0);
                color_idx = 0.5;
                continue;
            }

            if i < 20 { continue; } // Warmup

            // Map to pixel — flame coordinates typically in [-2, 2]
            let px = ((x + 2.5) / 5.0 * width as f64) as i64;
            let py = ((y + 1.5) / 3.0 * height as f64) as i64;
            if px >= 0 && px < width as i64 && py >= 0 && py < height as i64 {
                let idx = py as usize * width as usize + px as usize;
                histogram.increment(px as u32, py as u32);
                // Fixed-point color accumulation
                color_acc[idx].fetch_add((color_idx * 1_000_000.0) as u64, Ordering::Relaxed);
            }
        }
    });

    let density = tone_map_log(&histogram.to_vec_f64());
    let raw_density = histogram.to_vec_f64();

    // Compute average color index per pixel
    let color_map: Vec<f64> = color_acc
        .iter()
        .enumerate()
        .map(|(i, c)| {
            let count = raw_density[i];
            if count > 0.0 {
                (c.load(Ordering::Relaxed) as f64 / 1_000_000.0) / count
            } else {
                0.0
            }
        })
        .collect();

    (density, color_map)
}

/// Render flame fractal with per-pixel color blending from the palette.
pub fn render_flame(
    density: &[f64], color_map: &[f64],
    width: u32, height: u32,
    anchors: &[[u8; 3]], color_offset: f64,
) -> RgbImage {
    let cmap = build_colormap_from_anchors(anchors, 2048);
    let cmap_len = cmap.len();
    let offset = (color_offset * cmap_len as f64) as usize;

    let d_max = density
        .iter()
        .copied()
        .filter(|&v| v > 0.0)
        .fold(0.0f64, f64::max)
        .max(1.0);

    let mut img = RgbImage::new(width, height);
    for (i, (&d, &c)) in density.iter().zip(color_map.iter()).enumerate() {
        let x = (i % width as usize) as u32;
        let y = (i / width as usize) as u32;
        if d == 0.0 {
            img.put_pixel(x, y, Rgb([0, 0, 0]));
        } else {
            // Use color index to select palette position, brightness from density
            let brightness = (d / d_max).powf(0.5); // Gamma for visibility
            let color_idx = ((c * cmap_len as f64) as usize + offset) % cmap_len;
            let base = cmap[color_idx];
            img.put_pixel(x, y, Rgb([
                (base[0] as f64 * brightness).min(255.0) as u8,
                (base[1] as f64 * brightness).min(255.0) as u8,
                (base[2] as f64 * brightness).min(255.0) as u8,
            ]));
        }
    }
    img
}

pub fn random_flame_transform(rng: &mut Rng) -> FlameTransform {
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

/// Downsample an image by averaging NxN blocks of pixels.
pub fn downsample(img: &RgbImage, factor: u32) -> RgbImage {
    if factor <= 1 { return img.clone(); }
    let new_w = img.width() / factor;
    let new_h = img.height() / factor;
    let mut out = RgbImage::new(new_w, new_h);
    let f2 = (factor * factor) as u32;

    for ny in 0..new_h {
        for nx in 0..new_w {
            let mut r_sum = 0u32;
            let mut g_sum = 0u32;
            let mut b_sum = 0u32;
            for sy in 0..factor {
                for sx in 0..factor {
                    let p = img.get_pixel(nx * factor + sx, ny * factor + sy);
                    r_sum += p[0] as u32;
                    g_sum += p[1] as u32;
                    b_sum += p[2] as u32;
                }
            }
            out.put_pixel(nx, ny, Rgb([
                (r_sum / f2) as u8,
                (g_sum / f2) as u8,
                (b_sum / f2) as u8,
            ]));
        }
    }
    out
}

pub fn render(
    data: &[f64], width: u32, height: u32, anchors: &[[u8; 3]],
    color_offset: f64, cycle_factor: f64,
) -> RgbImage {
    let cmap = build_colormap_from_anchors(anchors, 2048);
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


// ── Public Library API ──────────────────────────────────────────────────────

/// Resolve a palette to concrete color anchors.
/// For `Palette::Random`, generates a new color-theory palette using the RNG.
/// For all other palettes, returns the preset anchors.
pub fn resolve_palette(palette: Palette, rng: &mut Rng) -> Vec<[u8; 3]> {
    if palette == Palette::Random {
        generate_random_palette(rng)
    } else {
        palette_anchors(palette)
    }
}

/// Find visually interesting parameters for a given fractal type.
/// Uses boundary-finding and viewport scoring to locate rich detail regions.
pub fn find_interesting_params(
    fractal: FractalType,
    width: u32,
    height: u32,
    max_iter: u32,
    rng: &mut Rng,
) -> FractalParams {
    match fractal {
        FractalType::Mandelbrot => find_interesting_mandelbrot(rng, max_iter, width, height),
        FractalType::Julia => find_interesting_julia(rng, max_iter, width, height),
        FractalType::BurningShip => find_interesting_burning_ship(rng, max_iter, width, height),
        FractalType::Newton => find_interesting_newton(rng, max_iter),
        FractalType::Tricorn => find_interesting_tricorn(rng, max_iter, width, height),
        FractalType::Phoenix => find_interesting_phoenix(rng, max_iter, width, height),
        FractalType::Flame => find_interesting_flame(rng),
        FractalType::Buddhabrot => find_interesting_buddhabrot(rng),
        FractalType::StrangeAttractor => find_interesting_attractor(rng),
    }
}

/// Generate a fractal image from pre-computed parameters.
///
/// This is the main high-level API. It handles:
/// - Computing the fractal data at the requested resolution (with optional supersampling)
/// - Rendering with the provided color anchors
/// - Downsampling if supersample > 1
///
/// Returns the final `RgbImage` ready to save or use.
pub fn generate(
    fractal: FractalType,
    params: &FractalParams,
    width: u32,
    height: u32,
    max_iter: u32,
    anchors: &[[u8; 3]],
    supersample: u32,
) -> RgbImage {
    let render_w = width * supersample;
    let render_h = height * supersample;
    let seed = params.color_offset.to_bits(); // deterministic seed from params

    let img = match fractal {
        FractalType::Mandelbrot => {
            let data = compute_mandelbrot(render_w, render_h, params.center, params.zoom, max_iter);
            render(&data, render_w, render_h, anchors, params.color_offset, 12.0)
        }
        FractalType::Julia => {
            let c = params.julia_c.unwrap_or((0.0, 0.0));
            let data = compute_julia(render_w, render_h, c, params.zoom, max_iter);
            render(&data, render_w, render_h, anchors, params.color_offset, 12.0)
        }
        FractalType::BurningShip => {
            let data = compute_burning_ship(render_w, render_h, params.center, params.zoom, max_iter);
            render(&data, render_w, render_h, anchors, params.color_offset, 12.0)
        }
        FractalType::Newton => {
            let data = compute_newton(render_w, render_h, params.center, params.zoom, max_iter.min(500));
            render(&data, render_w, render_h, anchors, params.color_offset, 12.0)
        }
        FractalType::Tricorn => {
            let data = compute_tricorn(render_w, render_h, params.center, params.zoom, max_iter);
            render(&data, render_w, render_h, anchors, params.color_offset, 12.0)
        }
        FractalType::Phoenix => {
            let c = params.julia_c.unwrap_or((0.5667, -0.5));
            let data = compute_phoenix(render_w, render_h, c, params.zoom, max_iter);
            render(&data, render_w, render_h, anchors, params.color_offset, 12.0)
        }
        FractalType::StrangeAttractor => {
            let (a, b, c, d) = params.attractor_params.unwrap_or((1.7, 1.7, 0.6, 1.2));
            let at = params.attractor_type.unwrap_or(AttractorType::Clifford);
            let data = compute_attractor(render_w, render_h, a, b, c, d, at, params.samples, seed);
            render(&data, render_w, render_h, anchors, params.color_offset, 2.0)
        }
        FractalType::Buddhabrot => {
            let (ri, gi, bi) = params.buddhabrot_iters.unwrap_or((5000, 500, 50));
            let (r, g, b) = compute_buddhabrot(render_w, render_h, params.samples, ri, gi, bi, seed);
            render_buddhabrot(&r, &g, &b, render_w, render_h, anchors, params.color_offset)
        }
        FractalType::Flame => {
            let transforms = params.flame_transforms.as_ref();
            if let Some(transforms) = transforms {
                let (density, color_map) = compute_flame(render_w, render_h, transforms, params.samples, seed);
                render_flame(&density, &color_map, render_w, render_h, anchors, params.color_offset)
            } else {
                // Empty flame — return black image
                RgbImage::new(render_w, render_h)
            }
        }
    };

    downsample(&img, supersample)
}
