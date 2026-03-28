use clap::{Parser, ValueEnum};
use image::{Rgb, RgbImage};
use rayon::prelude::*;
use std::path::PathBuf;
use std::time::Instant;

const WIDTH: u32 = 5440;
const HEIGHT: u32 = 1440;

// ── CLI ─────────────────────────────────────────────────────────────────────

#[derive(Parser)]
#[command(name = "fractal-wallpaper", about = "Fractal wallpaper generator (5440×1440)")]
struct Cli {
    /// Fractal type to generate
    #[arg(default_value = "mandelbrot")]
    fractal: FractalType,

    /// Color palette
    #[arg(short, long)]
    palette: Option<Palette>,

    /// Maximum iterations
    #[arg(short, long, default_value_t = 1500)]
    max_iter: u32,

    /// Output file path
    #[arg(short, long)]
    output: Option<PathBuf>,
}

#[derive(Clone, Copy, ValueEnum)]
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

fn default_palette(fractal: FractalType) -> Palette {
    match fractal {
        FractalType::Mandelbrot => Palette::Twilight,
        FractalType::Julia => Palette::Neon,
        FractalType::BurningShip => Palette::Fire,
        FractalType::Newton => Palette::Ocean,
        FractalType::All => Palette::Twilight,
    }
}

// ── Fractal Computation ─────────────────────────────────────────────────────

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
                row[px] = if i < max_iter {
                    let mag = (zr * zr + zi * zi).sqrt();
                    i as f64 + 1.0 - mag.ln().ln() / std::f64::consts::LN_2
                } else {
                    0.0
                };
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
                let mut zr = x_min + px as f64 * x_step;
                let mut zi = y_min + py as f64 * y_step;
                let mut i = 0u32;
                while i < max_iter {
                    let zr2 = zr * zr;
                    let zi2 = zi * zi;
                    if zr2 + zi2 > 65536.0 {
                        break;
                    }
                    let new_zr = zr2 - zi2 + c.0;
                    zi = 2.0 * zr * zi + c.1;
                    zr = new_zr;
                    i += 1;
                }
                row[px] = if i < max_iter {
                    let mag = (zr * zr + zi * zi).sqrt();
                    i as f64 + 1.0 - mag.ln().ln() / std::f64::consts::LN_2
                } else {
                    0.0
                };
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
                row[px] = if i < max_iter {
                    let mag = (zr * zr + zi * zi).sqrt();
                    i as f64 + 1.0 - mag.ln().ln() / std::f64::consts::LN_2
                } else {
                    0.0
                };
            }
        });
    result
}

fn compute_newton(
    width: u32, height: u32,
    zoom: f64, max_iter: u32,
) -> Vec<f64> {
    let w = width as usize;
    let h = height as usize;
    let aspect = width as f64 / height as f64;
    let x_range = 4.0 / zoom;
    let y_range = x_range / aspect;
    let x_min = -x_range / 2.0;
    let y_min = -y_range / 2.0;
    let x_step = x_range / width as f64;
    let y_step = y_range / height as f64;

    // Roots of z^3 - 1 = 0
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
                    // f(z) = z^3 - 1, f'(z) = 3z^2
                    let zr2 = zr * zr;
                    let zi2 = zi * zi;

                    // z^2
                    let z2r = zr2 - zi2;
                    let z2i = 2.0 * zr * zi;

                    // z^3
                    let z3r = z2r * zr - z2i * zi;
                    let z3i = z2r * zi + z2i * zr;

                    // f'(z) = 3z^2
                    let dr = 3.0 * z2r;
                    let di = 3.0 * z2i;

                    // Complex division: (z^3 - 1) / (3z^2)
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

fn render(data: &[f64], width: u32, height: u32, palette: Palette) -> RgbImage {
    let cmap = build_colormap(palette, 2048);
    let cmap_len = cmap.len();

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
                let idx = ((normed * 12.0 * cmap_len as f64) % cmap_len as f64) as usize;
                let idx = idx.min(cmap_len - 1);
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

fn generate(fractal: FractalType, palette: Palette, max_iter: u32, output: Option<PathBuf>) {
    let out_path = output.unwrap_or_else(|| {
        PathBuf::from(format!("fractal_{fractal}_{palette}_{WIDTH}x{HEIGHT}.png"))
    });

    println!(
        "Generating {fractal} ({WIDTH}x{HEIGHT}, palette={palette}, max_iter={max_iter})..."
    );

    let t0 = Instant::now();
    let data = match fractal {
        FractalType::Mandelbrot => {
            compute_mandelbrot(WIDTH, HEIGHT, (-0.75, 0.0), 1.0, max_iter)
        }
        FractalType::Julia => {
            compute_julia(WIDTH, HEIGHT, (-0.7269, 0.1889), 1.0, max_iter)
        }
        FractalType::BurningShip => {
            compute_burning_ship(WIDTH, HEIGHT, (-1.75, -0.04), 30.0, max_iter)
        }
        FractalType::Newton => {
            compute_newton(WIDTH, HEIGHT, 1.0, max_iter.min(500))
        }
        FractalType::All => unreachable!(),
    };
    let t1 = Instant::now();
    println!("  Computed in {:.2}s", (t1 - t0).as_secs_f64());

    let img = render(&data, WIDTH, HEIGHT, palette);
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

    match cli.fractal {
        FractalType::All => {
            let types = [
                FractalType::Mandelbrot,
                FractalType::Julia,
                FractalType::BurningShip,
                FractalType::Newton,
            ];
            for ft in types {
                let pal = cli.palette.unwrap_or_else(|| default_palette(ft));
                generate(ft, pal, cli.max_iter, None);
            }
        }
        ft => {
            let pal = cli.palette.unwrap_or_else(|| default_palette(ft));
            generate(ft, pal, cli.max_iter, cli.output);
        }
    }
}
