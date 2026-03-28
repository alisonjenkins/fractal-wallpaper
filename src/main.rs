use clap::{Parser, ValueEnum};
use fractal_wallpaper::*;
use std::path::PathBuf;
use std::time::Instant;

// ── CLI ─────────────────────────────────────────────────────────────────────

/// Wrapper enum that adds `All` variant for CLI use only.
#[derive(Clone, Copy, ValueEnum)]
enum CliFractalType {
    Mandelbrot,
    Julia,
    BurningShip,
    Newton,
    Flame,
    Buddhabrot,
    StrangeAttractor,
    Tricorn,
    Phoenix,
    All,
}

impl CliFractalType {
    fn to_lib(self) -> Option<FractalType> {
        match self {
            Self::Mandelbrot => Some(FractalType::Mandelbrot),
            Self::Julia => Some(FractalType::Julia),
            Self::BurningShip => Some(FractalType::BurningShip),
            Self::Newton => Some(FractalType::Newton),
            Self::Flame => Some(FractalType::Flame),
            Self::Buddhabrot => Some(FractalType::Buddhabrot),
            Self::StrangeAttractor => Some(FractalType::StrangeAttractor),
            Self::Tricorn => Some(FractalType::Tricorn),
            Self::Phoenix => Some(FractalType::Phoenix),
            Self::All => None, // Handled as a loop in main
        }
    }
}

/// Wrapper enum to bridge clap's ValueEnum with the library Palette.
#[derive(Clone, Copy, ValueEnum)]
enum CliPalette {
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

impl CliPalette {
    fn to_lib(self) -> Palette {
        match self {
            Self::Twilight => Palette::Twilight,
            Self::Ocean => Palette::Ocean,
            Self::Fire => Palette::Fire,
            Self::Neon => Palette::Neon,
            Self::Frost => Palette::Frost,
            Self::Earth => Palette::Earth,
            Self::Sakura => Palette::Sakura,
            Self::CatppuccinMocha => Palette::CatppuccinMocha,
            Self::CatppuccinMacchiato => Palette::CatppuccinMacchiato,
            Self::CatppuccinFrappe => Palette::CatppuccinFrappe,
            Self::CatppuccinLatte => Palette::CatppuccinLatte,
            Self::Random => Palette::Random,
        }
    }
}

#[derive(Parser)]
#[command(name = "fractal-wallpaper", about = "Fractal wallpaper generator")]
struct Cli {
    /// Fractal type to generate
    #[arg(default_value = "mandelbrot")]
    fractal: CliFractalType,

    /// Color palette (random if not specified)
    #[arg(short, long)]
    palette: Option<CliPalette>,

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

    /// Image width in pixels
    #[arg(long, default_value_t = DEFAULT_WIDTH)]
    width: u32,

    /// Image height in pixels
    #[arg(long, default_value_t = DEFAULT_HEIGHT)]
    height: u32,

    /// Supersampling factor for anti-aliasing (2 = render at 2x, then downsample)
    #[arg(long, default_value_t = 1)]
    supersample: u32,

    /// Save parameters to JSON file for later reproduction
    #[arg(long)]
    save_params: Option<PathBuf>,

    /// Load parameters from JSON file (overrides fractal type and randomization)
    #[arg(long)]
    load_params: Option<PathBuf>,

    /// Load custom palette from JSON file (array of [R,G,B] arrays, values 0-255)
    #[arg(long)]
    palette_file: Option<PathBuf>,
}

// ── Helpers ─────────────────────────────────────────────────────────────────

fn generate_and_save(
    fractal: FractalType,
    palette: Palette,
    max_iter: u32,
    samples: Option<u64>,
    width: u32,
    height: u32,
    supersample: u32,
    custom_palette: Option<&Vec<[u8; 3]>>,
    palette_name: &str,
    output: Option<PathBuf>,
    save_params: Option<&PathBuf>,
    rng: &mut Rng,
) {
    let seed = rng.next_u64();
    let t_search = Instant::now();
    let mut params = find_interesting_params(fractal, width, height, max_iter, rng);
    let search_time = t_search.elapsed();

    // Override samples if provided via CLI
    if let Some(s) = samples {
        params.samples = s;
    }

    // Resolve palette anchors: custom file > random generation > preset
    if let Some(custom) = custom_palette {
        params.random_palette = Some(custom.clone());
    } else if palette == Palette::Random && params.random_palette.is_none() {
        params.random_palette = Some(generate_random_palette(rng));
    }

    let anchors = if let Some(ref custom) = params.random_palette {
        custom.clone()
    } else {
        palette_anchors(palette)
    };

    let out_path = output.unwrap_or_else(|| {
        PathBuf::from(format!("fractal_{fractal}_{palette_name}_{width}x{height}.png"))
    });

    println!(
        "Generating {fractal} ({width}x{height}, palette={palette_name}, max_iter={max_iter})"
    );
    println!(
        "  Found interesting params in {:.2}s",
        search_time.as_secs_f64(),
    );

    // Save params if requested
    if let Some(path) = save_params {
        let saved = SavedParams {
            fractal,
            palette,
            max_iter,
            seed,
            params: FractalParams {
                center: params.center,
                zoom: params.zoom,
                julia_c: params.julia_c,
                color_offset: params.color_offset,
                attractor_params: params.attractor_params,
                attractor_type: params.attractor_type,
                buddhabrot_iters: params.buddhabrot_iters,
                flame_transforms: params.flame_transforms.clone(),
                samples: params.samples,
                random_palette: params.random_palette.clone(),
            },
        };
        let json = serde_json::to_string_pretty(&saved).expect("Failed to serialize params");
        std::fs::write(path, json).expect("Failed to write params file");
        println!("  Saved params to {}", path.display());
    }

    if supersample > 1 {
        println!("  Rendering at {}x{} (supersample {supersample}x)", width * supersample, height * supersample);
    }

    let t0 = Instant::now();
    let img = generate(fractal, &params, width, height, max_iter, &anchors, supersample);
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

// ── Main ────────────────────────────────────────────────────────────────────

fn main() {
    let cli = Cli::parse();

    // Load params from file if requested
    if let Some(ref load_path) = cli.load_params {
        let json = std::fs::read_to_string(load_path)
            .unwrap_or_else(|e| { eprintln!("Error reading params file {}: {e}", load_path.display()); std::process::exit(1); });
        let saved: SavedParams = serde_json::from_str(&json)
            .unwrap_or_else(|e| { eprintln!("Error parsing params file: {e}"); std::process::exit(1); });
        println!("Loaded params from {}", load_path.display());
        println!("Seed: {}", saved.seed);

        let anchors = if let Some(ref custom) = saved.params.random_palette {
            custom.clone()
        } else {
            palette_anchors(saved.palette)
        };

        let out_path = cli.output.unwrap_or_else(|| {
            PathBuf::from(format!(
                "fractal_{}_{}_{}x{}.png",
                saved.fractal, saved.palette, cli.width, cli.height
            ))
        });

        if cli.supersample > 1 {
            println!("  Rendering at {}x{} (supersample {}x)", cli.width * cli.supersample, cli.height * cli.supersample, cli.supersample);
        }

        let t0 = Instant::now();
        let img = generate(saved.fractal, &saved.params, cli.width, cli.height, saved.max_iter, &anchors, cli.supersample);
        let t1 = Instant::now();
        println!("  Computed in {:.2}s", (t1 - t0).as_secs_f64());

        img.save(&out_path).expect("Failed to save image");
        println!("  Saved to {}", out_path.display());
        return;
    }

    let seed = cli.seed.unwrap_or_else(|| {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64
    });
    println!("Seed: {seed}");
    let mut rng = Rng::new(seed);

    // Load custom palette from file if specified
    let custom_palette: Option<Vec<[u8; 3]>> = cli.palette_file.as_ref().map(|path| {
        let json = std::fs::read_to_string(path)
            .unwrap_or_else(|e| { eprintln!("Error reading palette file {}: {e}", path.display()); std::process::exit(1); });
        let anchors: Vec<[u8; 3]> = serde_json::from_str(&json)
            .unwrap_or_else(|e| { eprintln!("Error parsing palette file: {e}\nExpected JSON array of [R,G,B] arrays, e.g. [[255,0,0],[0,255,0],...]"); std::process::exit(1); });
        if anchors.len() < 2 {
            eprintln!("Palette file must contain at least 2 color anchors");
            std::process::exit(1);
        }
        println!("Loaded custom palette from {} ({} anchors)", path.display(), anchors.len());
        anchors
    });

    // Determine palette display name
    let palette_name_override = cli.palette_file.as_ref().map(|path| {
        path.file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("custom")
            .to_string()
    });

    match cli.fractal.to_lib() {
        None => {
            // All types
            for &ft in ALL_FRACTAL_TYPES {
                let pal = cli.palette.map(|p| p.to_lib()).unwrap_or_else(|| rng.choose(ALL_PALETTES));
                let name = palette_name_override.clone().unwrap_or_else(|| pal.to_string());
                generate_and_save(ft, pal, cli.max_iter, cli.samples, cli.width, cli.height, cli.supersample, custom_palette.as_ref(), &name, None, cli.save_params.as_ref(), &mut rng);
            }
        }
        Some(ft) => {
            let pal = cli.palette.map(|p| p.to_lib()).unwrap_or_else(|| rng.choose(ALL_PALETTES));
            let name = palette_name_override.clone().unwrap_or_else(|| pal.to_string());
            generate_and_save(ft, pal, cli.max_iter, cli.samples, cli.width, cli.height, cli.supersample, custom_palette.as_ref(), &name, cli.output, cli.save_params.as_ref(), &mut rng);
        }
    }
}
