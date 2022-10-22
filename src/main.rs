use std::path::PathBuf;

use clap::Parser;
mod alphas;
mod autoencoder_kl;
mod clip_encoder;
mod clip_tokenizer;
mod diffusion_model;
mod stable_diffusion;

/// Rust port of stable-diffusion-tensorflow
#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    /// The prompt to render
    prompt: String,

    /// The input image path (do not use if txt2img)
    #[arg(short, long)]
    input: Option<PathBuf>,

    /// The input image mask path for inpainting (do not use if txt2img)
    #[arg(long)]
    input_mask: Option<PathBuf>,

    /// Where to save the output image (defaults to <prompt_with_underscores>.png)
    #[arg(short, long)]
    output: Option<String>,

    /// Image height, in pixels
    #[arg(long, default_value_t = 512)]
    height: u16,

    /// Image width, in pixels
    #[arg(long, default_value_t = 512)]
    width: u16,

    /// Unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))
    #[arg(long, default_value_t = 7.5)]
    scale: f32,

    /// Number of DDIM sampling steps
    #[arg(short, long, default_value_t = 50)]
    steps: u8,

    /// Optionally specify a seed for reproducible results
    #[arg(long)]
    seed: Option<u32>,

    /// Enable mixed precision (fp16 computation)
    #[arg(short, long)]
    mixed_precision: bool,
}

fn main() {
    let cli = Cli::parse();
    let generator = stable_diffusion::StableDiffusion::new(cli.height, cli.width);
    generator.generate(
        cli.prompt,
        cli.steps,
        cli.scale,
        1,
        cli.seed,
        cli.input,
        cli.input_mask,
        0.5,
    );
}
