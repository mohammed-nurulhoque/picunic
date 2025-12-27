//! picunic CLI - Convert images to Unicode art using CNN embeddings

use clap::Parser;
use picunic::{Converter, PicunicError};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "picunic", about = "Convert images to Unicode art")]
struct Args {
    /// Input image file
    input: PathBuf,
    /// Output width in characters
    #[arg(short, long, default_value = "80")]
    width: u32,
    /// Path to model directory
    #[arg(short, long, default_value = "assets")]
    model_dir: PathBuf,
    /// Invert the image
    #[arg(short, long)]
    invert: bool,
    /// Use only ASCII characters
    #[arg(short, long)]
    ascii: bool,
    /// Enable Atkinson dithering
    #[arg(short, long)]
    dither: bool,
    /// Edge weight for edge vs luminosity matching (0.0-1.0, default: 1.0 = pure edge)
    #[arg(long, default_value = "1.0")]
    edge_weight: f32,
}

fn main() -> Result<(), PicunicError> {
    let args = Args::parse();

    let mut converter = Converter::new(
        args.model_dir.join("encoder.onnx"),
        args.model_dir.join("encoder.embeddings.bin"),
        args.model_dir.join("encoder.chars.json"),
    )?
    .with_width(args.width)
    .with_dither(args.dither)
    .with_edge_weight(args.edge_weight);

    if args.ascii {
        converter = converter.ascii_only();
    }

    let mut image = image::open(&args.input)?;
    if args.invert {
        image.invert();
    }

    print!("{}", converter.convert(&image));
    Ok(())
}
