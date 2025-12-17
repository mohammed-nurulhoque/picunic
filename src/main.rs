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
    /// Output height in characters (auto if not specified)
    #[arg(short = 'H', long)]
    height: Option<u32>,
    /// Path to model directory (default: assets/)
    #[arg(short, long, default_value = "assets")]
    model_dir: PathBuf,
    /// Invert the output
    #[arg(short, long)]
    invert: bool,
    /// Use only ASCII characters (0x20-0x7E)
    #[arg(short, long)]
    ascii: bool,
    /// Include all characters (including emoji/colored)
    #[arg(long)]
    all: bool,
}

fn main() -> Result<(), PicunicError> {
    let args = Args::parse();

    let mut converter = Converter::new(
        args.model_dir.join("encoder.onnx"),
        args.model_dir.join("encoder.embeddings.bin"),
        args.model_dir.join("encoder.chars.json"),
    )?
    .with_width(args.width);

    if let Some(h) = args.height {
        converter = converter.with_height(h);
    }

    if args.ascii {
        converter = converter.ascii_only();
    } else if !args.all {
        converter = converter.monochrome_only();
    }

    let mut image = image::open(&args.input)?;
    if args.invert {
        image.invert();
    }

    print!("{}", converter.convert(&image));
    Ok(())
}
