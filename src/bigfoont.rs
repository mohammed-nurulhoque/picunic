//! bigfoont - Render text as large Unicode art using picunic embeddings

use clap::Parser;
use image::{GrayImage, Luma};
use fontdue::{Font, FontSettings};
use picunic::{EmbeddingMatcher, PicunicError};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "bigfoont", about = "Render text as large Unicode art")]
struct Args {
    /// Text to render
    text: String,
    /// Output width per character (in Unicode chars)
    #[arg(short, long, default_value = "2")]
    width: u32,
    /// Output height per character (in Unicode chars)
    #[arg(short = 'H', long, default_value = "2")]
    height: u32,
    /// Font file for rendering source text
    #[arg(short, long, default_value = "assets/DejaVuSansMono.ttf")]
    font: PathBuf,
    /// Path to model directory
    #[arg(short, long, default_value = "assets")]
    model_dir: PathBuf,
    /// Use only ASCII output characters
    #[arg(short, long)]
    ascii: bool,
}

const CHUNK_W: u32 = 8;
const CHUNK_H: u32 = 16;

fn main() -> Result<(), PicunicError> {
    let args = Args::parse();

    // Load font for rendering input chars
    let font_data = std::fs::read(&args.font)?;
    let font = Font::from_bytes(font_data, FontSettings::default())
        .map_err(|e| PicunicError::Model(e.to_string()))?;

    // Load embedding matcher
    let mut matcher = EmbeddingMatcher::new(
        args.model_dir.join("encoder.onnx"),
        args.model_dir.join("encoder.embeddings.bin"),
        args.model_dir.join("encoder.chars.json"),
    )?;

    if args.ascii {
        matcher.filter_ascii();
    } else {
        matcher.filter_monochrome();
    }

    // Dimensions per source char
    let render_w = CHUNK_W * args.width;
    let render_h = CHUNK_H * args.height;
    let font_size = render_h as f32 * 0.875; // ~87.5% to fit with baseline

    // Render each char and convert to Unicode grid
    let mut output_rows: Vec<String> = vec![String::new(); args.height as usize];

    for ch in args.text.chars() {
        let grid = render_char_to_grid(&font, ch, render_w, render_h, font_size, &mut matcher)?;
        for (i, row) in grid.iter().enumerate() {
            output_rows[i].push_str(row);
        }
    }

    for row in output_rows {
        println!("{}", row);
    }

    Ok(())
}

fn render_char_to_grid(
    font: &Font,
    ch: char,
    width: u32,
    height: u32,
    font_size: f32,
    matcher: &mut EmbeddingMatcher,
) -> Result<Vec<String>, PicunicError> {
    // Render char to grayscale image
    let img = render_char(font, ch, width, height, font_size);

    // Chunk and match
    let cols = width / CHUNK_W;
    let rows = height / CHUNK_H;

    let mut grid = Vec::with_capacity(rows as usize);
    for row in 0..rows {
        let mut line = String::with_capacity(cols as usize);
        for col in 0..cols {
            let chunk = extract_chunk(&img, col * CHUNK_W, row * CHUNK_H, CHUNK_W, CHUNK_H);
            let matched = matcher.find_best_match(&chunk)?;
            line.push(matched);
        }
        grid.push(line);
    }

    Ok(grid)
}

fn render_char(font: &Font, ch: char, width: u32, height: u32, font_size: f32) -> GrayImage {
    let mut img = GrayImage::new(width, height);

    let (metrics, bitmap) = font.rasterize(ch, font_size);

    if metrics.width == 0 || metrics.height == 0 {
        return img;
    }

    // Baseline at ~75% down
    let baseline_y = (height as f32 * 0.75) as i32;
    let y_offset = baseline_y - metrics.height as i32 - metrics.ymin;

    // Center horizontally
    let x_offset = (width as i32 - metrics.width as i32) / 2;

    for sy in 0..metrics.height {
        for sx in 0..metrics.width {
            let tx = x_offset + sx as i32;
            let ty = y_offset + sy as i32;
            if tx >= 0 && tx < width as i32 && ty >= 0 && ty < height as i32 {
                let val = bitmap[sy * metrics.width + sx];
                img.put_pixel(tx as u32, ty as u32, Luma([val]));
            }
        }
    }

    img
}

fn extract_chunk(img: &GrayImage, x: u32, y: u32, w: u32, h: u32) -> Vec<f32> {
    let mut chunk = Vec::with_capacity((w * h) as usize);
    for ty in 0..h {
        for tx in 0..w {
            let px = img.get_pixel(x + tx, y + ty).0[0];
            chunk.push(px as f32 / 255.0);
        }
    }
    chunk
}
