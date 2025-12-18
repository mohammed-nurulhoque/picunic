//! Image to Unicode converter using CNN embeddings.

pub mod chunk;
pub mod dither;
pub mod embedding;

pub use chunk::ImageChunker;
pub use embedding::EmbeddingMatcher;

use thiserror::Error;

#[derive(Error, Debug)]
pub enum PicunicError {
    #[error("Image error: {0}")]
    Image(#[from] image::ImageError),
    #[error("Model error: {0}")]
    Model(String),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, PicunicError>;

/// Main converter using CNN embeddings
pub struct Converter {
    width: u32,
    matcher: EmbeddingMatcher,
    dither: bool,
}

impl Converter {
    pub fn new(
        model_path: impl AsRef<std::path::Path>,
        embeddings_path: impl AsRef<std::path::Path>,
        chars_path: impl AsRef<std::path::Path>,
    ) -> Result<Self> {
        let matcher = EmbeddingMatcher::new(model_path, embeddings_path, chars_path)?;
        Ok(Self { width: 80, matcher, dither: false })
    }

    pub fn with_width(mut self, width: u32) -> Self {
        self.width = width;
        self
    }

    pub fn with_dither(mut self, enabled: bool) -> Self {
        self.dither = enabled;
        self
    }

    pub fn ascii_only(mut self) -> Self {
        self.matcher.filter_ascii();
        self
    }

    pub fn convert(&mut self, image: &image::DynamicImage) -> String {
        let gray = image.to_luma8();
        let (img_w, img_h) = (gray.width(), gray.height());

        let out_w = self.width;
        // Terminal chars are ~1:2 aspect ratio
        let aspect = img_w as f32 / img_h as f32;
        let out_h = (out_w as f32 / aspect * 0.5).round().max(1.0) as u32;

        // Apply dithering if enabled
        // Scale = pixels per character (character-sized features)
        let gray = if self.dither {
            let pixels_per_char = img_w / out_w;
            let scale = pixels_per_char.max(1);
            dither::dither_atkinson(&gray, scale)
        } else {
            gray
        };

        let chunker = ImageChunker::new(gray, out_w, out_h);

        let mut rows = Vec::with_capacity(out_h as usize);
        for y in 0..out_h {
            let row: String = (0..out_w)
                .map(|x| self.matcher.find_best_match(&chunker.get_chunk(x, y)).unwrap_or(' '))
                .collect();
            rows.push(row);
        }

        rows.join("\n") + "\n"
    }
}
