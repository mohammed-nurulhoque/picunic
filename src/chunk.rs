//! Image chunking - splits image into 8×16 patches for character matching.

use image::GrayImage;

// Terminal cell aspect ratio 1:2
const CHUNK_W: usize = 8;
const CHUNK_H: usize = 16;

pub struct ImageChunker {
    image: GrayImage,
    chunk_w: f32,
    chunk_h: f32,
}

impl ImageChunker {
    pub fn new(image: GrayImage, cols: u32, rows: u32) -> Self {
        let chunk_w = image.width() as f32 / cols as f32;
        let chunk_h = image.height() as f32 / rows as f32;
        Self { image, chunk_w, chunk_h }
    }

    /// Extract chunk at (col, row), resized to CHUNK_W × CHUNK_H
    pub fn get_chunk(&self, col: u32, row: u32) -> Vec<f32> {
        let x0 = (col as f32 * self.chunk_w) as u32;
        let y0 = (row as f32 * self.chunk_h) as u32;
        let x1 = ((col + 1) as f32 * self.chunk_w).ceil() as u32;
        let y1 = ((row + 1) as f32 * self.chunk_h).ceil() as u32;

        let (x1, y1) = (x1.min(self.image.width()), y1.min(self.image.height()));
        let (cw, ch) = ((x1 - x0).max(1), (y1 - y0).max(1));

        let mut result = vec![0.0; CHUNK_W * CHUNK_H];
        for ty in 0..CHUNK_H {
            for tx in 0..CHUNK_W {
                let sx = x0 + (tx as f32 / CHUNK_W as f32 * cw as f32) as u32;
                let sy = y0 + (ty as f32 / CHUNK_H as f32 * ch as f32) as u32;
                let sx = sx.min(self.image.width() - 1);
                let sy = sy.min(self.image.height() - 1);
                result[ty * CHUNK_W + tx] = self.image.get_pixel(sx, sy).0[0] as f32 / 255.0;
            }
        }
        result
    }
}
