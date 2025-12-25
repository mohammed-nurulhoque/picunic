//! WebAssembly bindings for picunic

use wasm_bindgen::prelude::*;
use image::DynamicImage;
use crate::chunk;
use crate::dither;

#[wasm_bindgen]
pub struct WasmConverter {
    width: u32,
    char_embeddings: Vec<f32>,
    chars: Vec<char>,
    embedding_dim: usize,
    dither: bool,
    ascii_only: bool,
}

#[wasm_bindgen]
impl WasmConverter {
    /// Create a new converter with pre-loaded embeddings
    /// 
    /// # Arguments
    /// * `char_embeddings` - Flat array of embeddings: [char0_emb[0..dim], char1_emb[0..dim], ...]
    /// * `chars` - Array of characters corresponding to embeddings
    /// * `embedding_dim` - Dimension of each embedding vector
    #[wasm_bindgen(constructor)]
    pub fn new(
        char_embeddings: Vec<f32>,
        chars: Vec<String>,
        embedding_dim: usize,
    ) -> Result<WasmConverter, JsValue> {
        let chars: Vec<char> = chars
            .into_iter()
            .filter_map(|s| s.chars().next())
            .collect();
        
        if char_embeddings.len() != chars.len() * embedding_dim {
            return Err(JsValue::from_str("Embeddings length doesn't match chars * dim"));
        }

        Ok(WasmConverter {
            width: 80,
            char_embeddings,
            chars,
            embedding_dim,
            dither: false,
            ascii_only: false,
        })
    }

    #[wasm_bindgen]
    pub fn set_width(&mut self, width: u32) {
        self.width = width;
    }

    #[wasm_bindgen]
    pub fn set_dither(&mut self, enabled: bool) {
        self.dither = enabled;
    }

    #[wasm_bindgen]
    pub fn set_ascii_only(&mut self, enabled: bool) {
        self.ascii_only = enabled;
    }

    /// Process image and return chunk data for each position
    /// Returns a flat array: [chunk0_data..., chunk1_data..., ...] where each chunk is 128 floats (8x16)
    #[wasm_bindgen]
    pub fn process_image(&self, image_data: &[u8], width: u32, height: u32) -> Result<js_sys::Object, JsValue> {
        // Convert RGBA to DynamicImage
        let img = image::RgbaImage::from_raw(width, height, image_data.to_vec())
            .ok_or_else(|| JsValue::from_str("Invalid image dimensions"))?;
        let dynamic_img = DynamicImage::ImageRgba8(img);
        
        // Convert to grayscale
        let mut gray = dynamic_img.to_luma8();
        
        // Apply dithering if enabled
        if self.dither {
            let img_w = gray.width();
            let pixels_per_char = img_w / self.width;
            let scale = pixels_per_char.max(1);
            gray = dither::dither_atkinson(&gray, scale);
        }

        // Calculate output dimensions
        let (img_w, img_h) = (gray.width(), gray.height());
        let out_w = self.width;
        let aspect = img_w as f32 / img_h as f32;
        let out_h = (out_w as f32 / aspect * 0.5).round().max(1.0) as u32;

        // Create chunker
        let chunker = chunk::ImageChunker::new(gray, out_w, out_h);

        // Extract all chunks
        let mut chunks = Vec::new();
        for y in 0..out_h {
            for x in 0..out_w {
                chunks.push(chunker.get_chunk(x, y));
            }
        }

        // Return as object with chunks and dimensions
        let result = js_sys::Object::new();
        let chunks_array = js_sys::Array::new();
        for chunk in chunks {
            chunks_array.push(&js_sys::Float32Array::from(&chunk[..]));
        }
        js_sys::Reflect::set(&result, &"chunks".into(), &chunks_array)?;
        js_sys::Reflect::set(&result, &"width".into(), &(out_w as u32).into())?;
        js_sys::Reflect::set(&result, &"height".into(), &(out_h as u32).into())?;

        Ok(result)
    }

    /// Find best matching character for an embedding
    #[wasm_bindgen]
    pub fn find_best_char(&self, embedding: &[f32]) -> Result<String, JsValue> {
        let best_char = self.find_best_match(embedding)?;
        Ok(best_char.to_string())
    }

    fn find_best_match(&self, embedding: &[f32]) -> Result<char, JsValue> {
        if embedding.len() != self.embedding_dim {
            return Err(JsValue::from_str("Embedding dimension mismatch"));
        }

        // Cosine similarity (assuming embeddings are normalized)
        let mut best_idx = 0;
        let mut best_sim = f32::NEG_INFINITY;

        for (i, char_emb) in self.char_embeddings.chunks_exact(self.embedding_dim).enumerate() {
            if self.ascii_only {
                let c = self.chars[i];
                if (c as u32) < 0x20 || (c as u32) > 0x7E {
                    continue;
                }
            }

            let sim: f32 = embedding.iter()
                .zip(char_emb.iter())
                .map(|(a, b)| a * b)
                .sum();

            if sim > best_sim {
                best_sim = sim;
                best_idx = i;
            }
        }

        Ok(self.chars[best_idx])
    }
}


#[wasm_bindgen(start)]
pub fn init() {
    // WASM initialization
}
