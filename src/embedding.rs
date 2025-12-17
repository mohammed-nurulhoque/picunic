//! CNN embedding-based character matcher using ONNX runtime.

use crate::{PicunicError, Result};
use ndarray::{Array2, ArrayView1};
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Tensor;
use std::path::Path;

pub struct EmbeddingMatcher {
    session: Session,
    char_embeddings: Array2<f32>,
    chars: Vec<char>,
}

impl EmbeddingMatcher {
    pub fn new(
        model_path: impl AsRef<Path>,
        embeddings_path: impl AsRef<Path>,
        chars_path: impl AsRef<Path>,
    ) -> Result<Self> {
        let session = Session::builder()
            .map_err(|e| PicunicError::Model(e.to_string()))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| PicunicError::Model(e.to_string()))?
            .commit_from_file(model_path.as_ref())
            .map_err(|e| PicunicError::Model(e.to_string()))?;

        // Load character list and embedding dimension
        let json: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(chars_path)?)
            .map_err(|e| PicunicError::Model(e.to_string()))?;
        let chars: Vec<char> = json["chars"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|v| v.as_str()?.chars().next())
            .collect();
        let dim = json["embedding_dim"].as_u64().unwrap() as usize;

        // Load precomputed embeddings
        let bytes = std::fs::read(embeddings_path)?;
        let floats: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let char_embeddings = Array2::from_shape_vec((chars.len(), dim), floats)
            .map_err(|e| PicunicError::Model(e.to_string()))?;

        Ok(Self { session, char_embeddings, chars })
    }

    /// Filter to ASCII characters only (0x20-0x7E)
    pub fn filter_ascii(&mut self) {
        self.filter(|c| c as u32 >= 0x20 && c as u32 <= 0x7E);
    }

    /// Filter out emoji/colored characters, keep only monochrome-safe Unicode
    pub fn filter_monochrome(&mut self) {
        self.filter(|c| {
            let cp = c as u32;
            matches!(cp,
                0x0020..=0x007E |  // ASCII
                0x00A0..=0x00FF |  // Latin-1 Supplement
                0x2500..=0x257F |  // Box Drawing
                0x2580..=0x259F |  // Block Elements
                0x25A0..=0x25FF    // Geometric Shapes (mostly monochrome)
            )
        });
    }

    fn filter(&mut self, predicate: impl Fn(char) -> bool) {
        let mut indices = Vec::new();
        let mut new_chars = Vec::new();
        
        for (i, &c) in self.chars.iter().enumerate() {
            if predicate(c) {
                indices.push(i);
                new_chars.push(c);
            }
        }
        
        let dim = self.char_embeddings.ncols();
        let new_embeddings: Vec<f32> = indices.iter()
            .flat_map(|&i| self.char_embeddings.row(i).to_vec())
            .collect();
        
        self.char_embeddings = Array2::from_shape_vec((new_chars.len(), dim), new_embeddings)
            .expect("shape mismatch");
        self.chars = new_chars;
    }

    pub fn find_best_match(&mut self, chunk: &[f32]) -> Result<char> {
        // Input shape: (batch=1, channels=1, H=16, W=8)
        let input = Tensor::from_array(([1usize, 1, 16, 8], chunk.to_vec()))
            .map_err(|e| PicunicError::Model(e.to_string()))?;

        let outputs = self.session.run(ort::inputs![input])
            .map_err(|e| PicunicError::Model(e.to_string()))?;

        let emb = outputs[0].try_extract_tensor::<f32>()
            .map_err(|e| PicunicError::Model(e.to_string()))?;

        // Cosine similarity (embeddings are normalized)
        let emb_view = ArrayView1::from(emb.1);
        let sims = self.char_embeddings.dot(&emb_view);

        let (best_idx, _) = sims.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap();

        Ok(self.chars[best_idx])
    }
}
