//! Atkinson dithering for grayscale to binary conversion.

use image::GrayImage;

/// Atkinson dithering with automatic scale based on output resolution
/// Scale determines the block size for error diffusion
pub fn dither_atkinson(image: &GrayImage, scale: u32) -> GrayImage {
    let (orig_w, orig_h) = image.dimensions();
    let scale = scale.max(1);
    
    // Downsample if scale > 1
    let (work_w, work_h) = if scale > 1 {
        ((orig_w + scale - 1) / scale, (orig_h + scale - 1) / scale)
    } else {
        (orig_w, orig_h)
    };
    
    // Create downsampled working buffer (average pixels in each block)
    let mut errors: Vec<f32> = if scale > 1 {
        let mut buf = vec![0.0; (work_w * work_h) as usize];
        for by in 0..work_h {
            for bx in 0..work_w {
                let mut sum = 0.0;
                let mut count = 0;
                for dy in 0..scale {
                    for dx in 0..scale {
                        let x = bx * scale + dx;
                        let y = by * scale + dy;
                        if x < orig_w && y < orig_h {
                            sum += image.get_pixel(x, y).0[0] as f32;
                            count += 1;
                        }
                    }
                }
                buf[(by * work_w + bx) as usize] = sum / count as f32;
            }
        }
        buf
    } else {
        image.pixels().map(|p| p.0[0] as f32).collect()
    };
    
    // Apply Atkinson error diffusion at working resolution
    let mut dithered = vec![0u8; (work_w * work_h) as usize];
    
    for y in 0..work_h {
        for x in 0..work_w {
            let idx = (y * work_w + x) as usize;
            let old_val = errors[idx].clamp(0.0, 255.0);
            let new_val = if old_val > 127.5 { 255.0 } else { 0.0 };
            dithered[idx] = new_val as u8;
            
            // Atkinson: distribute 1/8 of error to 6 neighbors (total 6/8)
            // This loses some luminosity but increases contrast
            let e = (old_val - new_val) / 8.0;
            
            if x + 1 < work_w { errors[idx + 1] += e; }
            if x + 2 < work_w { errors[idx + 2] += e; }
            if y + 1 < work_h {
                let row = idx + work_w as usize;
                if x > 0 { errors[row - 1] += e; }
                errors[row] += e;
                if x + 1 < work_w { errors[row + 1] += e; }
            }
            if y + 2 < work_h {
                errors[idx + 2 * work_w as usize] += e;
            }
        }
    }
    
    // Upsample back to original resolution
    let mut output = GrayImage::new(orig_w, orig_h);
    for y in 0..orig_h {
        for x in 0..orig_w {
            let bx = (x / scale).min(work_w - 1);
            let by = (y / scale).min(work_h - 1);
            let val = dithered[(by * work_w + bx) as usize];
            output.put_pixel(x, y, image::Luma([val]));
        }
    }
    
    output
}
