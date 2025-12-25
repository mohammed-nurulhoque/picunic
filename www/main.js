// Main application logic for Picunic WebAssembly demo

import init, { WasmConverter } from './pkg/picunic.js';

let converter = null;
let onnxSession = null;

// Initialize the application
async function initApp() {
    const statusEl = document.getElementById('status');
    
    try {
        statusEl.textContent = 'Loading WebAssembly module...';
        statusEl.className = 'status loading';
        statusEl.style.display = 'block';
        
        // Initialize WASM module
        await init();
        
        statusEl.textContent = 'Loading ONNX model...';
        
        // ONNX Runtime Web should be loaded via script tag
        if (typeof ort === 'undefined') {
            throw new Error('ONNX Runtime Web not loaded. Make sure ort.min.js is included.');
        }
        
        // Load model as ArrayBuffer to avoid external data file resolution issues
        const modelResponse = await fetch('assets/encoder.onnx');
        const modelBuffer = await modelResponse.arrayBuffer();
        
        // Create session from buffer (this prevents ONNX Runtime Web from looking for external data files)
        onnxSession = await ort.InferenceSession.create(modelBuffer, {
            executionProviders: ['wasm'],
        });
        
        statusEl.textContent = 'Loading embeddings and character data...';
        
        // Load embeddings and chars
        const [embeddingsResponse, charsResponse] = await Promise.all([
            fetch('assets/encoder.embeddings.bin'),
            fetch('assets/encoder.chars.json'),
        ]);
        
        const embeddingsBuffer = await embeddingsResponse.arrayBuffer();
        const charsData = await charsResponse.json();
        
        // Parse embeddings (little-endian f32)
        const embeddingsArray = new Float32Array(embeddingsBuffer);
        const embeddings = Array.from(embeddingsArray);
        
        // Create converter
        converter = new WasmConverter(
            embeddings,
            charsData.chars,
            charsData.embedding_dim
        );
        
        statusEl.textContent = 'Ready! Upload an image to get started.';
        statusEl.className = 'status success';
        
        // Enable convert button
        document.getElementById('convert-btn').disabled = false;
        
    } catch (error) {
        console.error('Initialization error:', error);
        statusEl.textContent = `Error: ${error.message}`;
        statusEl.className = 'status error';
    }
}

// Get embedding for a chunk using ONNX Runtime
async function getEmbedding(chunkData) {
    if (!onnxSession) {
        throw new Error('ONNX session not initialized');
    }
    
    // Prepare input tensor: shape [1, 1, 16, 8]
    const inputTensor = new ort.Tensor('float32', chunkData, [1, 1, 16, 8]);
    const feeds = { [onnxSession.inputNames[0]]: inputTensor };
    
    const results = await onnxSession.run(feeds);
    const output = results[onnxSession.outputNames[0]];
    
    // Return embedding as array
    return Array.from(output.data);
}

// Handle image upload
document.getElementById('image-input').addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (!file) return;
    
    const reader = new FileReader();
    reader.onload = (event) => {
        const img = new Image();
        img.onload = () => {
            const preview = document.getElementById('image-preview');
            preview.innerHTML = `<img src="${event.target.result}" alt="Preview">`;
        };
        img.src = event.target.result;
    };
    reader.readAsDataURL(file);
});

// Handle convert button
document.getElementById('convert-btn').addEventListener('click', async () => {
    const fileInput = document.getElementById('image-input');
    const file = fileInput.files[0];
    
    if (!file || !converter) {
        alert('Please select an image first');
        return;
    }
    
    const statusEl = document.getElementById('status');
    const outputEl = document.getElementById('output');
    const copyBtn = document.getElementById('copy-btn');
    
    try {
        statusEl.textContent = 'Processing image...';
        statusEl.className = 'status loading';
        statusEl.style.display = 'block';
        
        // Load and process image
        const img = await loadImage(file);
        const imageData = getImageData(img);
        
        // Get settings
        const width = parseInt(document.getElementById('width-input').value);
        const dither = document.getElementById('dither-checkbox').checked;
        const asciiOnly = document.getElementById('ascii-checkbox').checked;
        
        converter.set_width(width);
        converter.set_dither(dither);
        converter.set_ascii_only(asciiOnly);
        
        // Process image to get chunks
        const processed = converter.process_image(imageData, img.width, img.height);
        const chunks = processed.chunks;
        const outWidth = processed.width;
        const outHeight = processed.height;
        
        // Process each chunk: get embedding from ONNX, then find best char
        const rows = [];
        for (let y = 0; y < outHeight; y++) {
            let row = '';
            for (let x = 0; x < outWidth; x++) {
                const chunkIdx = y * outWidth + x;
                const chunk = Array.from(chunks[chunkIdx]);
                
                // Get embedding from ONNX
                const embedding = await getEmbedding(chunk);
                
                // Find best matching character
                const bestChar = converter.find_best_char(embedding);
                row += bestChar;
            }
            rows.push(row);
        }
        
        const result = rows.join('\n') + '\n';
        outputEl.textContent = result;
        copyBtn.style.display = 'block';
        
        statusEl.textContent = 'Conversion complete!';
        statusEl.className = 'status success';
        
    } catch (error) {
        console.error('Conversion error:', error);
        statusEl.textContent = `Error: ${error.message}`;
        statusEl.className = 'status error';
    }
});

// Handle copy button
document.getElementById('copy-btn').addEventListener('click', () => {
    const output = document.getElementById('output').textContent;
    navigator.clipboard.writeText(output).then(() => {
        const btn = document.getElementById('copy-btn');
        const originalText = btn.textContent;
        btn.textContent = 'Copied!';
        setTimeout(() => {
            btn.textContent = originalText;
        }, 2000);
    });
});

// Helper: Load image from file
function loadImage(file) {
    return new Promise((resolve, reject) => {
        const img = new Image();
        img.onload = () => resolve(img);
        img.onerror = reject;
        img.src = URL.createObjectURL(file);
    });
}

// Helper: Get RGBA pixel data from image
function getImageData(img) {
    const canvas = document.createElement('canvas');
    canvas.width = img.width;
    canvas.height = img.height;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(img, 0, 0);
    const imageData = ctx.getImageData(0, 0, img.width, img.height);
    return imageData.data;
}

// Initialize on page load
initApp();
