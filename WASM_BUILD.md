# Building Picunic for WebAssembly

This guide explains how to build and serve the Picunic WebAssembly application.

## Prerequisites

1. **Rust** - Install from [rustup.rs](https://rustup.rs/)
2. **wasm-pack** - Install with: `cargo install wasm-pack`
3. **Python 3** (or any HTTP server) - For serving the web app

## Build Steps

1. **Build the WebAssembly module:**

   From the project root directory:
   
   ```bash
   wasm-pack build --target web --out-dir www/pkg
   ```

   Or use the build script (recommended):
   
   ```bash
   ./build-wasm.sh
   ```

   This will compile the Rust code to WebAssembly and generate the JavaScript bindings in `www/pkg/`.

2. **Copy assets:**

   The ONNX model and related files need to be accessible from the web server:
   
   ```bash
   # From project root
   mkdir -p www/assets
   cp assets/encoder.onnx www/assets/
   cp assets/encoder.embeddings.bin www/assets/
   cp assets/encoder.chars.json www/assets/
   ```

   Or the build script does this automatically.

   **Note:** If your model has external data (`.onnx.data` file), you must merge it first:
   
   ```bash
   # Requires: pip install onnx (or use training/.venv/bin/python)
   python3 merge_onnx_data.py assets/encoder.onnx www/assets/encoder.onnx
   ```
   
   ONNX Runtime Web doesn't support external data files, so the model must be self-contained.

3. **Serve the application:**

   ```bash
   cd www
   python3 -m http.server 8000
   ```

   Or use any HTTP server:
   ```bash
   cd www
   # Python 3
   python3 -m http.server 8000
   # or Node.js
   npx serve .
   # or PHP
   php -S localhost:8000
   ```

4. **Open in browser:**

   Navigate to `http://localhost:8000` in your web browser.

## File Structure

```
www/
├── index.html          # Main HTML page
├── style.css           # Styling
├── main.js             # JavaScript application logic
├── package.json        # Build configuration
├── pkg/                # Generated WASM package (after build)
│   ├── picunic.js      # JavaScript bindings
│   ├── picunic_bg.wasm # WebAssembly binary
│   └── ...
└── assets/             # Model files (copy from root assets/)
    ├── encoder.onnx
    ├── encoder.embeddings.bin
    └── encoder.chars.json
```

## How It Works

1. **WebAssembly Module**: Handles image processing, chunking, dithering, and character matching
2. **ONNX Runtime Web**: Runs the CNN model in the browser to generate embeddings
3. **JavaScript Bridge**: Coordinates between WASM and ONNX Runtime

The application:
- Loads the ONNX model and embeddings on startup
- Processes uploaded images in the browser
- Converts images to Unicode art using the AI model
- Displays results in a monospace font

## Troubleshooting

### Build Errors

- **Missing wasm-pack**: Install with `cargo install wasm-pack`
- **Rust version**: Ensure you're using Rust 1.70+ (`rustc --version`)
- **`ort` crate error for WASM**: The `ort` crate (ONNX Runtime Rust bindings) doesn't support WASM. This is expected and handled automatically - the crate is conditionally excluded for WASM builds. The web version uses ONNX Runtime Web (JavaScript) instead.

### Runtime Errors

- **CORS issues**: Make sure you're serving via HTTP, not `file://`
- **Model loading**: Check that all assets are in `www/assets/`
- **ONNX Runtime**: The app loads ONNX Runtime Web from CDN; ensure internet connection
- **External data file error**: If you see "Failed to load external data file encoder.onnx.data":
  - Merge external data into model: `python3 merge_onnx_data.py assets/encoder.onnx www/assets/encoder.onnx`
  - Or use training venv: `training/.venv/bin/python merge_onnx_data.py assets/encoder.onnx www/assets/encoder.onnx`

### Performance

- Large images may take time to process
- Consider resizing images before upload for better performance
- The ONNX model runs in WebAssembly, which may be slower than native

## Quick Start

After building, you should have:

```
www/
├── pkg/                    # WASM package (generated)
│   ├── picunic.js
│   ├── picunic_bg.wasm
│   └── ...
├── assets/                 # Model files (copied from root)
│   ├── encoder.onnx
│   ├── encoder.embeddings.bin
│   └── encoder.chars.json
├── index.html
├── main.js
└── style.css
```

Then serve and open in your browser!

## Development

To rebuild after code changes:

```bash
# From project root
wasm-pack build --target web --out-dir www/pkg
# Refresh browser
```

For faster iteration during development:

```bash
wasm-pack build --target web --out-dir www/pkg --dev
```
