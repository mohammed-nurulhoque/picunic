#!/bin/bash
# Build script for WebAssembly

set -e

echo "Building WebAssembly module..."

# Check if wasm-pack is installed
if ! command -v wasm-pack &> /dev/null; then
    echo "Error: wasm-pack not found. Install with: cargo install wasm-pack"
    exit 1
fi

# Create www directory if it doesn't exist
mkdir -p www/assets

# Build WASM module (from project root, output to www/pkg)
wasm-pack build --target web --out-dir www/pkg

echo "Copying assets..."

# Copy model files to www/assets
cp assets/encoder.onnx www/assets/ 2>/dev/null || echo "Warning: encoder.onnx not found"
cp assets/encoder.embeddings.bin www/assets/ 2>/dev/null || echo "Warning: encoder.embeddings.bin not found"
cp assets/encoder.chars.json www/assets/ 2>/dev/null || echo "Warning: encoder.chars.json not found"

# Merge external data if it exists (required for ONNX Runtime Web)
if [ -f assets/encoder.onnx.data ]; then
    echo "⚠️  Model has external data file. Merging for web compatibility..."
    if command -v python3 &> /dev/null && python3 -c "import onnx" 2>/dev/null; then
        python3 merge_onnx_data.py assets/encoder.onnx www/assets/encoder.onnx
    elif [ -f training/.venv/bin/python ]; then
        training/.venv/bin/python merge_onnx_data.py assets/encoder.onnx www/assets/encoder.onnx
    else
        echo "⚠️  Could not merge external data automatically."
        echo "   Run manually: python3 merge_onnx_data.py assets/encoder.onnx www/assets/encoder.onnx"
        echo "   Or use training venv: training/.venv/bin/python merge_onnx_data.py assets/encoder.onnx www/assets/encoder.onnx"
    fi
fi

echo "Build complete!"
echo ""
echo "To serve the application:"
echo "  cd www"
echo "  python3 -m http.server 8000"
echo ""
echo "Then open http://localhost:8000 in your browser"
