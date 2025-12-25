#!/usr/bin/env python3
"""
Merge ONNX external data file into the main model file.
This makes the model self-contained for web deployment.

Required for ONNX Runtime Web, which doesn't support external data files.

Usage:
    python3 merge_onnx_data.py <input.onnx> <output.onnx>
    
Or if you have the training venv:
    training/.venv/bin/python merge_onnx_data.py assets/encoder.onnx www/assets/encoder.onnx
"""
import sys
import os

try:
    import onnx
except ImportError:
    print("Error: 'onnx' package not found.")
    print("Install it with: pip install onnx")
    sys.exit(1)

def merge_external_data(model_path, output_path):
    """Load ONNX model and merge external data into it."""
    if not os.path.exists(model_path):
        print(f"Error: Model file not found: {model_path}")
        sys.exit(1)
    
    print(f"Loading model from {model_path}...")
    model = onnx.load(model_path)
    
    # Check if external data file exists
    data_path = model_path + '.data'
    if os.path.exists(data_path):
        print(f"Found external data file: {data_path}")
        print("Merging external data into model...")
    else:
        print("No external data file found, model may already be self-contained.")
    
    # Save without external data (this merges it into the model)
    onnx.save_model(model, output_path, save_as_external_data=False)
    
    # Get file sizes for comparison
    input_size = os.path.getsize(model_path)
    output_size = os.path.getsize(output_path)
    if os.path.exists(data_path):
        data_size = os.path.getsize(data_path)
        print(f"\nFile sizes:")
        print(f"  Input model:  {input_size:,} bytes")
        print(f"  External data: {data_size:,} bytes")
        print(f"  Merged model:  {output_size:,} bytes")
        print(f"  Total saved:   {input_size + data_size:,} bytes")
    
    print(f"\n✓ Saved merged model to {output_path}")
    print("  You can now delete the .onnx.data file if desired.")

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python merge_onnx_data.py <input.onnx> <output.onnx>")
        print("\nExample:")
        print("  python merge_onnx_data.py assets/encoder.onnx www/assets/encoder.onnx")
        sys.exit(1)
    
    merge_external_data(sys.argv[1], sys.argv[2])
