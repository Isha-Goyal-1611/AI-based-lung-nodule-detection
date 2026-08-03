import torch
import torch.nn as nn
import time
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.cnn_2d import SimpleCNN

def benchmark_model(model, input_tensor, n_runs=100):
    """
    Measure average inference time over n_runs
    Returns: average time in milliseconds
    """
    model.eval()
    with torch.no_grad():
        # Warmup run (first run is always slower)
        _ = model(input_tensor)
        
        # Timed runs
        start = time.time()
        for _ in range(n_runs):
            _ = model(input_tensor)
        end = time.time()
    
    avg_time_ms = (end - start) / n_runs * 1000
    return avg_time_ms

def get_model_size_mb(model):
    """Calculate model size in MB"""
    total_params = sum(p.numel() for p in model.parameters())
    # FP32: 4 bytes per parameter
    size_mb = total_params * 4 / 1_000_000
    return total_params, size_mb

def export_to_onnx(model, input_tensor, output_path='outputs/model.onnx'):
    """Export PyTorch model to ONNX format"""
    model.eval()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    torch.onnx.export(
        model,
        input_tensor,
        output_path,
        export_params=True,
        opset_version=17,
        input_names=['ct_patch'],
        output_names=['nodule_probability'],
        dynamic_axes={
            'ct_patch': {0: 'batch_size'},
            'nodule_probability': {0: 'batch_size'}
        }
    )
    
    # Get ONNX file size
    onnx_size_bytes = os.path.getsize(output_path)
    onnx_size_mb = onnx_size_bytes / 1_000_000
    onnx_size_kb = onnx_size_bytes / 1_000
    return onnx_size_mb, onnx_size_kb

def apply_dynamic_quantization(model):
    """
    Apply dynamic quantization (FP32 → INT8)
    Dynamic = weights quantized, activations quantized at runtime
    """
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {nn.Linear},  # quantize Linear layers
        dtype=torch.qint8
    )
    return quantized_model

if __name__ == "__main__":
    # Load REAL trained model, not a fresh untrained one
    import torch
    model = SimpleCNN()
    checkpoint = torch.load('checkpoints/best_model_gpu.pt', map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Loaded real trained checkpoint (epoch {checkpoint['epoch']}, "
          f"val_loss {checkpoint['val_loss']:.4f})\n")
    
    # Fake input
    fake_input = torch.randn(1, 1, 32, 32)
    
    print("=" * 50)
    print("MODEL OPTIMIZATION REPORT")
    print("=" * 50)
    
    # Step 1: Benchmark original FP32 model
    total_params, size_mb = get_model_size_mb(model)
    fp32_time = benchmark_model(model, fake_input)
    print(f"\n📊 Original FP32 Model:")
    print(f"   Parameters:     {total_params:,}")
    print(f"   Size:           {size_mb:.2f} MB")
    print(f"   Inference time: {fp32_time:.2f} ms")
    
    # Step 2: Export to ONNX
    print(f"\n📦 Exporting to ONNX...")
    onnx_size_mb, onnx_size_kb = export_to_onnx(model, fake_input)
    print(f"   ONNX file size: {onnx_size_mb:.2f} MB ({onnx_size_kb:.2f} KB)")
    print(f"   Saved to: outputs/model.onnx ✅")
    
    # Step 3: Apply quantization
    print(f"\n⚡ Applying Dynamic Quantization (FP32 → INT8)...")
    quantized_model = apply_dynamic_quantization(model)
    int8_time = benchmark_model(quantized_model, fake_input)
    
    print(f"\n📊 Quantized INT8 Model:")
    print(f"   Inference time: {int8_time:.2f} ms")
    print(f"   Speedup:        {fp32_time/int8_time:.2f}x faster")
    
    # Step 4: Compare predictions
    with torch.no_grad():
        fp32_pred = model(fake_input).item()
        int8_pred = quantized_model(fake_input).item()
    
    print(f"\n🔍 Prediction Comparison:")
    print(f"   FP32 prediction: {fp32_pred:.6f}")
    print(f"   INT8 prediction: {int8_pred:.6f}")
    print(f"   Difference:      {abs(fp32_pred-int8_pred):.6f}")
    
    if abs(fp32_pred - int8_pred) < 0.01:
        print(f"   ✅ Acceptable accuracy drop (<0.01)")
    else:
        print(f"   ⚠️  Large accuracy drop — review before deploying!")
    
    print("\n" + "=" * 50)