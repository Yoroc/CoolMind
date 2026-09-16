"""Direct test of quantization functionality"""

import torch
import torch.nn as nn
from coolmind.quantization import apply_dynamic_quantization

def test_quantization_directly():
    """Test the quantization function directly"""
    print("=== Testing Quantization Function Directly ===")
    
    # Create a simple linear model to test
    model = nn.Sequential(
        nn.Linear(10, 5),
        nn.ReLU(),
        nn.Linear(5, 2)
    )
    
    print(f"Original model:")
    print(f"  Type: {type(model)}")
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            print(f"  {name}: Linear({module.in_features}, {module.out_features})")
    
    # Count parameters
    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {param_count}")
    
    # Apply quantization
    print("\nApplying dynamic quantization...")
    quantized_model = apply_dynamic_quantization(model)
    
    print(f"Quantized model:")
    print(f"  Type: {type(quantized_model)}")
    for name, module in quantized_model.named_modules():
        if isinstance(module, nn.Linear):
            print(f"  {name}: Linear({module.in_features}, {module.out_features})")
        elif hasattr(module, '_weight_quantize'):  # Quantized linear layer
            print(f"  {name}: QuantizedLinear")
    
    # Count parameters (should be same)
    quant_param_count = sum(p.numel() for p in quantized_model.parameters())
    print(f"  Parameters: {quant_param_count}")
    
    # Test forward pass
    import torch
    test_input = torch.randn(1, 10)
    
    with torch.no_grad():
        original_output = model(test_input)
        quantized_output = quantized_model(test_input)
    
    print(f"\nForward pass test:")
    print(f"  Input shape: {test_input.shape}")
    print(f"  Original output: {original_output}")
    print(f"  Quantized output: {quantized_output}")
    print(f"  Outputs close: {torch.allclose(original_output, quantized_output, atol=1e-4)}")

if __name__ == "__main__":
    test_quantization_directly()