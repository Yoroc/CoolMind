"""Direct demonstration of quantization being applied"""

from coolmind.core.engine import CoolEngine, ThermalConfig
from coolmind.quantization import apply_dynamic_quantization
import torch.nn as nn

def demo_direct_quantization():
    """Directly demonstrate quantization being applied"""
    print("=== Direct Quantization Demonstration ===\n")
    
    # Create engine
    config = ThermalConfig(
        cooldown_threshold=50.0,
        offload_threshold=55.0
    )
    
    engine = CoolEngine(
        model_name="sshleifer/tiny-gpt2",
        config=config
    )
    
    # Don't start monitoring - we'll manually test
    
    print("Initial state:")
    status = engine.get_status()
    print(f"  Temperature: {status.temperature:.1f}°C")
    print(f"  Quantized: {status.quantized}")
    print(f"  Quantization level: {status.quantization_level}")
    print()
    
    # Manually set temperature low and check quantization level
    engine._current_temp = 25.0
    quant_level = engine._get_quantization_level_for_temp(engine._current_temp)
    print(f"At 25.0°C:")
    print(f"  Target quantization level: {quant_level}")
    print(f"  Should be 'none': {quant_level == 'none'}")
    print()
    
    # Manually set temperature medium and check quantization level
    engine._current_temp = 52.0
    quant_level = engine._get_quantization_level_for_temp(engine._current_temp)
    print(f"At 52.0°C:")
    print(f"  Target quantization level: {quant_level}")
    print(f"  Should be 'int8': {quant_level == 'int8'}")
    print()
    
    # Manually set temperature high and check quantization level
    engine._current_temp = 57.0
    quant_level = engine._get_quantization_level_for_temp(engine._current_temp)
    print(f"At 57.0°C:")
    print(f"  Target quantization level: {quant_level}")
    print(f"  Should be 'int4': {quant_level == 'int4'}")
    print()
    
    # Now actually demonstrate applying quantization
    print("--- Applying INT8 quantization manually ---")
    print("Before quantization:")
    print(f"  Model type: {type(engine.model)}")
    
    # Store original model for comparison
    original_model = engine.model
    
    # Apply quantization (this is what would happen at 52°C+)
    engine.model = apply_dynamic_quantization(engine.model)
    engine._is_quantized = True
    engine._quantization_level = 'int8'
    
    print("After quantization:")
    print(f"  Model type: {type(engine.model)}")
    print(f"  Is quantized: {engine._is_quantized}")
    print(f"  Quantization level: {engine._quantization_level}")
    print()
    
    # Test that we can still generate
    try:
        response = engine.generate("Hello", max_length=10)
        print(f"Generation after quantization: '{response.strip()}'")
    except Exception as e:
        print(f"Generation error: {e}")
    
    print("\n--- Reloading original precision model ---")
    # Simulate cooling down and reloading
    engine._load_model()  # This reloads the original precision model
    engine._is_quantized = False
    engine._quantization_level = 'none'
    
    print("After reload:")
    print(f"  Model type: {type(engine.model)}")
    print(f"  Is quantized: {engine._is_quantized}")
    print(f"  Quantization level: {engine._quantization_level}")
    
    try:
        response = engine.generate("Hello", max_length=10)
        print(f"Generation after reload: '{response.strip()}'")
    except Exception as e:
        print(f"Generation error: {e}")

if __name__ == "__main__":
    demo_direct_quantization()