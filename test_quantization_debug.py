"""Simple test to debug quantization logic"""

import logging
from coolmind import CoolEngine
from coolmind.core.engine import ThermalConfig

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

def test_quantization():
    """Test quantization with debug output"""
    print("=== Testing Quantization Logic ===")
    
    # Create engine with thresholds that will trigger changes at low temps for testing
    thermal_config = ThermalConfig(
        max_gpu_temp=80.0,
        offload_threshold=30.0,   # Very low for testing
        cooldown_threshold=25.0,  # Very low for testing
        check_interval=1.0
    )
    
    engine = CoolEngine(
        model_name="sshleifer/tiny-gpt2",
        thermal_config=thermal_config
    )
    
    print(f"Initial state:")
    status = engine.get_status()
    print(f"  Temperature: {status['temperature']:.1f}°C")
    print(f"  Device: {status['model_device']}")
    print(f"  Quantized: {status['is_quantized']} ({status['quantization_level']})")
    
    # Test different temperatures
    test_temps = [20.0, 25.0, 28.0, 32.0]
    
    for temp in test_temps:
        print(f"\n--- Testing Temperature: {temp}°C ---")
        
        # Manually set temperature
        engine._current_temp = temp
        
        # Check what quantization level should be applied
        from coolmind.quantization import get_quantization_level, should_offload_instead_of_quantize
        quant_level = get_quantization_level(temp, thermal_config.offload_threshold, thermal_config.cooldown_threshold)
        should_offload = should_offload_instead_of_quantize(temp, thermal_config.offload_threshold)
        
        print(f"  Expected quant level: {quant_level}")
        print(f"  Should offload: {should_offload}")
        print(f"  Current state - Quantized: {engine._is_quantized}, Level: {engine._quantization_level}")
        
        # Apply quantization logic
        engine._apply_quantization_if_needed()
        
        status = engine.get_status()
        print(f"  After apply - Quantized: {status['is_quantized']} ({status['quantization_level']})")
        print(f"  Device: {status['model_device']}")
        
        # Quick generation test
        try:
            response = engine.generate("Test", max_length=3)
            print(f"  Generation: '{response.strip()}'")
        except Exception as e:
            print(f"  Generation failed: {e}")

if __name__ == "__main__":
    test_quantization()