"""Demonstration of CoolMind's quantization capabilities"""

import time
from unittest.mock import patch
from coolmind import CoolEngine
from coolmind.core.engine import ThermalConfig

def demo_quantization():
    """Demonstrate how CoolMind applies quantization based on temperature"""
    print("=== CoolMind Quantization Demo ===")
    
    # Create engine with LOW thresholds for demo (so we can see changes at lower temps)
    thermal_config = ThermalConfig(
        max_gpu_temp=80.0,
        offload_threshold=55.0,   # Low threshold for demo
        cooldown_threshold=50.0,  # Low threshold for demo
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
    print(f"  Model Size: {status['model_size_mb']} MB")
    
    # Simulate different temperatures and see quantization response
    test_temps = [45.0, 50.0, 52.0, 57.0, 60.0]
    
    for temp in test_temps:
        print(f"\n--- Simulating Temperature: {temp}°C ---")
        
        # Manually set temperature to test quantization logic
        engine._current_temp = temp
        engine._apply_quantization_if_needed()
        
        status = engine.get_status()
        print(f"  Quantization Level: {status['quantization_level']}")
        print(f"  Is Quantized: {status['is_quantized']}")
        print(f"  Model Device: {status['model_device']}")
        
        # Test generation to make sure it still works
        response = engine.generate("AI", max_length=5)
        print(f"  Test Generation: '{response.strip()}'")
    
    print(f"\n=== Final State ===")
    final_status = engine.get_status()
    print(f"Temperature: {final_status['temperature']:.1f}°C")
    print(f"Device: {final_status['model_device']}")
    print(f"Quantized: {final_status['is_quantized']} ({final_status['quantization_level']})")
    print(f"Model Size: {final_status['model_size_mb']} MB")

if __name__ == "__main__":
    demo_quantization()