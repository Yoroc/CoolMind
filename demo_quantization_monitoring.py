"""Demonstration of CoolMind's quantization capabilities with thermal monitoring"""

import time
import threading
from unittest.mock import patch
from coolmind import CoolEngine
from coolmind.core.engine import ThermalConfig

def demo_quantization_with_monitoring():
    """Demo quantization with active thermal monitoring"""
    print("=== CoolMind Quantization Demo with Monitoring ===\n")
    
    # Create engine with monitoring enabled
    config = ThermalConfig(
        cooldown_threshold=30.0,   # Lower thresholds for demo
        offload_threshold=35.0
    )
    
    print(f"Creating engine with thresholds:")
    print(f"  Cooldown threshold: {config.cooldown_threshold}°C")
    print(f"  Offload threshold: {config.offload_threshold}°C\n")
    
    # Create engine - this starts monitoring thread automatically
    engine = CoolEngine(
        model_name="sshleifer/tiny-gpt2",
        config=config
    )
    
    print("Engine created and monitoring started!\n")
    
    # Test different temperatures by mocking the WMI interface
    test_temps = [25.0, 32.0, 38.0, 28.0]  # Move through different zones
    
    for temp in test_temps:
        print(f"--- Setting temperature to {temp}°C ---")
        
        # Mock WMI to return our test temperature
        with patch('coolmind.core.engine.wmi') as mock_wmi:
            mock_instance = mock_wmi.WMI.return_value
            mock_sensor = mock_instance.MSAcpi_ThermalZoneTemperature.return_value
            mock_sensor.__iter__.return_value = [mock_sensor]
            mock_sensor.CurrentTemperature.return_value = int(temp * 10)  # WMI uses tenths of degree
            
            # Give the monitoring thread a moment to update
            time.sleep(0.2)
            
            # Check engine state
            status = engine.get_status()
            quant_level = engine._get_quantization_level_for_temp(status.temperature)
            
            print(f"  Actual temp: {status.temperature:.1f}°C")
            print(f"  Target quant level: {quant_level}")
            print(f"  Current quantized: {status.quantized}")
            print(f"  Quantization level: {status.quantization_level}")
            
            # Test generation
            try:
                response = engine.generate("Test", max_length=5)
                print(f"  Generation: '{response.strip()}'")
            except Exception as e:
                print(f"  Generation error: {e}")
            
            print()
    
    # Stop monitoring
    engine.stop_monitoring()
    print("Monitoring stopped.")
    print(f"Final state - Quantized: {engine._is_quantized}, Level: {engine._quantization_level}")

if __name__ == "__main__":
    demo_quantization_with_monitoring()