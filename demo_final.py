"""Final demonstration showing quantization being applied"""

import time
import threading
from unittest.mock import patch, MagicMock
from coolmind import CoolEngine
from coolmind.core.engine import ThermalConfig

def demo_quantization_applied():
    """Demo showing quantization actually being applied"""
    print("=== CoolMind Quantization Application Demo ===\n")
    
    # Create engine with low thresholds for fast demo
    config = ThermalConfig(
        cooldown_threshold=25.0,   # Quantize above 25°C
        offload_threshold=30.0,    # More aggressive above 30°C
        update_interval=0.5        # Fast updates for demo
    )
    
    print(f"Configuration:")
    print(f"  Cooldown threshold: {config.cooldown_threshold}°C")
    print(f"  Offload threshold: {config.offload_threshold}°C")
    print(f"  Update interval: {config.update_interval}s\n")
    
    # Create engine (this starts monitoring automatically)
    engine = CoolEngine(
        model_name="sshleifer/tiny-gpt2",
        config=config
    )
    
    print("Engine created with monitoring started!\n")
    
    # Wait a moment for initial reading
    time.sleep(1.0)
    
    # Test temperatures by mocking WMI
    test_temps = [20.0, 27.0, 32.0, 22.0]  # Move through zones
    labels = ["Cool", "Warm", "Hot", "Cooling"]
    
    for label, temp in zip(labels, test_temps):
        print(f"--- {label} ({temp}°C) ---")
        
        # Mock WMI to return our temperature
        with patch('coolmind.core.engine.wmi') as mock_wmi:
            mock_wmi_instance = MagicMock()
            mock_wmi.WMI.return_value = mock_wmi_instance
            
            mock_sensor = MagicMock()
            mock_wmi_instance.MSAcpi_ThermalZoneTemperature.return_value = [mock_sensor]
            
            # Convert to WMI format (tenths of Kelvin)
            mock_sensor.CurrentTemperature = int((temp + 273.15) * 10)
            
            # Wait for monitoring thread to pick up the change
            time.sleep(config.update_interval * 2.5)
            
            # Check status
            status = engine.get_status()
            print(f"  Current temp: {status.temperature:.1f}°C")
            print(f"  Quantized: {status.quantized}")
            print(f"  Quantization level: {status.quantization_level}")
            
            # Quick generation test
            try:
                response = engine.generate("Hi", max_length=5)
                print(f"  Generation: '{response.strip()}'")
            except Exception as e:
                print(f"  Generation error: {e}")
            
            print()
    
    # Stop monitoring
    engine.stop_monitoring()
    print("Demo complete - monitoring stopped.")
    print(f"Final state - Quantized: {engine._is_quantized}, Level: {engine._quantization_level}")

if __name__ == "__main__":
    demo_quantization_applied()