"""Demo script showcasing CoolMind's thermal-aware capabilities"""

import time
from coolmind import CoolEngine
from coolmind.core.engine import ThermalConfig
from coolmind.quantization import get_quantization_level, should_offload_instead_of_quantize

def demo_thermal_awareness():
    """Demonstrate thermal-aware model management"""
    print("=== CoolMind Thermal Awareness Demo ===\n")
    
    # Create engine with conservative thresholds for demo
    thermal_config = ThermalConfig(
        max_gpu_temp=75.0,
        cooldown_threshold=50.0,  # Start quantizing at 50°C
        offload_threshold=60.0,   # More aggressive at 60°C
        update_interval=1.0       # Check every second
    )
    
    print("Creating CoolMind engine with thermal configuration...")
    print(f"Max GPU Temp: {thermal_config.max_gpu_temp}°C")
    print(f"Cooldown Threshold: {thermal_config.cooldown_threshold}°C") 
    print(f"Offload Threshold: {thermal_config.offload_threshold}°C")
    print(f"Update Interval: {thermal_config.update_interval}s\n")
    
    # Initialize the engine
    engine = CoolEngine(
        model_name="sshleifer/tiny-gpt2",
        config=thermal_config
    )
    
    print("Engine created successfully!\n")
    
    # Show initial status
    status = engine.get_status()
    print("Initial Engine Status:")
    print(f"  Temperature: {status.temperature:.1f}°C")
    print(f"  Device: {status.device}")
    print(f"  Quantized: {status.quantized}")
    print(f"  Quantization Level: {status.quantization_level}")
    print()
    
    # Demonstrate the quantization level logic
    print("--- Quantization Level Logic ---\n")
    
    test_temps = [30.0, 52.0, 65.0, 40.0]
    temp_labels = ["Cool (Idle)", "Warm (Light Load)", "Hot (Heavy Load)", "Cooling Down"]
    
    for label, temp in zip(temp_labels, test_temps):
        print(f"{label} ({temp}°C):")
        
        # Get quantization level from our logic
        quant_level = get_quantization_level(
            temp, 
            thermal_config.offload_threshold, 
            thermal_config.cooldown_threshold
        )
        
        # Check if we should offload instead of quantize
        should_offload = should_offload_instead_of_quantize(
            temp, 
            thermal_config.offload_threshold
        )
        
        print(f"  Quantization Level: {quant_level}")
        print(f"  Should Offload: {should_offload}")
        
        # Show what the engine would do
        engine._current_temp = temp
        engine_status = engine.get_status()
        print(f"  Engine State - Quantized: {engine_status.quantized}, Level: {engine_status.quantization_level}")
        print()
    
    print("--- Demo Complete ---")
    print("In a real implementation:")
    print("- WMI would provide actual GPU temperatures") 
    print("- Monitoring thread would automatically adjust model precision")
    print("- Models would be dynamically quantized based on heat")
    print("- Users get sustained performance without thermal throttling")

if __name__ == "__main__":
    demo_thermal_awareness()