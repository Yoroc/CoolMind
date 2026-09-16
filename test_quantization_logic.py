"""Unit test for CoolMind quantization logic"""

from coolmind.core.engine import CoolEngine, ThermalConfig

def test_quantization_logic():
    """Test that quantization levels are calculated correctly"""
    print("=== Testing Quantization Logic ===\n")
    
    # Test with default thresholds (cooldown=50.0, offload=55.0)
    config = ThermalConfig(
        cooldown_threshold=50.0,
        offload_threshold=55.0
    )
    
    engine = CoolEngine(
        model_name="sshleifer/tiny-gpt2",
        config=config
    )
    
    # Don't start monitoring for this test - we'll test the logic directly
    test_cases = [
        (25.0, 'none'),   # Below cooldown
        (50.0, 'int8'),   # At cooldown boundary 
        (52.0, 'int8'),   # Between thresholds
        (55.0, 'int4'),   # At offload boundary 
        (57.0, 'int4'),   # Above offload
        (80.0, 'int4'),   # Well above offload
    ]
    
    for temp, expected in test_cases:
        actual = engine._get_quantization_level_for_temp(temp)
        status = "✓" if actual == expected else "✗"
        print(f"{status} Temp: {temp:4.1f}°C → Expected: {expected:5} → Actual: {actual:5}")
        
        if actual != expected:
            print(f"   ERROR: Expected {expected}, got {actual}")
    
    print("\n=== Testing should_offload_instead_of_quantize ===\n")
    
    from coolmind.quantization import should_offload_instead_of_quantize
    
    offload_tests = [
        (50.0, 55.0, False),  # Below offload threshold
        (55.0, 55.0, True),   # At offload threshold
        (60.0, 55.0, True),   # Above offload threshold
    ]
    
    for temp, threshold, expected in offload_tests:
        actual = should_offload_instead_of_quantize(temp, threshold)
        status = "✓" if actual == expected else "✗"
        print(f"{status} Temp: {temp:4.1f}°C, Threshold: {threshold:4.1f} → Expected: {expected} → Actual: {actual}")

if __name__ == "__main__":
    test_quantization_logic()