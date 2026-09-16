import coolmind
import time

def test_basic():
    # Use a publicly available, tiny model for testing
    engine = coolmind.CoolEngine(model_name="sshleifer/tiny-gpt2")
    engine.start_monitoring()
    time.sleep(3)  # Let monitoring run briefly
    response = engine.generate("Hello", max_length=10)
    print(f"Response: {response}")
    engine.stop_monitoring()
    print("Basic test passed")

if __name__ == "__main__":
    test_basic()