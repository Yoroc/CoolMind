# CoolMind

![CI](https://github.com/Yoroc/CoolMind/actions/workflows/ci.yml/badge.svg)
![PyPI - Version](https://img.shields.io/pypi/v/coolmind?style=flat-square)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/coolmind?style=flat-square)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)

![Demo](./@yoro.svg)

Run powerful AI models locally without overheating your PC.

---

## 🚀 Why CoolMind?

- **Thermal-aware**: Automatically manages model placement based on GPU temperature
- **Zero-config**: Works out of the box with sensible defaults
- **Windows-optimized**: Uses WMI for precise thermal monitoring on Windows
- **Lightweight**: Built on Hugging Face Transformers with minimal dependencies
- **Smart quantization**: Applies model quantization when temperatures rise
- **Flexible CLI**: Both interactive and single-query modes available
- **Safe to use**: Prevents thermal throttling without complex setup

Stop worrying about your PC overheating when running local AI models.  
CoolMind keeps your system cool while maintaining performance.

---

## 📦 Installation

```bash
pip install coolmind

# Or install from source
git clone https://github.com/Yoroc/CoolMind.git
cd CoolMind
pip install -e .
```

---

## 🛠️ Usage

```bash
# Interactive chat mode
coolmind -m microsoft/DialoGPT-medium

# Single query
coolmind -m sshleifer/tiny-gpt2 -q "Explain machine learning simply" --max-length 100

# With custom settings
coolmind -m facebook/opt-125m -q "What is artificial intelligence?" --temperature 0.8 --top-p 0.9

# Disable thermal monitoring (for testing)
coolmind -m sshleifer/tiny-gpt2 -q "Test query" --no-monitor

# Show model status after generation
coolmind -m sshleifer/tiny-gpt2 -q "Hello" --max-length 10 --show-status
```

### Options

| Flag | Description |
|------|-------------|
| `-m, --model <name>` | Model name or path from Hugging Face Hub |
| `-q, --query <text>` | Single query mode (exit after response) |
| `--max-length <int>` | Maximum generation length |
| `--temperature <float>` | Sampling temperature (0.0 to 1.0) |
| `--top-p <float>` | Top-p sampling parameter |
| `--no-monitor` | Disable thermal monitoring |
| `--show-status` | Display model status after generation |
| `-c, --config <file>` | Path to configuration file |

---

## 🧩 Configuration

Create a `config.yaml` file to customize behavior:

```yaml
model:
  default_name: "sshleifer/tiny-gpt2"
  max_length: 50
  temperature: 0.7
  top_p: 0.9
  do_sample: true

thermal:
  max_gpu_temp: 80.0          # Temperature to trigger actions
  cooldown_threshold: 50.0    # Below this: no quantization
  offload_threshold: 55.0     # At/below: INT8, above: INT4
  update_interval: 2.0        # Check interval (seconds)
  enable_monitoring: true     # Enable thermal monitoring thread

logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
```

---

## 🔧 How It Works

CoolMind continuously monitors your system's temperature using Windows Management Instrumentation (WMI). When temperatures exceed thresholds:

1. **Warm temperatures (50-55°C)**: Applies INT8 dynamic quantization to reduce computational load
2. **Hot temperatures (55°C+)**: Applies more aggressive INT4 quantization or prepares for CPU offload
3. **Cool temperatures (<50°C)**: Reloads model at full precision for optimal performance
4. **Continuous cycle**: Maintains performance while preventing thermal throttling

The monitoring runs in a background thread with configurable check intervals, ensuring real-time response to temperature changes.

---

## 🏗️ Architecture

```
coolmind/
├── cli.py              # Command-line interface with argparse
├── core/
│   ├── engine.py       # Thermal-aware inference engine
│   └── quantization.py # Dynamic quantization utilities
├── __init__.py         # Package exports (main, CoolEngine, ThermalConfig)
└── __main__.py         # Entry point for `python -m coolmind`
```

---

## 📋 Requirements

- Windows 10/11 (for WMI-based thermal monitoring)
- Python 3.9+
- Recommended: NVIDIA GPU with CUDA support (optional)
- Minimum 4GB RAM
- Internet connection for initial model download

---

## 📜 License

MIT © Yoroc  

---

*Built for developers who want to run local AI models without turning their workstation into a space heater.*