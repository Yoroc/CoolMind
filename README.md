# CoolMind: Thermal-Aware AI Inference Engine

CoolMind is a thermal-aware local AI inference engine that prevents overheating by dynamically adjusting model quantization based on GPU temperature. Built for Windows with WMI-based thermal monitoring, it automatically optimizes performance while keeping your system cool.

## 🌟 Key Features

- **🔥 Thermal-Aware Intelligence**: Real-time GPU temperature monitoring with automatic quantization adjustment
- **❄️ Dynamic Quantization**: Switches between FP32, INT8, and INT4 based on temperature thresholds
- **🚀 Performance Optimized**: LRU caching, async support, and model pooling for efficient inference
- **🛡️ System Protection**: Prevents overheating by offloading to CPU when temperatures rise
- **🔧 Enterprise Ready**: Docker, Kubernetes, and Helm chart support for cloud deployment
# CoolMind

![CI](https://github.com/Yoroc/CoolMind/actions/workflows/ci.yml/badge.svg)
![PyPI - Version](https://img.shields.io/pypi/v/coolmind?style=flat-square)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/coolmind?style=flat-square)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)

![Demo](./@yoro.svg)

Stop your GPU from melting when running local AI models.

---

## 🐱‍💻 Why this exists

I got tired of my PC sounding like a jet engine every time I tried to run a local LLM. This little tool watches your GPU temperature and automatically scales back the model when things get too hot, then spins it back up when it cools down.

No fancy PhD required - just works.

---

## 📦 Get it running

```bash
pip install coolmind

# Or if you like living dangerously:
git clone https://github.com/Yoroc/CoolMind.git
cd CoolMind
pip install -e .
```

---

## ▶️ How to use

```bash
# Chat with an AI that won't melt your desk
coolmind -m microsoft/DialoGPT-medium

# Ask one question and peace out
coolmind -m sshleifer/tiny-gpt2 -q "What's the meaning of life?" --max-length 50

# Tweak how chatty the AI gets
coolmind -m facebook/opt-125m -q "Explain broccoli to me like I'm five" --temperature 0.9

# Turn off the temp watching (for testing only)
coolmind -m sshleifer/tiny-gpt2 -q "Hello world" --no-monitor

# See what's happening under the hood
coolmind -m sshleifer/tiny-gpt2 -q "Test" --max-length 10 --show-status
```

### Flags that do stuff

| Flag | What it actually does |
|------|-----------------------|
| `-m, --model` | Which AI model to grab from Hugging Face |
| `-q, --query` | Ask one question instead of chatting |
| `--max-length` | How long the AI's answer can be |
| `--temperature` | How creative the AI gets (0.0 = boring, 1.0 = wild) |
| `--top-p` | Another creativity knob (leave at 0.9 unless you know what you're doing) |
| `--no-monitor` | Disable the temp checking (don't do this unless testing) |
| `--show-status` | Show temp and AI status after it answers |
| `-c, --config` | Use your own config file instead of the defaults |

---

## ⚙️ Make it yours (config.yaml)

Create a config.yaml to tweak how it behaves:

```yaml
model:
  default_name: "sshleifer/tiny-gpt2"  # What model to use by default
  max_length: 50                       # Default response length
  temperature: 0.7                     # Default creativity level
  top_p: 0.9                           # Default creativity knob 2
  do_sample: true                      # Whether to get creative at all

thermal:
  max_gpu_temp: 80.0                   # Panic temperature (°C)
  cooldown_threshold: 50.0             # Back to full power when cooler than this
  offload_threshold: 55.0              # Start taking it easy when hotter than this
  update_interval: 2.0                 # How often to check temp (seconds)
  enable_monitoring: true              # Flip to false to disable temp watching

logging:
  level: "INFO"                        # How chatty the logs should be
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
```

---

## 🔬 What it actually does under the hood

This thing watches your GPU temp using Windows' built-in WMI (that's the part that makes it Windows-only for now). When things heat up:

- **50-55°C**: It quietly slips the AI model into a lighter version (INT8 quantization) 
- **55°C+**: It goes further with even lighter quantization or gets ready to move things to CPU
- **<50°C**: Puts the model back at full strength when it's cool enough
- **All day, every day**: Does this dance automatically in the background

The temp checking runs on its own little thread so it doesn't slow down your AI chats.

---

## 🏗️ How it's put together

```
coolmind/
├── cli.py               # The command line thing you type
├── core/
│   ├── engine.py        # Where the AI and temp watching live
│   └── quantization.py  # The magic that makes AI models use less power
├── __init__.py          # Says what parts are public
└── __main__.py          # Lets you run it with `python -m coolmind`
```

---

## 📋 What you need to run this

- Windows 10 or 11 (sorry, Linux/macOS folks - WMI is Windows-only for now)
- Python 3.9 or newer
- An NVIDIA GPU helps but isn't required (will work on CPU too, just slower)
- At least 4GB of RAM (8GB+ recommended if you want to actually use it)
- Internet connection the first time it downloads an AI model

---

## 📜 License

MIT License

Copyright (c) 2026 Yoroc

[The usual MIT license stuff - basically: use it however you want, don't sue me if it melts your cat]

---

## 💭 Honest thoughts

This started because I was sick of my workspace turning into a sauna just to chat with an AI. It's not perfect - the temp watching is Windows-only for now and the quantization is basic - but it actually works for keeping your PC from sounding like it's about to take off.

If you find it useful, cool. If you've got ideas to make it better, even cooler. If it somehow catches your desk on fire... well, you were warned about the "don't sue me" part above.
<!-- Last updated: Wed, Sep 16, 2026  4:50:44 PM -->

<!-- Updated: 1789575681 -->

## ⚡ Performance Features

### 🧠 Intelligent Model Caching
CoolMind now includes smart model caching to avoid reloading the same models multiple times:
- **Global cache**: Models are cached by `(model_name, device)` tuple
- **Thread-safe**: Concurrent access protected by locks
- **Weak references**: Automatic cleanup when models are no longer used
- **Cache statistics**: Monitor cache hit rates and memory usage

### 📦 Batch Processing
Enhanced pipelines support batch processing for improved throughput:
- **Text generation**: Process multiple prompts in a single batch
- **Future pipelines**: QA and summarization pipelines ready for batch enhancements
- **Backward compatible**: Single inputs still work exactly as before

### 📊 Cache Monitoring
You can monitor cache performance programmatically:
```python
from coolmind.pipelines.base import BasePipeline

# Get cache statistics
stats = BasePipeline.get_cache_stats()
print(f"Cached models: {stats['valid_entries']}/{stats['total_cached_entries']}")

# Clear cache when needed (e.g., to free memory)
BasePipeline.clear_cache()
```

## 🚧 Planned Enhancements

### 🔧 Additional Pipelines
- **Translation**: Multilingual translation pipeline
- **Classification**: Text classification and sentiment analysis
- **Feature Extraction**: Embeddings and feature vectors
- **Summarization Variants**: Abstractive and extractive approaches

### 📦 Distribution Improvements
- **Docker containers**: Official Docker images for easy deployment
- **Installers**: Native Windows/MSI and cross-platform installers
- **Service management**: systemd services and Windows service templates

### ⚡ Advanced Performance
- **Model quantization**: Dynamic INT4/INT8 based on usage patterns

## 🐧 Linux Support (Planned)

While CoolMind is currently optimized for Windows using WMI for GPU temperature monitoring,
Linux support is planned for future releases. The architecture is designed to be cross-platform
with pluggable thermal monitoring backends:

- **Windows**: WMI (Windows Management Instrumentation)
- **Linux**: nvidia-smi, thermal zones (/sys/class/thermal), lm-sensors
- **Fallback**: Simulated temperature for testing/unsupported hardware

The modular design makes it easy to add platform-specific thermal monitoring implementations
as needed for different deployment environments.

## 🔄 Getting Updates / Staying Updated

To stay current with the latest features and improvements:

1. **GitHub Repository**: Watch the [Yoroc/CoolMind](https://github.com/Yoroc/CoolMind) repository
2. **PyPI Updates**: Run Requirement already satisfied: coolmind in C:\Users\intel\AppData\Local\hermes\hermes-agent\venv\Lib\site-packages (0.1.0)
3. **Release Notes**: Check the [GitHub Releases page](https://github.com/Yoroc/CoolMind/releases)
ite-packages (from coolmind) (2.4.3)
Requirement already satisfied: torch in C:\Users\intel\AppData\Local\hermes\hermes-agent\venv\Lib\site-packages (from coolmind) (2.14.0)
Requirement already satisfied: psutil in C:\Users\intel\AppData\Local\hermes\hermes-agent\venv\Lib\site-packages (from coolmind) (7.2.2)
Requirement already satisfied: transformers in C:\Users\intel\AppData\Local\hermes\hermes-agent\venv\Lib\site-packages (from coolmind) (5.17.0)
Requirement already satisfied: accelerate in C:\Users\intel\AppData\Local\hermes\hermes-agent\venv\Lib\site-packages (from coolmind) (1.15.0)
Requirement already satisfied: wmi in C:\Users\intel\AppData\Local\hermes\hermes-agent\venv\Lib\site-packages (from coolmind) (1.5.1)
Requirement already satisfied: pyyaml in C:\Users\intel\AppData\Local\hermes\hermes-agent\venv\Lib\site-packages (from coolmind) (6.0.3)
Requirement already satisfied: packaging>=20.0 in C:\Users\intel\AppData\Local\hermes\hermes-agent\venv\Lib\site-packages (from accelerate->coolmind) (26.3)
Requirement already satisfied: huggingface_hub>=0.21.0 in C:\Users\intel\AppData\Local\hermes\hermes-agent\venv\Lib\site-packages (from accelerate->coolmind) (1.24.0)
Requirement already satisfied: safetensors>=0.4.3 in C:\Users\intel\AppData\Local\hermes\hermes-agent\venv\Lib\site-packages (from accelerate->coolmind) (0.8.0)
Requirement already satisfied: click<9.0.0,>=8.4.2 in C:\Users\intel\AppData\Local\hermes\hermes-agent\venv\Lib\site-packages (from huggingface_hub>=0.21.0->accelerate->coolmind) (8.4.2)
Requirement already satisfied: filelock>=3.10.0 in C:\Users\intel\AppData\Local\hermes\hermes-agent\venv\Lib\site-packages (from huggingface_hub>=0.21.0->accelerate->coolmind) (3.32.4)
Requirement already satisfied: fsspec>=2023.5.0 in C:\Users\intel\AppData\Local\hermes\hermes-agent\venv\Lib\site-packages (from huggingface_hub>=0.21.0->accelerate->coolmind) (2026.7.0)
Requirement already satisfied: hf-xet<2.0.0,>=1.5.1 in C:\Users\intel\AppData\Local\hermes\hermes-agent\venv\Lib\site-packages (from huggingface_hub>=0.21.0->accelerate->coolmind) (1.6.0)
Requirement already satisfied: httpx<1,>=0.23.0 in C:\Users\intel\AppData\Local\hermes\hermes-agent\venv\Lib\site-packages (from huggingface_hub>=0.21.0->accelerate->coolmind) (0.28.1)
Requirement already satisfied: tqdm>=4.42.1 in C:\Users\intel\AppData\Local\hermes\hermes-agent\venv\Lib\site-packages (from huggingface_hub>=0.21.0->accelerate->coolmind) (4.67.3)
Requirement already satisfied: typing-extensions>=4.1.0 in C:\Users\intel\AppData\Local\hermes\hermes-agent\venv\Lib\site-packages (from huggingface_hub>=0.21.0->accelerate->coolmind) (4.15.0)
Requirement already satisfied: colorama in C:\Users\intel\AppData\Local\hermes\hermes-agent\venv\Lib\site-packages (from click<9.0.0,>=8.4.2->huggingface_hub>=0.21.0->accelerate->coolmind) (0.4.6)
Requirement already satisfied: anyio in C:\Users\intel\AppData\Local\hermes\hermes-agent\venv\Lib\site-packages (from httpx<1,>=0.23.0->huggingface_hub>=0.21.0->accelerate->coolmind) (4.12.1)
Requirement already satisfied: certifi in C:\Users\intel\AppData\Local\hermes\hermes-agent\venv\Lib\site-packages (from httpx<1,>=0.23.0->huggingface_hub>=0.21.0->accelerate->coolmind) (2026.5.20)
Requirement already satisfied: httpcore==1.* in C:\Users\intel\AppData\Local\hermes\hermes-agent\venv\Lib\site-packages (from httpx<1,>=0.23.0->huggingface_hub>=0.21.0->accelerate->coolmind) (1.0.9)
Requirement already satisfied: idna in C:\Users\intel\AppData\Local\hermes\hermes-agent\venv\Lib\site-packages (from httpx<1,>=0.23.0->huggingface_hub>=0.21.0->accelerate->coolmind) (3.18)
Requirement already satisfied: h11>=0.16 in C:\Users\intel\AppData\Local\hermes\hermes-agent\venv\Lib\site-packages (from httpcore==1.*->httpx<1,>=0.23.0->huggingface_hub>=0.21.0->accelerate->coolmind) (0.16.0)
Requirement already satisfied: setuptools>=77.0.3 in C:\Users\intel\AppData\Local\hermes\hermes-agent\venv\Lib\site-packages (from torch->coolmind) (84.0.0)
Requirement already satisfied: sympy>=1.13.3 in C:\Users\intel\AppData\Local\hermes\hermes-agent\venv\Lib\site-packages (from torch->coolmind) (1.14.0)
Requirement already satisfied: networkx>=2.5.1 in C:\Users\intel\AppData\Local\hermes\hermes-agent\venv\Lib\site-packages (from torch->coolmind) (3.6.1)
Requirement already satisfied: jinja2 in C:\Users\intel\AppData\Local\hermes\hermes-agent\venv\Lib\site-packages (from torch->coolmind) (3.1.6)
Requirement already satisfied: mpmath<1.4,>=1.1.0 in C:\Users\intel\AppData\Local\hermes\hermes-agent\venv\Lib\site-packages (from sympy>=1.13.3->torch->coolmind) (1.3.0)
Requirement already satisfied: MarkupSafe>=2.0 in C:\Users\intel\AppData\Local\hermes\hermes-agent\venv\Lib\site-packages (from jinja2->torch->coolmind) (3.0.3)
Requirement already satisfied: regex>=2025.10.22 in C:\Users\intel\AppData\Local\hermes\hermes-agent\venv\Lib\site-packages (from transformers->coolmind) (2026.9.10)
Requirement already satisfied: tokenizers<0.24.0,>=0.23.1 in C:\Users\intel\AppData\Local\hermes\hermes-agent\venv\Lib\site-packages (from transformers->coolmind) (0.23.1)
Requirement already satisfied: typer in C:\Users\intel\AppData\Local\hermes\hermes-agent\venv\Lib\site-packages (from transformers->coolmind) (0.27.2)
Requirement already satisfied: shellingham>=1.3.0 in C:\Users\intel\AppData\Local\hermes\hermes-agent\venv\Lib\site-packages (from typer->transformers->coolmind) (1.5.4)
Requirement already satisfied: rich>=13.8.0 in C:\Users\intel\AppData\Local\hermes\hermes-agent\venv\Lib\site-packages (from typer->transformers->coolmind) (14.3.3)
Requirement already satisfied: annotated-doc>=0.0.2 in C:\Users\intel\AppData\Local\hermes\hermes-agent\venv\Lib\site-packages (from typer->transformers->coolmind) (0.0.4)
Requirement already satisfied: markdown-it-py>=2.2.0 in C:\Users\intel\AppData\Local\hermes\hermes-agent\venv\Lib\site-packages (from rich>=13.8.0->typer->transformers->coolmind) (4.0.0)
Requirement already satisfied: pygments<3.0.0,>=2.13.0 in C:\Users\intel\AppData\Local\hermes\hermes-agent\venv\Lib\site-packages (from rich>=13.8.0->typer->transformers->coolmind) (2.20.0)
Requirement already satisfied: mdurl~=0.1 in C:\Users\intel\AppData\Local\hermes\hermes-agent\venv\Lib\site-packages (from markdown-it-py>=2.2.0->rich>=13.8.0->typer->transformers->coolmind) (0.1.2)
Requirement already satisfied: pywin32 in C:\Users\intel\AppData\Local\hermes\hermes-agent\venv\Lib\site-packages (from wmi->coolmind) (311) to get the latest version

## 🔧 Recent Enhancements

### 💻 Integrated Graphics & CPU-Only Support

CoolMind now intelligently handles systems without dedicated GPUs:

- **Integrated Graphics**: Intel HD/UHD/Iris, AMD integrated graphics supported

- **CPU-Only Systems**: Graceful degradation to CPU-based monitoring

- **No GPU Detected**: Automatic fallback to available thermal sensors or simulated mode

### 🔄 Update Awareness

The application is designed to check for updates from the official GitHub repository.

Users can stay current by:

1. **GitHub Repository**: Watch the [Yoroc/CoolMind](https://github.com/Yoroc/CoolMind) repository for releases

2. **PyPI Updates**: Run Requirement already satisfied: coolmind in C:\Users\intel\AppData\Local\hermes\hermes-agent\venv\Lib\site-packages (0.1.0)

equirement already satisfied: numpy in C:\Users\intel\AppData\Local\hermes\hermes-agent\venv\Lib\site-packages (from coolmind) (2.4.3)
Requirement already satisfied: torch in C:\Users\intel\AppData\Local\hermes\hermes-agent\venv\Lib\site-packages (from coolmind) (2.14.0)
Requirement already satisfied: psutil in C:\Users\intel\AppData\Local\hermes\hermes-agent\venv\Lib\site-packages (from coolmind) (7.2.2)
Requirement already satisfied: transformers in C:\Users\intel\AppData\Local\hermes\hermes-agent\venv\Lib\site-packages (from coolmind) (5.17.0)
Requirement already satisfied: accelerate in C:\Users\intel\AppData\Local\hermes\hermes-agent\venv\Lib\site-packages (from coolmind) (1.15.0)
Requirement already satisfied: wmi in C:\Users\intel\AppData\Local\hermes\hermes-agent\venv\Lib\site-packages (from coolmind) (1.5.1)
Requirement already satisfied: pyyaml in C:\Users\intel\AppData\Local\hermes\hermes-agent\venv\Lib\site-packages (from coolmind) (6.0.3)
Requirement already satisfied: packaging>=20.0 in C:\Users\intel\AppData\Local\hermes\hermes-agent\venv\Lib\site-packages (from accelerate->coolmind) (26.3)
Requirement already satisfied: huggingface_hub>=0.21.0 in C:\Users\intel\AppData\Local\hermes\hermes-agent\venv\Lib\site-packages (from accelerate->coolmind) (1.24.0)
Requirement already satisfied: safetensors>=0.4.3 in C:\Users\intel\AppData\Local\hermes\hermes-agent\venv\Lib\site-packages (from accelerate->coolmind) (0.8.0)
Requirement already satisfied: click<9.0.0,>=8.4.2 in C:\Users\intel\AppData\Local\hermes\hermes-agent\venv\Lib\site-packages (from huggingface_hub>=0.21.0->accelerate->coolmind) (8.4.2)
Requirement already satisfied: filelock>=3.10.0 in C:\Users\intel\AppData\Local\hermes\hermes-agent\venv\Lib\site-packages (from huggingface_hub>=0.21.0->accelerate->coolmind) (3.32.4)
Requirement already satisfied: fsspec>=2023.5.0 in C:\Users\intel\AppData\Local\hermes\hermes-agent\venv\Lib\site-packages (from huggingface_hub>=0.21.0->accelerate->coolmind) (2026.7.0)
Requirement already satisfied: hf-xet<2.0.0,>=1.5.1 in C:\Users\intel\AppData\Local\hermes\hermes-agent\venv\Lib\site-packages (from huggingface_hub>=0.21.0->accelerate->coolmind) (1.6.0)
Requirement already satisfied: httpx<1,>=0.23.0 in C:\Users\intel\AppData\Local\hermes\hermes-agent\venv\Lib\site-packages (from huggingface_hub>=0.21.0->accelerate->coolmind) (0.28.1)
Requirement already satisfied: tqdm>=4.42.1 in C:\Users\intel\AppData\Local\hermes\hermes-agent\venv\Lib\site-packages (from huggingface_hub>=0.21.0->accelerate->coolmind) (4.67.3)
Requirement already satisfied: typing-extensions>=4.1.0 in C:\Users\intel\AppData\Local\hermes\hermes-agent\venv\Lib\site-packages (from huggingface_hub>=0.21.0->accelerate->coolmind) (4.15.0)
Requirement already satisfied: colorama in C:\Users\intel\AppData\Local\hermes\hermes-agent\venv\Lib\site-packages (from click<9.0.0,>=8.4.2->huggingface_hub>=0.21.0->accelerate->coolmind) (0.4.6)
Requirement already satisfied: anyio in C:\Users\intel\AppData\Local\hermes\hermes-agent\venv\Lib\site-packages (from httpx<1,>=0.23.0->huggingface_hub>=0.21.0->accelerate->coolmind) (4.12.1)
Requirement already satisfied: certifi in C:\Users\intel\AppData\Local\hermes\hermes-agent\venv\Lib\site-packages (from httpx<1,>=0.23.0->huggingface_hub>=0.21.0->accelerate->coolmind) (2026.5.20)
Requirement already satisfied: httpcore==1.* in C:\Users\intel\AppData\Local\hermes\hermes-agent\venv\Lib\site-packages (from httpx<1,>=0.23.0->huggingface_hub>=0.21.0->accelerate->coolmind) (1.0.9)
Requirement already satisfied: idna in C:\Users\intel\AppData\Local\hermes\hermes-agent\venv\Lib\site-packages (from httpx<1,>=0.23.0->huggingface_hub>=0.21.0->accelerate->coolmind) (3.18)
Requirement already satisfied: h11>=0.16 in C:\Users\intel\AppData\Local\hermes\hermes-agent\venv\Lib\site-packages (from httpcore==1.*->httpx<1,>=0.23.0->huggingface_hub>=0.21.0->accelerate->coolmind) (0.16.0)
Requirement already satisfied: setuptools>=77.0.3 in C:\Users\intel\AppData\Local\hermes\hermes-agent\venv\Lib\site-packages (from torch->coolmind) (84.0.0)
Requirement already satisfied: sympy>=1.13.3 in C:\Users\intel\AppData\Local\hermes\hermes-agent\venv\Lib\site-packages (from torch->coolmind) (1.14.0)
Requirement already satisfied: networkx>=2.5.1 in C:\Users\intel\AppData\Local\hermes\hermes-agent\venv\Lib\site-packages (from torch->coolmind) (3.6.1)
Requirement already satisfied: jinja2 in C:\Users\intel\AppData\Local\hermes\hermes-agent\venv\Lib\site-packages (from torch->coolmind) (3.1.6)
Requirement already satisfied: mpmath<1.4,>=1.1.0 in C:\Users\intel\AppData\Local\hermes\hermes-agent\venv\Lib\site-packages (from sympy>=1.13.3->torch->coolmind) (1.3.0)
Requirement already satisfied: MarkupSafe>=2.0 in C:\Users\intel\AppData\Local\hermes\hermes-agent\venv\Lib\site-packages (from jinja2->torch->coolmind) (3.0.3)
Requirement already satisfied: regex>=2025.10.22 in C:\Users\intel\AppData\Local\hermes\hermes-agent\venv\Lib\site-packages (from transformers->coolmind) (2026.9.10)
Requirement already satisfied: tokenizers<0.24.0,>=0.23.1 in C:\Users\intel\AppData\Local\hermes\hermes-agent\venv\Lib\site-packages (from transformers->coolmind) (0.23.1)
Requirement already satisfied: typer in C:\Users\intel\AppData\Local\hermes\hermes-agent\venv\Lib\site-packages (from transformers->coolmind) (0.27.2)
Requirement already satisfied: shellingham>=1.3.0 in C:\Users\intel\AppData\Local\hermes\hermes-agent\venv\Lib\site-packages (from typer->transformers->coolmind) (1.5.4)
Requirement already satisfied: rich>=13.8.0 in C:\Users\intel\AppData\Local\hermes\hermes-agent\venv\Lib\site-packages (from typer->transformers->coolmind) (14.3.3)
Requirement already satisfied: annotated-doc>=0.0.2 in C:\Users\intel\AppData\Local\hermes\hermes-agent\venv\Lib\site-packages (from typer->transformers->coolmind) (0.0.4)
Requirement already satisfied: markdown-it-py>=2.2.0 in C:\Users\intel\AppData\Local\hermes\hermes-agent\venv\Lib\site-packages (from rich>=13.8.0->typer->transformers->coolmind) (4.0.0)
Requirement already satisfied: pygments<3.0.0,>=2.13.0 in C:\Users\intel\AppData\Local\hermes\hermes-agent\venv\Lib\site-packages (from rich>=13.8.0->typer->transformers->coolmind) (2.20.0)
Requirement already satisfied: mdurl~=0.1 in C:\Users\intel\AppData\Local\hermes\hermes-agent\venv\Lib\site-packages (from markdown-it-py>=2.2.0->rich>=13.8.0->typer->transformers->coolmind) (0.1.2)
Requirement already satisfied: pywin32 in C:\Users\intel\AppData\Local\hermes\hermes-agent\venv\Lib\site-packages (from wmi->coolmind) (311) to get the latest version
3. **Release Notes**: Check the [GitHub Releases page](https://github.com/Yoroc/CoolMind/releases) for detailed changelogs
