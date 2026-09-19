# CoolMind: Thermal-Aware AI Inference Engine

CoolMind is a thermal-aware local AI inference engine that prevents overheating by dynamically adjusting model quantization based on GPU temperature. Built for Windows with WMI-based thermal monitoring, it automatically optimizes performance while keeping your system cool.

##  Key Features

- **Thermal-Aware Intelligence** : Real-time GPU temperature monitoring with automatic quantization adjustment
- **Dynamic Quantization** : Switches between FP32, INT8, and INT4 based on temperature thresholds
- **Performance Optimized** : LRU caching, async support, and model pooling for efficient inference
- **System Protection** : Prevents overheating by offloading to CPU when temperatures rise
- **Enterprise Ready** : Docker, Kubernetes, and Helm chart support for cloud deployment
# CoolMind

![CI](https://github.com/Yoroc/CoolMind/actions/workflows/ci.yml/badge.svg)
![PyPI - Version](https://img.shields.io/pypi/v/coolmind?style=flat-square)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/coolmind?style=flat-square)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)

![Demo](./@yoro.svg)

Stop your GPU from melting when running local AI models.

---

##  Why this exists

I got tired of my PC sounding like a jet engine every time I tried to run a local LLM. This little tool watches your GPU temperature and automatically scales back the model when things get too hot, then spins it back up when it cools down.

No fancy PhD required - just works.

---

##  Get it running

```bash
pip install coolmind

# Or if you like living dangerously:
git clone https://github.com/Yoroc/CoolMind.git
cd CoolMind
pip install -e .
```

---

##  How to use

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

##  Make it yours (config.yaml)

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

##  What it actually does under the hood

This thing watches your GPU temp using Windows' built-in WMI (that's the part that makes it Windows-only for now). When things heat up:

- **50-55°C**: It quietly slips the AI model into a lighter version (INT8 quantization) 
- **55°C+**: It goes further with even lighter quantization or gets ready to move things to CPU
- **<50°C**: Puts the model back at full strength when it's cool enough
- **All day, every day**: Does this dance automatically in the background

The temp checking runs on its own little thread so it doesn't slow down your AI chats.

---

##  How it's put together

```
coolmind/
├── cli.py               # The command line thing you type
├── core/
│   ├── engine.py        # Where the AI and temp watching live
│   └── quantization.py  # The magic that makes AI models use less power
├__init__.py          # Says what parts are public
└── __main__.py          # Lets you run it with `python -m coolmind`
```

---

##  What you need to run this

- Windows 10 or 11 (sorry, Linux/macOS folks - WMI is Windows-only for now)
- Python 3.9 or newer
- An NVIDIA GPU helps but isn't required (will work on CPU too, just slower)
- At least 4GB of RAM (8GB+ recommended if you want to actually use it)
- Internet connection the first time it downloads an AI model

---

##  License

MIT License

Copyright (c) 2026 Yoroc

[The usual MIT license stuff - basically: use it however you want, don't sue me if it melts your cat]

---

##  Honest thoughts

This started because I was sick of my workspace turning into a sauna just to chat with an AI. It's not perfect - the temp watching is Windows-only for now and the quantization is basic - but it actually works for keeping your PC from sounding like it's about to take off.

If you find it useful, cool. If you've got ideas to make it better, even cooler. If it somehow catches your desk on fire... well, you were warned about the "don't sue me" part above.
