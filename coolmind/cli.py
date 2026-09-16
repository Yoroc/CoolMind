#!/usr/bin/env python3
"""CoolMind CLI - Thermal-aware AI inference engine"""

import argparse
import yaml
import logging
import sys
from pathlib import Path
from .core.engine import CoolEngine, ThermalConfig

def setup_logging(config):
    """Setup logging based on config"""
    log_level = config.get('logging', {}).get('level', 'INFO')
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def load_config(config_path: Path = None) -> dict:
    """Load configuration from file or return defaults"""
    default_config = {
        'model': {
            'name': 'sshleifer/tiny-gpt2',
            'max_length': 50,
            'temperature': 0.7,
            'top_p': 0.9,
            'do_sample': True
        },
        'thermal': {
            'max_gpu_temp': 80.0,
            'cooldown_threshold': 50.0,
            'offload_threshold': 55.0,
            'update_interval': 2.0,
            'enable_monitoring': True
        },
        'logging': {
            'level': 'INFO'
        },
        'inference': {
            'max_length': 50,
            'temperature': 0.7,
            'top_p': 0.9,
            'do_sample': True
        }
    }
    
    if config_path is None or not config_path.exists():
        return default_config
    
    try:
        with open(config_path, 'r') as f:
            user_config = yaml.safe_load(f)
        
        # Merge with defaults
        for key, value in user_config.items():
            if key in default_config and isinstance(default_config[key], dict):
                default_config[key].update(value)
            else:
                default_config[key] = value
        
        return default_config
    except Exception as e:
        logging.warning(f"Failed to load config from {config_path}: {e}")
        return default_config

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(description="CoolMind - Thermal-aware AI inference engine")
    parser.add_argument("-m", "--model", help="Model name or path")
    parser.add_argument("-q", "--query", help="Single query mode (exit after response)")
    parser.add_argument("--max-length", type=int, help="Maximum generation length")
    parser.add_argument("--temperature", type=float, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, help="Top-p sampling parameter")
    parser.add_argument("--no-monitor", action="store_true", help="Disable thermal monitoring")
    parser.add_argument("--show-status", action="store_true", help="Show model status after generation")
    parser.add_argument("--config", type=Path, help="Path to configuration file")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    setup_logging(config)
    
    # Override config with command line arguments
    model_name = args.model or config['model']['name']
    
    # Create thermal config from file
    thermal_config = ThermalConfig(
        max_gpu_temp=config['thermal']['max_gpu_temp'],
        offload_threshold=config['thermal']['offload_threshold'],
        cooldown_threshold=config['thermal']['cooldown_threshold'],
        update_interval=config['thermal']['update_interval'],
        enable_monitoring=config['thermal']['enable_monitoring'] and not args.no_monitor
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Starting CoolMind with model: {model_name}")
    
    # Initialize engine
    engine = CoolEngine(model_name=model_name, config=thermal_config)
    
    if not args.no_monitor:
        engine.start_monitoring()
        logger.info("Thermal monitoring started")
    
    try:
        # Get inference parameters
        max_length = args.max_length or config['inference']['max_length']
        temperature = args.temperature or config['inference']['temperature']
        top_p = config['inference']['top_p']
        do_sample = config['inference']['do_sample']
        
        if args.query:
            # Single query mode
            response = engine.generate(
                args.query,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample
            )
            print(response)
            
            if args.show_status:
                status = engine.get_status()
                print("\n--- Model Status ---")
                print(f"Temperature: {status.temperature:.1f}°C")
                print(f"Device: {status.device}")
                print(f"Quantized: {status.quantized} ({status.quantization_level})")
                
        else:
            # Interactive mode
            print("CoolMind AI Ready. Type 'exit' or 'quit' to stop.")
            print("Commands:")
            print("  exit/quit - Stop the application")
            print("  stats     - Show current temperature and device info")
            print("  help      - Show this help")
            print()
            
            while True:
                try:
                    prompt = input("\nYou: ").strip()
                    
                    if prompt.lower() in ['exit', 'quit']:
                        break
                    elif prompt.lower() == 'help':
                        print("Commands:")
                        print("  exit/quit - Stop the application")
                        print("  stats     - Show current temperature and device info")
                        print("  help      - Show this help")
                        continue
                    elif prompt.lower() == 'stats':
                        status = engine.get_status()
                        print(f"\n--- Current Status ---")
                        print(f"Temperature: {status.temperature:.1f}°C")
                        print(f"Device: {status.device}")
                        print(f"Quantized: {status.quantized} ({status.quantization_level})")
                        continue
                    elif not prompt:
                        continue
                    
                    # Generate response
                    response = engine.generate(
                        prompt,
                        max_length=max_length,
                        temperature=temperature,
                        top_p=top_p,
                        do_sample=do_sample
                    )
                    print(f"AI: {response}")
                    
                    if args.show_status:
                        status = engine.get_status()
                        print(f"\n--- Status ---")
                        print(f"Temp: {status.temperature:.1f}°C | "
                              f"Device: {status.device} | "
                              f"Quant: {status.quantization_level}")
                    
                except KeyboardInterrupt:
                    print("\nInterrupted. Type 'exit' to quit.")
                    continue
                except EOFError:
                    break
    
    finally:
        if not args.no_monitor:
            engine.stop_monitoring()
            logger.info("Thermal monitoring stopped")
        logger.info("CoolMind shutting down")

if __name__ == "__main__":
    main()