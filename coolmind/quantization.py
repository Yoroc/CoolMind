"""
Quantization utilities for CoolMind
Handles dynamic model quantization based on thermal conditions
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

def apply_dynamic_quantization(model: nn.Module) -> nn.Module:
    """
    Apply dynamic quantization to a model
    Returns quantized model
    """
    try:
        # Quantize linear layers dynamically
        quantized_model = torch.quantization.quantize_dynamic(
            model, 
            {nn.Linear}, 
            dtype=torch.qint8
        )
        return quantized_model
    except Exception as e:
        # If quantization fails, return original model
        print(f"Quantization failed: {e}")
        return model

def get_quantization_level(current_temp: float, offload_threshold: float, cooldown_threshold: float) -> str:
    """
    Determine what quantization level to use based on temperature
    
    Returns:
        'none' - No quantization (temp < cooldown_threshold)
        'int8' - INT8 dynamic quantization (cooldown_threshold <= temp < offload_threshold)
        'int4' - More aggressive quantization/offloading (temp >= offload_threshold)
    """
    if current_temp < cooldown_threshold:
        return 'none'
    elif current_temp < offload_threshold:
        return 'int8'
    else:
        return 'int4'

def should_offload_instead_of_quantize(current_temp: float, offload_threshold: float) -> bool:
    """
    Determine if we should offload to CPU instead of quantizing
    For now, we'll offload when temperature exceeds offload_threshold
    """
    return current_temp >= offload_threshold

def get_model_size_info(model: torch.nn.Module) -> Tuple[int, float]:
    """
    Calculate model size in parameters and MB
    Returns: (parameter_count, size_in_mb)
    """
    param_count = sum(p.numel() for p in model.parameters())
    # Estimate size: 4 bytes per parameter for FP32, adjust for actual dtype
    # This is a rough estimate
    size_mb = param_count * 4 / (1024 * 1024)  # Assuming FP32 for simplicity
    return param_count, size_mb