import torch
import time
import threading
from typing import Optional
from dataclasses import dataclass
import platform
import logging

logger = logging.getLogger(__name__)

try:
    import wmi
    WMI_AVAILABLE = True
except ImportError:
    WMI_AVAILABLE = False

from ..quantization import apply_dynamic_quantization, get_quantization_level

@dataclass
class ThermalConfig:
    """Configuration for thermal-aware behavior"""
    max_gpu_temp: float = 80.0          # Temperature to trigger quantization/offload
    cooldown_threshold: float = 50.0    # Below this: no quantization
    offload_threshold: float = 55.0     # At/below this: INT8, above: INT4
    update_interval: float = 2.0        # How often to check temperature (seconds)
    enable_monitoring: bool = True      # Whether to start thermal monitoring thread

class CoolEngine:
    """Thermal-aware AI inference engine"""
    
    def __init__(self, model_name: str = "sshleifer/tiny-gpt2", config: Optional[ThermalConfig] = None):
        self.model_name = model_name
        self.config = config or ThermalConfig()
        self.tokenizer = None
        self.model = None
        self.device = None
        self._is_quantized = False
        self._quantization_level: Optional[str] = None  # 'none', 'int8', 'int4'
        self._wmi_interface = None
        self._monitoring = False
        self._monitor_thread = None
        self._current_temp = 0.0
        
        self._initialize()
    
    def _initialize(self):
        """Initialize the engine components"""
        # Initialize WMI if available and on Windows
        if WMI_AVAILABLE and platform.system() == "Windows":
            try:
                self._wmi_interface = wmi.WMI(namespace="root\\WMI")
                logger.debug("WMI interface initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize WMI: {e}")
                self._wmi_interface = None
        
        # Load model
        self._load_model()
        
        # Start monitoring if enabled
        if self.config.enable_monitoring:
            self.start_monitoring()
    
    def _load_model(self):
        """Load the model and tokenizer"""
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        logger.info(f"Loading model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        
        # Set pad token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Determine device
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        
        self.model.to(self.device)
        logger.info(f"Model loaded on {self.device}")
    
    def _get_temperature(self) -> float:
        """Get current GPU temperature in Celsius"""
        if not self._wmi_interface:
            # Simulate temperature for testing/demo purposes
            import random
            return random.uniform(30.0, 70.0)
        
        try:
            for sensor in self._wmi_interface.MSAcpi_ThermalZoneTemperature():
                # Convert from tenths of Kelvin to Celsius
                temp_celsius = (sensor.CurrentTemperature / 10.0) - 273.15
                if 0 <= temp_celsius <= 125:  # Reasonable range for GPU temp
                    return temp_celsius
        except Exception as e:
            logger.debug(f"Error reading temperature: {e}")
        
        return 0.0  # Default if reading fails
    
    def _get_quantization_level_for_temp(self, temp: float) -> str:
        """Determine quantization level based on temperature"""
        return get_quantization_level(
            temp, 
            self.config.offload_threshold, 
            self.config.cooldown_threshold
        )
    
    def _apply_quantization_if_needed(self):
        """Apply quantization based on current temperature"""
        if not self._monitoring:
            return
            
        current_temp = self._get_temperature()
        self._current_temp = current_temp
        
        target_quant_level = self._get_quantization_level_for_temp(current_temp)
        
        # Only re-quantize if level changed and we're not offloading
        if target_quant_level != self._quantization_level and target_quant_level != 'offload':
            logger.info(f"Temperature: {current_temp:.1f}°C, Target quantization: {target_quant_level}")
            
            try:
                if target_quant_level == 'int8' and not self._is_quantized:
                    self.model = apply_dynamic_quantization(self.model)
                    self._is_quantized = True
                    self._quantization_level = 'int8'
                    logger.info("Applied INT8 dynamic quantization")
                    
                elif target_quant_level == 'int4' and not self._is_quantized:
                    # For INT4, we'd typically use bitsandbytes or similar
                    # For now, we'll simulate by noting the intent
                    self._is_quantized = True
                    self._quantization_level = 'int4'
                    logger.info("Applied INT4 quantization (simulated)")
                    
                elif target_quant_level == 'none' and self._is_quantized:
                    # Reload original model (simplified - in practice would reload)
                    self._load_model()  # Reloads original precision model
                    self._is_quantized = False
                    self._quantization_level = 'none'
                    logger.info("Reloaded model at full precision")
                    
            except Exception as e:
                logger.error(f"Failed to apply quantization: {e}")
    
    def _monitoring_loop(self):
        """Background thread to monitor temperature and adjust quantization"""
        logger.info("Starting thermal monitoring thread")
        while self._monitoring:
            self._apply_quantization_if_needed()
            time.sleep(self.config.update_interval)
        logger.info("Thermal monitoring thread stopped")
    
    def start_monitoring(self):
        """Start the thermal monitoring thread"""
        if not self._monitoring:
            self._monitoring = True
            self._monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self._monitor_thread.start()
            logger.info("Thermal monitoring started")
    
    def stop_monitoring(self):
        """Stop the thermal monitoring thread"""
        if self._monitoring:
            self._monitoring = False
            if self._monitor_thread:
                self._monitor_thread.join(timeout=5.0)
            logger.info("Thermal monitoring stopped")
    
    def generate(self, prompt: str, max_length: int = 50, **kwargs) -> str:
        """Generate text from prompt"""
        if self.tokenizer is None or self.model is None:
            raise RuntimeError("Model not initialized")
        
        # Apply quantization if needed (non-blocking check)
        self._apply_quantization_if_needed()
        
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Set generation parameters
        gen_kwargs = {
            "max_length": max_length,
            "do_sample": True,
            "temperature": 0.7,
            "pad_token_id": self.tokenizer.eos_token_id,
            **kwargs
        }
        
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def get_status(self):
        """Get current engine status"""
        from types import SimpleNamespace
        return SimpleNamespace(
            temperature=self._current_temp,
            device=str(self.device),
            quantized=self._is_quantized,
            quantization_level=self._quantization_level or 'none'
        )