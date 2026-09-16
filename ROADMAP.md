# CoolMind Roadmap & Enhancement Ideas

## Immediate Enhancements (Next 1-2 weeks)

### 1. Aggressive Model Quantization When Hot
- **Dynamic quantization based on temperature**
  - Below 60°C: Use FP16/FP32 for quality
  - 60-75°C: Use INT8 dynamic quantization
  - Above 75°C: Use INT4 quantization or CPU offload
- **Implementation**: 
  - Add quantization state tracking to `CoolEngine`
  - Use `torch.quantization` for dynamic quant
  - Cache quantized models to avoid recomputation

### 2. Web API Interface
- **FastAPI-based REST API**
  - Endpoint: `POST /generate` with JSON payload
  - Endpoint: `GET /status` for thermal/device state
  - Endpoint: `POST /offload` and `POST /reload` for manual control
- **Features**:
  - Automatic docs at `/docs`
  - CORS support for web interfaces
  - Authentication middleware (API key)
  - Background task thermal monitoring

### 3. Desktop GUI Application
- **Electron + React** (or **PySimpleGUI** for pure Python)
  - Chat interface with message history
  - Real-time temperature gauge
  - Model selector dropdown
  - Settings panel for thresholds
  - System tray integration
- **Alternative**: Use Hermes Agent's desktop capabilities as base

### 4. Support for Model Pipelines
- **Beyond text generation**:
  - Text classification pipeline
  - Question answering pipeline
  - Summarization pipeline
  - Translation pipeline
  - Feature extraction (embeddings)
- **Implementation**:
  - Pipeline registry in `CoolEngine`
  - Task-type detection from input
  - Automatic pipeline selection/caching

## Technical Implementation Plan

### Phase 1: Quantization Enhancement (Week 1)
1. Add `quantization_level` property to `CoolEngine`
2. Implement temperature-based quantization switching
3. Add model caching for different quantization states
4. Update CLI/API to show quantization status

### Phase 2: Web API (Week 2)
1. Create `coolmind/api.py` with FastAPI app
2. Implement core endpoints (/generate, /status)
3. Add thermal monitoring as background task
4. Test with curl/Postman
5. Add Dockerfile for containerization

### Phase 3: Desktop GUI (Week 3)
1. Choose framework (Electron vs PySimpleGUI)
2. Implement basic chat window
3. Add real-time temp visualization
4. Integrate with CoolMind engine
5. Package as executable

### Phase 4: Pipeline Support (Week 4)
1. Add `PipelineType` enum
2. Implement pipeline factory
3. Add task detection heuristics
4. Update generate() to route to correct pipeline
5. Add CLI flag for task type: `--task qa`

## Files to Modify/Create

### New Files:
- `coolmind/quantization.py` - Quantization utilities
- `coolmind/api.py` - FastAPI web server
- `coolmind/gui/` - Desktop GUI components
- `coolmind/pipelines/` - Pipeline implementations

### Modified Files:
- `coolmind/core/engine.py` - Add quantization/pipeline support
- `coolmind/cli.py` - Add new CLI options
- `pyproject.toml` - Add new dependencies (fastapi, uvicorn, etc.)

## Dependencies to Add
```
fastapi
uvicorn[standard]
pydantic
python-multipart  # for file uploads
# Optional GUI:
PySimpleGUI  # or electron + node deps
# Optional quantization:
bitsandbytes  # for 4-bit quantization
```

## Example Usage After Enhancements

```bash
# Web API mode
coolmind-api --model gpt2 --port 8000

# Desktop app
coolmind-gui

# Quantization info in CLI
coolmind -q "Explain AI" --show-quant

# Pipeline support
coolmind --task qa -q "What is CoolMind?" --context "It's a thermal-aware AI engine"
```

## Risk Mitigation
- Quantization: Fallback to original model if quant fails
- API: Rate limiting and request size limits
- GUI: Graceful degradation to CLI if GUI deps missing
- Pipelines: Default to text generation on unknown task