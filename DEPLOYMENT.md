# CoolMind Deployment Guide

## 🐳 Docker Deployment

### Quick Start
```bash
# Build the image
docker build -t coolmind:latest .

# Run basic CLI mode
docker run --rm coolmind:latest --help

# Run interactive mode
docker run -it --rm coolmind:latest

# Run with specific query
docker run --rm coolmind:latest --query "What is AI?" --max-length 30

# Run web API mode
docker run -d -p 8000:8000 --name coolmind-api coolmind:latest web-api
```

### Docker Compose
```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

## ☸️ Kubernetes Deployment

### Simple Deployment
```bash
# Apply the deployment
kubectl apply -f k8s-deployment.yaml

# Check status
kubectl get pods -l app=coolmind

# Expose for testing (development only)
kubectl port-forward svc/coolmind-service 8080:80
```

### Helm Chart Installation
```bash
# Install dependencies
helm dependency update coolmind-chart/

# Install the chart
helm install coolmind ./coolmind-chart/

# Upgrade existing release
helm upgrade coolmind ./coolmind-chart/

# Uninstall
helm uninstall coolmind
```

### Custom Values
Create `my-values.yaml`:
```yaml
replicaCount: 5
resources:
  limits:
    cpu: "1000m"
    memory: "2Gi"
  requests:
    cpu: "500m"
    memory: "1Gi"

# Then install with:
helm install coolmind ./coolmind-chart/ -f my-values.yaml
```

## 🔧 Configuration

### Environment Variables
- `MODEL`: Hugging Face model name (default: sshleifer/tiny-gpt2)
- `PIPELINE`: Pipeline type (text-generation, qa, question-answering, summarization, summarize)
- `QUERY`: For single-query mode
- `MAX_LENGTH`: Maximum generation length
- `TEMPERATURE`: Sampling temperature (0.0-2.0)
- `TOP_P`: Top-p sampling parameter (0.0-1.0)
- `NO_MONITOR`: Disable thermal monitoring (true/false)
- `SHOW_STATUS`: Show model status after generation
- `CONFIG`: Path to configuration file

### Volume Mounts for Production
- `/root/.cache/huggingface`: Model cache persistence
- `/app/logs`: Application logs
- `/app/data`: Persistent data storage

## 📊 Monitoring & Scaling

### Health Checks
- Endpoint: `GET /health` (when web API enabled)
- Returns: `{"status": "healthy", "service": "coolmind-api"}`

### Metrics Available
- Cache hit/miss ratios (via Python API)
- Model loading statistics
- Inference latency measurements
- Resource utilization tracking

### Auto-scaling Triggers
- CPU utilization > 70%
- Memory utilization > 80%
- Custom metrics (when web API enabled)

## 🏗️ Production Best Practices

### Resource Management
- Set appropriate CPU/memory limits based on model size
- Use node affinity for GPU-enabled nodes when available
- Consider pod disruption budgets for high availability

### Security
- Run as non-root user (implemented)
- Read-only root filesystem (consider for high-security)
- Secrets management for model access tokens
- Network policies to restrict access

### Logging & Debugging
- Structured logging for easier parsing
- Log rotation and retention policies
- Debug mode for development
- Error tracking integration ready

## 🔄 Update Strategy

### Rolling Updates
- Kubernetes: Configure maxSurge and maxUnavailable
- Docker Swarm: Update delay and parallelism
- Blue/Green: Separate services for validation

### Backup & Restore
- Model cache persistence via volumes
- Configuration backup
- Log aggregation for audit trails

## 🛠️ Troubleshooting

### Common Issues
1. **Model download failures**: Check network and HF token
2. **Out of memory**: Reduce batch size or increase limits
3. **Slow first request**: Model loading time (subsequent requests cached)
4. **Permission issues**: Volume mount permissions

### Diagnostic Commands
```bash
# Check container logs
docker logs <container-id>

# Execute shell in running container
docker exec -it <container-id> /bin/bash

# Check Kubernetes pod status
kubectl describe pod <pod-name>

# Get detailed metrics
kubectl top pod <pod-name>
```

