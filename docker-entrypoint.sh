#!/bin/bash
# CoolMind Container Startup Script
# Provides easy entry points for different usage patterns

set -e

echo "🚀 Starting CoolMind Container..."

# Default values
MODEL="${MODEL:-sshleifer/tiny-gpt2}"
PIPELINE="${PIPELINE:-text-generation}"
QUERY="${QUERY:-}"
MAX_LENGTH="${MAX_LENGTH:-50}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.9}"
NO_MONITOR="${NO_MONITOR:-false}"
SHOW_STATUS="${SHOW_STATUS:-false}"
CONFIG="${CONFIG:-}"

# Build command
CMD="python -m coolmind"
CMD+=" --model \"$MODEL\""
CMD+=" --pipeline \"$PIPELINE\""
CMD+=" --max-length $MAX_LENGTH"
CMD+=" --temperature $TEMPERATURE"
CMD+=" --top-p $TOP_P"

if [ "$NO_MONITOR" = "true" ]; then
    CMD+=" --no-monitor"
fi

if [ "$SHOW_STATUS" = "true" ]; then
    CMD+=" --show-status"
fi

if [ -n "$QUERY" ]; then
    CMD+=" --query \"$QUERY\""
fi

if [ -n "$CONFIG" ]; then
    CMD+=" --config \"$CONFIG\""
fi

# Add web API mode if requested
if [ "$1" = "web-api" ]; then
    echo "🌐 Starting CoolMind Web API Server..."
    exec python -m coolmind.web.server --host 0.0.0.0 --port 8000 "$@"
elif [ "$1" = "help" ] || [ "$1" = "--help" ]; then
    echo "📖 CoolMind Help:"
    exec python -m coolmind --help
else
    echo "⚡ Starting CoolMind CLI..."
    echo "   Model: $MODEL"
    echo "   Pipeline: $PIPELINE"
    if [ -n "$QUERY" ]; then
        echo "   Query: $QUERY"
    fi
    echo "   Command: $CMD"
    echo ""
    exec $CMD
fi

