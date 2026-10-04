#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

# Ensure uv is in PATH
if ! command -v uv &> /dev/null; then
    if [ -x "$HOME/.local/bin/uv" ]; then
        export PATH="$HOME/.local/bin:$PATH"
    else
        echo "❌ Error: 'uv' command not found." >&2
        echo "Please ensure 'uv' is installed and in your PATH." >&2
        echo "If you just ran setup.sh, try restarting your terminal." >&2
        exit 1
    fi
fi

# Dynamically add pip-installed CUDA libraries to LD_LIBRARY_PATH
CUDA_LIB_PATH=$(uv run python -c "
try:
    import os, nvidia.cublas.lib, nvidia.cudnn.lib
    print(os.path.dirname(nvidia.cublas.lib.__file__) + ':' + os.path.dirname(nvidia.cudnn.lib.__file__))
except Exception:
    pass
" 2>/dev/null || true)
if [ -n "$CUDA_LIB_PATH" ]; then
    export LD_LIBRARY_PATH="$CUDA_LIB_PATH:$LD_LIBRARY_PATH"
fi

# Load environment variables from .env if present
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs -r) 2>/dev/null || true
fi

# Load runtime environment configurations
if [ -f .env.runtime ]; then
    export $(grep -v '^#' .env.runtime | xargs -r) 2>/dev/null || true
else
    export WHISPER_MODEL="tiny"
    export WHISPER_DEVICE="cpu"
    export WHISPER_COMPUTE_TYPE="int8"
fi

MODE="live"
AUDIO_FILE=""
QUERY_TEXT=""
DURATION="5.0"
LANGUAGE="hi"
AUTO_STOP=""
AUDIO_DEVICE=""
STT_PROVIDER="${STT_PROVIDER:-}"
EXTRA_ARGS=()

show_usage() {
    echo "================================================================="
    echo "Voice-Enabled Indic RAG — Runner Script (Personal Project Personal Project)"
    echo "================================================================="
    echo "Usage: ./run.sh [MODE / OPTIONS]"
    echo ""
    echo "Modes:"
    echo "  --live, --mic           Start Live Voice Command mode (default)"
    echo "  --file <path>           Execute voice query from an audio file (.wav/.mp3)"
    echo "  --text \"<query>\"        Execute direct text query in Hindi/English"
    echo "  --server                Launch the live FastAPI Web Server on port 7860"
    echo "  --bench                 Run the sub-200ms latency benchmark suite"
    echo "  --test                  Run the comprehensive pytest test suite"
    echo "  --devices               List all available microphone input devices"
    echo ""
    echo "Live Voice Options:"
    echo "  -d, --duration <sec>    Voice recording duration in seconds (default: 5.0)"
    echo "  --vad, --auto-stop      Auto-stop recording when speech pause is detected"
    echo "  -l, --lang <code>       Language code: 'hi', 'en', 'mr', 'ta', etc. (default: hi)"
    echo "  -p, --provider <name>   STT Provider: 'sarvam', 'elevenlabs', 'faster_whisper', 'mock'"
    echo "  --audio-device <idx>    Microphone device index or substring name"
    echo "  -h, --help              Show this help message"
    echo "================================================================="
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --live|--mic)
            MODE="live"; shift ;;
        --file)
            MODE="file"; AUDIO_FILE="$2"; shift 2 ;;
        --text)
            MODE="text"; QUERY_TEXT="$2"; shift 2 ;;
        --server)
            MODE="server"; shift ;;
        --bench)
            MODE="bench"; shift ;;
        --test)
            MODE="test"; shift ;;
        --devices|--list-devices)
            MODE="devices"; shift ;;
        -d|--duration)
            DURATION="$2"; shift 2 ;;
        --vad|--auto-stop)
            AUTO_STOP="--auto-stop"; shift ;;
        -l|--lang|--language)
            LANGUAGE="$2"; shift 2 ;;
        -p|--provider|--stt-provider)
            STT_PROVIDER="$2"; shift 2 ;;
        --audio-device)
            AUDIO_DEVICE="$2"; shift 2 ;;
        -h|--help)
            show_usage; exit 0 ;;
        *)
            EXTRA_ARGS+=("$1"); shift ;;
    esac
done

case "$MODE" in
    live)
        echo "==> Starting Live Voice Command Mode (Lang: ${LANGUAGE}, Duration: ${DURATION}s, STT: ${STT_PROVIDER:-default})"
        CMD=(uv run python scripts/demo_cli.py --live --duration "${DURATION}" --language "${LANGUAGE}" --stt-model "${WHISPER_MODEL}" --device "${WHISPER_DEVICE}" --compute-type "${WHISPER_COMPUTE_TYPE}")
        if [ -n "$AUTO_STOP" ]; then CMD+=("$AUTO_STOP"); fi
        if [ -n "$STT_PROVIDER" ]; then CMD+=(--stt-provider "${STT_PROVIDER}"); fi
        if [ -n "$AUDIO_DEVICE" ]; then CMD+=(--audio-device "${AUDIO_DEVICE}"); fi
        "${CMD[@]}" "${EXTRA_ARGS[@]}"
        ;;
    file)
        echo "==> Running Voice RAG from audio file: ${AUDIO_FILE}"
        CMD=(uv run python scripts/demo_cli.py --audio-file "${AUDIO_FILE}" --language "${LANGUAGE}" --stt-model "${WHISPER_MODEL}" --device "${WHISPER_DEVICE}" --compute-type "${WHISPER_COMPUTE_TYPE}")
        if [ -n "$STT_PROVIDER" ]; then CMD+=(--stt-provider "${STT_PROVIDER}"); fi
        "${CMD[@]}" "${EXTRA_ARGS[@]}"
        ;;
    text)
        echo "==> Running Text RAG Query: ${QUERY_TEXT}"
        uv run python scripts/demo_cli.py --query-text "${QUERY_TEXT}" --language "${LANGUAGE}" \
            --stt-model "${WHISPER_MODEL}" --device "${WHISPER_DEVICE}" --compute-type "${WHISPER_COMPUTE_TYPE}" "${EXTRA_ARGS[@]}"
        ;;
    server)
        echo "==> Launching Voice RAG FastAPI Web Server on http://0.0.0.0:7860"
        uv run uvicorn src.api.server:app --host 0.0.0.0 --port 7860 --reload
        ;;
    bench)
        echo "==> Running Voice RAG Latency Benchmark Suite"
        uv run python benchmarks/run_latency_bench.py "${EXTRA_ARGS[@]}"
        ;;
    test)
        echo "==> Running Test Suite"
        uv run pytest -v "${EXTRA_ARGS[@]}"
        ;;
    devices)
        uv run python scripts/demo_cli.py --list-devices
        ;;
esac

