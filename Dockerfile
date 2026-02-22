FROM python:3.11-slim AS base

# Install system dependencies for audio (soundfile, librosa) and Node.js
RUN apt-get update && apt-get install -y \
    curl \
    libsndfile1 \
    ffmpeg \
    build-essential \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Install Python package manager (uv)
RUN pip install uv

WORKDIR /app

# --- Frontend Build Stage ---
FROM base AS frontend-builder
WORKDIR /app/frontend
# Copy frontend dependency files
COPY frontend/package*.json ./
# Install dependencies
RUN npm ci
# Copy frontend source code
COPY frontend/ ./
# Build Next.js application (requires node and npm)
ENV NEXT_PUBLIC_BACKEND_ORIGIN=http://127.0.0.1:8000
RUN npm run build

# --- Final Image ---
FROM base
WORKDIR /app

# Hugging Face Spaces persistent storage path (when enabled)
ENV MODEL_PATH=/data/models/medgemma-27b-it-Q3_K_M.gguf
# Set MODEL_URL at runtime (Space Variables) to enable first-boot download
ENV MODEL_URL=
ENV INFERENCE_PROFILE=space
ENV STT_BACKEND=torch
ENV TTS_DEVICE=cuda
ENV HF_HOME=/data/.cache/huggingface
ENV HUGGINGFACE_HUB_CACHE=/data/.cache/huggingface/hub
ENV TRANSFORMERS_CACHE=/data/.cache/huggingface/transformers

# Copy built frontend from the builder stage
COPY --from=frontend-builder /app/frontend /app/frontend

# Copy backend source code
COPY backend /app/backend
WORKDIR /app/backend

# Install backend dependencies using uv
# --system flag installs to the system Python instead of a virtualenv
RUN uv sync --system || uv pip install --system -e .

WORKDIR /app

# Create a startup script to run both Next.js and FastAPI
RUN echo '#!/bin/bash\n\
set -euo pipefail\n\
\n\
# Download model to persistent storage on first boot (Hugging Face /data)\n\
if [ ! -f "$MODEL_PATH" ]; then\n\
  if [ -z "${MODEL_URL:-}" ]; then\n\
    echo "MODEL_URL is not set and model was not found at $MODEL_PATH"\n\
    exit 1\n\
  fi\n\
  mkdir -p "$(dirname "$MODEL_PATH")"\n\
  echo "Downloading model to $MODEL_PATH"\n\
  curl -fL "$MODEL_URL" -o "$MODEL_PATH"\n\
fi\n\
\n\
# Start the FastAPI backend in the background\n\
cd /app/backend\n\
export MODEL_PATH\n\
export MEDGEMMA_GGUF_PATH="$MODEL_PATH"\n\
PYTHONPATH=. uvicorn main:app --host 127.0.0.1 --port 8000 &\n\
\n\
# Start the Next.js frontend\n\
cd /app/frontend\n\
export BACKEND_ORIGIN=http://127.0.0.1:8000\n\
export PORT=7860\n\
exec npm run start\n\
' > /app/start.sh && chmod +x /app/start.sh

# Hugging Face Spaces exposes port 7860 by default
EXPOSE 7860
EXPOSE 8000

# Start both services
CMD ["/app/start.sh"]
