# PASCO Motors phone agent (LiveKit Cloud).
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # Model files live under /app, which the runtime user owns.
    HF_HOME=/app/.cache/huggingface

# LiveKit Cloud runs the container as a non-root user.
RUN useradd -m -u 10001 appuser
WORKDIR /app

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY *.py ./
COPY knowledge/ ./knowledge/

# Bake the VAD and English turn-detector models into the image.
RUN python agent.py download-files

RUN mkdir -p /app/data && chown -R appuser:appuser /app
USER appuser

CMD ["python", "agent.py", "start"]
