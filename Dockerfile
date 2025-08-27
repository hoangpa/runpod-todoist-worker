FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# Keep logs unbuffered and pip predictable
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Base system + modern pip (but <25 to avoid resolver churn)
RUN apt-get update && \
    apt-get install -y --no-install-recommends python3 python3-pip git && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    python3 -m pip install --upgrade "pip<25" setuptools wheel && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .

# Install the matching PyTorch stack FIRST (no deps), then the rest
# vLLM 0.7.x pairs cleanly with Torch 2.5.1 + cu121 wheels.
RUN pip install --no-deps --index-url https://download.pytorch.org/whl/cu121 \
      torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 \
 && pip install --no-cache-dir -r requirements.txt \
      --extra-index-url https://download.pytorch.org/whl/cu121 \
 && pip check

# Your worker
COPY handler.py .
CMD ["python3", "-u", "handler.py"]