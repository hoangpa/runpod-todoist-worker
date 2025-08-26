FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends python3 python3-pip git && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    python3 -m pip install --upgrade "pip<25" setuptools wheel && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .

# Install the matching Torch stack FIRST (no deps), aligned with vLLM 0.5.4
RUN pip install --no-deps --index-url https://download.pytorch.org/whl/cu121 \
      torch==2.4.0+cu121 torchvision==0.19.0+cu121 torchaudio==2.4.0+cu121 \
 && pip install --no-cache-dir -r requirements.txt \
      --extra-index-url https://download.pytorch.org/whl/cu121 \
 && pip check

COPY handler.py .
CMD ["python3", "-u", "handler.py"]