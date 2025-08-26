FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# sensible defaults
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# base tools + up-to-date pip/build tooling
RUN apt-get update && \
    apt-get install -y --no-install-recommends python3 python3-pip git && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    python3 -m pip install --upgrade pip setuptools wheel && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .

# 1) Install CUDA 12.1 PyTorch stack WITHOUT deps to avoid pulling numpy==2.x
# 2) Then install the rest (with deps), pointing at the cu121 index as an extra source
RUN pip install --no-deps --index-url https://download.pytorch.org/whl/cu121 \
      torch==2.3.1+cu121 torchvision==0.18.1+cu121 torchaudio==2.3.1+cu121 \
 && pip install --no-cache-dir -r requirements.txt \
      --extra-index-url https://download.pytorch.org/whl/cu121 \
 && pip check

COPY handler.py .

CMD ["python3", "-u", "handler.py"]