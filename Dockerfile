FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends python3 python3-pip git && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    python3 -m pip install --upgrade pip && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .

# good: force PyTorch GPU wheels first, then everything else
RUN pip install --index-url https://download.pytorch.org/whl/cu121 \
        torch==2.3.1+cu121 torchvision==0.18.1+cu121 torchaudio==2.3.1+cu121 \
    && pip install -r requirements.txt

COPY handler.py .

CMD ["python3", "-u", "handler.py"]