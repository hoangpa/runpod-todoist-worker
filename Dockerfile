# --- START OF FIX ---
# Use a generic NVIDIA CUDA base image instead of the vLLM-specific one.
# This gives us a clean slate with Python and CUDA ready to go.
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04
# --- END OF FIX ---

# Install system dependencies like python3 and pip
RUN apt-get update && \
    apt-get install -y python3 python3-pip git && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Copy your requirements file into the container
COPY requirements.txt .

# Install your specific Python packages
# We need to specify the correct extra-index-url for torch on CUDA 12.1
RUN pip3 install --no-cache-dir -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu121

# Copy your handler code into the container
COPY handler.py .

# This is the command that will be run when the container starts
# It will now work correctly because there's no conflicting entrypoint.
CMD ["python3", "-u", "handler.py"]