# Use the official vLLM image as a base
FROM vllm/vllm-openai:latest

# Set the working directory inside the container
WORKDIR /app

# Copy your requirements file into the container
COPY requirements.txt .

# Install your specific Python packages
# --no-cache-dir is a good practice for smaller image sizes
RUN pip install --no-cache-dir -r requirements.txt

# Copy your handler code into the container
COPY handler.py .

# This is the command that will be run when the container starts
# It executes your handler script.
CMD ["python", "-u", "handler.py"]