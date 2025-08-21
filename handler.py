# handler.py
import os
from vllm import LLM, EngineArgs
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.server.openai_api import OpenAIAPIHandler, LoRAModulePath
import runpod

# Get the model repository ID from environment variables
MODEL_REPO = os.environ.get("MODEL_NAME")
# Define the path on the persistent volume
MODEL_BASE_PATH = "/runpod-volume/models"

# Ensure the directory for the model exists
os.makedirs(MODEL_BASE_PATH, exist_ok=True)
model_path = os.path.join(MODEL_BASE_PATH, MODEL_REPO.replace("/", "--"))

# --- Download the model from Hugging Face if it's not already on the volume ---
if not os.path.exists(os.path.join(model_path, "config.json")):
    print(f"Model not found at {model_path}. Downloading...")
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=MODEL_REPO, local_dir=model_path, local_dir_use_symlinks=False)
    print("Download complete.")
else:
    print(f"Model already exists at {model_path}.")

# --- Configure and initialize the vLLM engine ---
# These arguments are equivalent to the vLLM CLI flags
engine_args = AsyncEngineArgs(
    model=model_path,
    tensor_parallel_size=1, # Use 1 for a single GPU
    dtype="auto",
    max_model_len=4096, # Adjust based on your needs
    gpu_memory_utilization=0.95, # Use 95% of GPU memory
    enforce_eager=False,
)
engine = AsyncLLMEngine.from_engine_args(engine_args)

# Create an OpenAI-compatible API handler
api_handler = OpenAIAPIHandler(
    engine,
    engine_args.served_model_names or [engine_args.model],
    "no-lora-module-path-needed", # Not using runtime LoRA
    None, # No chat template needed, will be in the request
)

# Define the RunPod handler function
async def handler(job):
    # Pass the job input directly to the OpenAI API handler
    return await api_handler.handle_request(job['input'])

# Start the RunPod worker
runpod.serverless.start({
    "handler": handler,
    "concurrency_modifier": lambda x: 128,
})