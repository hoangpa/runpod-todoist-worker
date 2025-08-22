# handler.py (Robust Version 2)
import os
import sys
from huggingface_hub import login, snapshot_download
from vllm import AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.server.openai_api import OpenAIAPIHandler
import runpod

print("--- Starting Worker ---")

# --- Step 1: Log in to Hugging Face ---
HF_TOKEN = os.environ.get("HF_TOKEN")
if HF_TOKEN:
    print("Logging into Hugging Face Hub...")
    try:
        login(token=HF_TOKEN)
        print("Hugging Face login successful.")
    except Exception as e:
        print(f"Hugging Face login failed: {e}")
        # Exit if we can't log in, as we won't be able to download the model
        sys.exit(1)
else:
    print("Warning: HF_TOKEN not set. Assuming public model.")

# --- Step 2: Get Model Configuration ---
MODEL_REPO = os.environ.get("MODEL_NAME")
if not MODEL_REPO:
    print("FATAL: MODEL_NAME environment variable not set.")
    sys.exit(1)

MODEL_BASE_PATH = "/runpod-volume/models"
print(f"Model repository: {MODEL_REPO}")
print(f"Volume path: {MODEL_BASE_PATH}")

# --- Step 3: Download Model if Necessary ---
os.makedirs(MODEL_BASE_PATH, exist_ok=True)
model_path = os.path.join(MODEL_BASE_PATH, MODEL_REPO.replace("/", "--"))

if not os.path.exists(os.path.join(model_path, "config.json")):
    print(f"Model not found at {model_path}. Starting download...")
    try:
        snapshot_download(repo_id=MODEL_REPO, local_dir=model_path, local_dir_use_symlinks=False)
        print("Download complete.")
    except Exception as e:
        print(f"FATAL: Model download failed: {e}")
        sys.exit(1)
else:
    print(f"Model already exists at {model_path}.")

# --- Step 4: Configure and Initialize vLLM ---
print("Configuring vLLM engine...")
engine_args = AsyncEngineArgs(
    model=model_path,
    tensor_parallel_size=1,
    dtype="auto",
    max_model_len=4096,
    gpu_memory_utilization=0.95,
    enforce_eager=False,
)

try:
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    print("vLLM engine initialized successfully.")
except Exception as e:
    print(f"FATAL: vLLM engine initialization failed: {e}")
    sys.exit(1)

# --- Step 5: Set up API Handler ---
print("Setting up OpenAI API handler...")
api_handler = OpenAIAPIHandler(
    engine,
    engine_args.served_model_names or [engine_args.model],
    "no-lora-module-path-needed",
    None,
)
print("API handler is ready.")

# --- Step 6: Define RunPod Handler and Start Server ---
async def handler(job):
    print("Received a job request.")
    return await api_handler.handle_request(job['input'])

print("Starting RunPod serverless worker...")
runpod.serverless.start({
    "handler": handler,
    "concurrency_modifier": lambda x: 128,
})