# handler.py (Robust Version 2)
import os
import sys
from huggingface_hub import login, snapshot_download
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.serving_completion import OpenAIServingCompletion
from vllm.entrypoints.openai.serving_embedding import OpenAIServingEmbedding
from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest, CompletionRequest, EmbeddingRequest
)
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

# --- Step 5: Set up OpenAI-compatible routes (no OpenAIAPIHandler) ---
print("Setting up OpenAI-compatible routes...")

# Present the friendly repo name to clients
SERVED_MODEL = MODEL_REPO

chat_server = OpenAIServingChat(
    engine, served_model=SERVED_MODEL, response_role="assistant", chat_template=None
)
completion_server = OpenAIServingCompletion(engine, served_model=SERVED_MODEL)
embedding_server = OpenAIServingEmbedding(engine, served_model=SERVED_MODEL)

async def handler(job):
    event = job.get("input", {}) if isinstance(job, dict) else {}
    path = event.get("path", "")
    body = event.get("body", {}) or {}

    # sensible defaults
    body.setdefault("model", SERVED_MODEL)
    body.setdefault("stream", False)  # RunPod expects a single JSON response

    try:
        if path.endswith("/v1/chat/completions"):
            req = ChatCompletionRequest(**body)
            resp = await chat_server.create_chat_completion(req, raw_request=None)
            return resp.model_dump()
        elif path.endswith("/v1/completions"):
            req = CompletionRequest(**body)
            resp = await completion_server.create_completion(req, raw_request=None)
            return resp.model_dump()
        elif path.endswith("/v1/embeddings"):
            req = EmbeddingRequest(**body)
            resp = await embedding_server.create_embedding(req, raw_request=None)
            return resp.model_dump()
        else:
            return {"error": f"Unknown path: {path}. Expected one of /v1/chat/completions, /v1/completions, /v1/embeddings"}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}

print("Starting RunPod serverless worker...")
runpod.serverless.start({
    "handler": handler,
    "concurrency_modifier": lambda _: 128,
})