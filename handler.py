
# handler_patched.py — vLLM (>=0.9) OpenAI-compatible RunPod worker
# - Uses modern OpenAI-serving API (ModelConfig + OpenAIServingModels)
# - Supports local HF snapshots and optional LoRA/adapter repos
# - Routes /v1/chat/completions, /v1/completions, /v1/embeddings

import os
import json
import runpod
from typing import Any, Dict, Optional

# HF auth + snapshot
try:
    from huggingface_hub import login as hf_login, snapshot_download
except Exception:
    hf_login = None
    snapshot_download = None

# vLLM imports (modern paths)
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.openai.serving_engine import ModelConfig
from vllm.entrypoints.openai.serving_models import (
    OpenAIServingModels,
    BaseModelPath
)
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.serving_completion import OpenAIServingCompletion
from vllm.entrypoints.openai.serving_embedding import OpenAIServingEmbedding
from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    CompletionRequest,
    EmbeddingRequest,
)

print("=== RunPod vLLM worker boot ===")

# -----------------------------
# 0) Env & defaults
# -----------------------------
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
MODEL_REPO = os.environ.get("MODEL") or os.environ.get("MODEL_REPO") or os.environ.get("MODEL_ID") or "Qwen/Qwen2.5-7B-Instruct"
BASE_MODEL_NAME = os.environ.get("BASE_MODEL_NAME")  # required if MODEL_REPO is a LoRA/adapter
CACHE_DIR = os.environ.get("CACHE_DIR", "/runpod-volume/models")
SERVED_NAME = os.environ.get("SERVED_MODEL_NAME", MODEL_REPO)  # what clients use as `model`

DTYPE = os.environ.get("DTYPE", "auto")
MAX_MODEL_LEN = int(os.environ.get("MAX_MODEL_LEN", "4096"))
TENSOR_PARALLEL = int(os.environ.get("TENSOR_PARALLEL_SIZE", "1"))
GPU_UTIL = float(os.environ.get("GPU_MEMORY_UTILIZATION", "0.95"))
TRUST_REMOTE_CODE = os.environ.get("TRUST_REMOTE_CODE", "1").lower() in ("1","true","yes")
ENFORCE_EAGER = os.environ.get("ENFORCE_EAGER", "0").lower() in ("1","true","yes")

# Allow specifying an explicit tokenizer (useful when MODEL_REPO is an adapter snapshot)
TOKENIZER_NAME = os.environ.get("TOKENIZER_NAME")  # fallback later to model/base

# -----------------------------
# 1) HF login (optional) & snapshot
# -----------------------------
local_model_path: Optional[str] = None
if HF_TOKEN and hf_login is not None:
    try:
        print("Logging into Hugging Face Hub...")
        hf_login(HF_TOKEN)
        print("Hugging Face login successful.")
    except Exception as e:
        print(f"HF login warning: {e}")

if snapshot_download is not None:
    try:
        print(f"Preparing snapshot for repo: {MODEL_REPO}")
        # Keep a flat path under CACHE_DIR for cleanliness
        local_model_path = os.path.join(CACHE_DIR, MODEL_REPO.replace("/", "--"))
        if not os.path.exists(os.path.join(local_model_path, "config.json")):
            os.makedirs(local_model_path, exist_ok=True)
            snapshot_download(
                repo_id=MODEL_REPO,
                local_dir=local_model_path,
                local_dir_use_symlinks=False,
                token=HF_TOKEN,
                allow_patterns=None,
                ignore_patterns=None,
                resume_download=True,
            )
            print(f"Snapshot ready at: {local_model_path}")
        else:
            print(f"Model already present at: {local_model_path}")
    except Exception as e:
        print(f"Snapshot warning: {e}. Will let vLLM fetch remotely.")
        local_model_path = None
else:
    print("huggingface_hub not available; will let vLLM fetch the model directly.")

# -----------------------------
# 2) Detect if MODEL_REPO looks like a LoRA/adapter folder (local) 
# -----------------------------
def _looks_like_lora(path: str) -> bool:
    if not path:
        return False
    return any(os.path.exists(os.path.join(path, fname)) for fname in ("adapter_config.json","adapter_model.safetensors","lora.safetensors"))

is_lora_local = _looks_like_lora(local_model_path or "")

# -----------------------------
# 3) Build engine args
# -----------------------------
# Choose model path: prefer local snapshot if available; otherwise use repo id
model_arg = local_model_path if local_model_path else MODEL_REPO

# Decide tokenizer default
if TOKENIZER_NAME:
    tokenizer_arg = TOKENIZER_NAME
else:
    # If adapter, prefer base model tokenizer; else use same as model
    tokenizer_arg = BASE_MODEL_NAME if (is_lora_local and BASE_MODEL_NAME) else model_arg

# LoRA preloading setup
enable_lora = False
lora_modules = None
if is_lora_local:
    if not BASE_MODEL_NAME:
        print("WARNING: Detected adapter files but BASE_MODEL_NAME is not set. "
            "Set BASE_MODEL_NAME to the base HF repo (e.g., Qwen/Qwen2.5-32B-Instruct).")
    else:
        enable_lora = True
        # OpenAI-serving also supports runtime loading, but preloading is simplest here
        lora_modules = [f"adapter={local_model_path}"]

print(f"""Configuring vLLM engine...
model: {model_arg}
tokenizer: {tokenizer_arg}
TP: {TENSOR_PARALLEL}
util: {GPU_UTIL}""")

engine_args = AsyncEngineArgs(
    model=model_arg,
    tokenizer=tokenizer_arg,
    dtype=DTYPE,
    max_model_len=MAX_MODEL_LEN,
    tensor_parallel_size=TENSOR_PARALLEL,
    gpu_memory_utilization=GPU_UTIL,
    trust_remote_code=TRUST_REMOTE_CODE,
    enforce_eager=ENFORCE_EAGER,
    enable_lora=enable_lora,
    lora_modules=lora_modules,
    download_dir=CACHE_DIR,
)

engine = AsyncLLMEngine.from_engine_args(engine_args)

# -----------------------------
# 4) OpenAI-serving objects (modern API)
# -----------------------------
models = OpenAIServingModels(
    engine_client=engine,
    base_model_paths=[BaseModelPath(name=SERVED_NAME)],
    # You can also add LoRA or prompt adapters here if you want runtime routes
    # lora_module_paths=[LoRAModulePath(name="adapter", path=local_model_path)] if is_lora_local else None,
    # prompt_adapter_paths=[PromptAdapterPath(name="...", path="...")],
)

model_config = ModelConfig(
    response_role="assistant",
    chat_template=None,
)

chat_server = OpenAIServingChat(engine, model_config, models)
completion_server = OpenAIServingCompletion(engine, model_config, models)
embedding_server = OpenAIServingEmbedding(engine, model_config, models)

# -----------------------------
# 5) RunPod job handler
# -----------------------------
async def handler(job):
    try:
        event = job.get("input", {}) if isinstance(job, dict) else {}
        path = event.get("path", "")
        body = event.get("body", {}) or {}

        # Defaults
        body = dict(body)  # shallow copy
        body.setdefault("model", SERVED_NAME)
        # RunPod likes a single JSON response; we keep stream False by default
        if path.endswith("/v1/chat/completions") or path.endswith("/v1/completions"):
            body.setdefault("stream", False)

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
        elif path.endswith("/v1/models"):
            # Optional: expose models list (handled by OpenAIServingModels)
            # The ServingModels object provides a starlette route typically;
            # here we return a minimal static response matching OpenAI format.
            return {"object": "list", "data": [{"id": SERVED_NAME, "object": "model"}]}
        else:
            return {"error": f"Unknown path: {path}. Expected one of /v1/chat/completions, /v1/completions, /v1/embeddings, /v1/models"}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}

print("Starting RunPod serverless worker...")
runpod.serverless.start({
    "handler": handler,
    "concurrency_modifier": lambda _: 128,
})
