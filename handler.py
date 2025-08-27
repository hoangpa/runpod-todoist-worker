# handler.py — vLLM OpenAI-compatible RunPod worker (v0.7.x)
# - Supports adapter (LoRA) or merged model
# - Detects adapter at repo root OR in a nested subfolder (e.g., epoch_4/)
# - Unifies HF cache paths to avoid duplicate downloads
# - Graceful fallback if LoRA kwargs are unsupported
# - Exposes /v1/chat/completions, /v1/completions, /v1/embeddings, /v1/models (update - #2)

import os
import runpod
from typing import Optional
import glob
# Optional: huggingface_hub for login + snapshot

def _has_weights(path: Optional[str]) -> bool:
    if not path or not os.path.isdir(path):
        return False
    # explicit index files
    for idx in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        if os.path.exists(os.path.join(path, idx)):
            return True
    # or any shard files
    if glob.glob(os.path.join(path, "*.safetensors")) or glob.glob(os.path.join(path, "*.bin")):
        return True
    return False

try:
    from huggingface_hub import login as hf_login, snapshot_download
except Exception:
    hf_login = None
    snapshot_download = None

# vLLM imports (OpenAI-serving entrypoints) — compatible with vLLM 0.7.x
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.openai.serving_engine import ModelConfig
from vllm.entrypoints.openai.serving_models import OpenAIServingModels, BaseModelPath
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.serving_completion import OpenAIServingCompletion
from vllm.entrypoints.openai.serving_embedding import OpenAIServingEmbedding
from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    CompletionRequest,
    EmbeddingRequest,
)

print("=== RunPod vLLM worker boot (adapter-aware, nested) ===")

# -----------------------------
# 0) Volume + env
# -----------------------------
VOLUME_ROOT = os.environ.get("VOLUME_ROOT")
if not VOLUME_ROOT:
    VOLUME_ROOT = "/runpod-volume" if os.path.ismount("/runpod-volume") else "/workspace"

# Core env (accept either MODEL_REPO or MODEL_NAME just in case)
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
MODEL_REPO = (
    os.environ.get("MODEL_REPO")
    or os.environ.get("MODEL")
    or os.environ.get("MODEL_ID")
    or os.environ.get("MODEL_NAME")
    or "Qwen/Qwen2.5-32B-Instruct"
)
BASE_MODEL_NAME = os.environ.get("BASE_MODEL_NAME")   # required if MODEL_REPO is an adapter
SERVED_NAME = os.environ.get("SERVED_MODEL_NAME", MODEL_REPO)

# Paths (avoid duplicates)
CACHE_DIR = os.environ.get("CACHE_DIR", os.path.join(VOLUME_ROOT, "models"))      # snapshots / small aux files
HF_HOME   = os.environ.get("HF_HOME",   os.path.join(VOLUME_ROOT, "model_cache")) # HF cache where base weights live
os.environ.setdefault("HF_HUB_CACHE", os.path.join(CACHE_DIR, ".hf_cache"))
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(HF_HOME, exist_ok=True)
os.environ.setdefault("HF_HOME", HF_HOME)
os.environ.setdefault("TRANSFORMERS_CACHE", HF_HOME)

# Engine knobs
DTYPE = os.environ.get("DTYPE", "auto")
MAX_MODEL_LEN = int(os.environ.get("MAX_MODEL_LEN", "4096"))
TENSOR_PARALLEL = int(os.environ.get("TENSOR_PARALLEL_SIZE", "1"))
GPU_UTIL = float(os.environ.get("GPU_MEMORY_UTILIZATION", "0.95"))
TRUST_REMOTE_CODE = os.environ.get("TRUST_REMOTE_CODE", "1").lower() in ("1", "true", "yes")
ENFORCE_EAGER = os.environ.get("ENFORCE_EAGER", "0").lower() in ("1", "true", "yes")

# Tokenizer override (useful for adapters)
TOKENIZER_NAME = os.environ.get("TOKENIZER_NAME")  # fallback set below

# Optional: let user hint a subfolder (e.g., "epoch_4")
ADAPTER_SUBDIR = os.environ.get("ADAPTER_SUBDIR")

# -----------------------------
# 1) HF login (optional) + snapshot MODEL_REPO
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
        local_model_path = os.path.join(CACHE_DIR, MODEL_REPO.replace("/", "--"))
        if not _has_weights(local_model_path):
            print(f"Snapshot missing weights. Hydrating {MODEL_REPO} -> {local_model_path}")
            os.makedirs(local_model_path, exist_ok=True)
            snapshot_download(
                repo_id=MODEL_REPO,
                local_dir=local_model_path,
                local_dir_use_symlinks=False,
                token=HF_TOKEN,
                resume_download=True,
            )
        else:
            print(f"Snapshot with weights present: {local_model_path}")
    except Exception as e:
        print(f"Snapshot warning: {e}. Will let vLLM fetch {MODEL_REPO} via HF cache.")
        local_model_path = None
else:
    print("huggingface_hub not available; will let vLLM fetch directly.")

# -----------------------------
# 2) Adapter detection (root or nested)
# -----------------------------
def _find_lora_root(path: Optional[str]) -> Optional[str]:
    """Return the directory that contains adapter_config.json and adapter weights.
       Works for repo root or nested layouts (e.g., epoch_4/)."""
    if not path:
        return None

    wanted = {"adapter_config.json", "adapter_model.safetensors", "lora.safetensors"}

    # User hint takes priority
    if ADAPTER_SUBDIR:
        candidate = os.path.join(path, ADAPTER_SUBDIR)
        if any(os.path.exists(os.path.join(candidate, f)) for f in wanted):
            print(f"Adapter found via ADAPTER_SUBDIR at: {candidate}")
            return candidate

    # Quick check at root
    if any(os.path.exists(os.path.join(path, f)) for f in wanted):
        print(f"Adapter found at repo root: {path}")
        return path

    # Fallback: walk subdirs to find a matching folder
    for root, _, files in os.walk(path):
        files_set = set(files)
        if ("adapter_config.json" in files_set) and (
            "adapter_model.safetensors" in files_set or "lora.safetensors" in files_set
        ):
            print(f"Adapter found nested at: {root}")
            return root

    return None

adapter_root = _find_lora_root(local_model_path)
is_lora_local = adapter_root is not None

# -----------------------------
# 3) Build engine args (correct base vs adapter wiring)
# -----------------------------
if is_lora_local:
    if not BASE_MODEL_NAME:
        raise RuntimeError(
            "Detected adapter files but BASE_MODEL_NAME is not set "
            "(e.g., Qwen/Qwen2.5-32B-Instruct)."
        )
    model_arg = BASE_MODEL_NAME
    tokenizer_arg = TOKENIZER_NAME or BASE_MODEL_NAME
    enable_lora = True
    lora_modules = [f"adapter={adapter_root}"]
    print(f"Serving BASE: {model_arg} with LoRA adapter at: {adapter_root}")
else:
    # Optional hardening: if BASE_MODEL_NAME is set, try serving base+adapter by repo id
    if BASE_MODEL_NAME:
        print("No local adapter files found; enabling LoRA via HF repo id.")
        model_arg = BASE_MODEL_NAME
        tokenizer_arg = TOKENIZER_NAME or BASE_MODEL_NAME
        enable_lora = True
        lora_modules = [f"adapter={MODEL_REPO}"]  # vLLM will resolve HF repo id
    else:
        # Non-adapter: serve MODEL_REPO itself (prefer local snapshot if present)
        model_arg = local_model_path or MODEL_REPO
        tokenizer_arg = TOKENIZER_NAME or model_arg
        enable_lora = False
        lora_modules = None
        print(f"Serving standalone model: {model_arg}")

print(
    "Configuring vLLM engine...\n"
    f"  model: {model_arg}\n"
    f"  tokenizer: {tokenizer_arg}\n"
    f"  TP: {TENSOR_PARALLEL}\n"
    f"  util: {GPU_UTIL}"
)

# Some vLLM builds don't support LoRA kwargs; try with, then fall back cleanly.
try:
    engine_args = AsyncEngineArgs(
        model=model_arg,
        tokenizer=tokenizer_arg,
        dtype=DTYPE,
        max_model_len=MAX_MODEL_LEN,
        tensor_parallel_size=TENSOR_PARALLEL,
        gpu_memory_utilization=GPU_UTIL,
        trust_remote_code=TRUST_REMOTE_CODE,
        enforce_eager=ENFORCE_EAGER,
        download_dir=CACHE_DIR,
        **({"enable_lora": True, "lora_modules": lora_modules} if enable_lora else {}),
    )
except TypeError as e:
    print(f"LoRA args unsupported on this vLLM: {e}. Falling back without LoRA kwargs.")
    engine_args = AsyncEngineArgs(
        model=model_arg,
        tokenizer=tokenizer_arg,
        dtype=DTYPE,
        max_model_len=MAX_MODEL_LEN,
        tensor_parallel_size=TENSOR_PARALLEL,
        gpu_memory_utilization=GPU_UTIL,
        trust_remote_code=TRUST_REMOTE_CODE,
        enforce_eager=ENFORCE_EAGER,
        download_dir=CACHE_DIR,
    )

engine = AsyncLLMEngine.from_engine_args(engine_args)

# -----------------------------
# 4) OpenAI-serving objects (modern API)
# -----------------------------
models = OpenAIServingModels(
    engine_client=engine,
    base_model_paths=[BaseModelPath(name=SERVED_NAME)],
)
model_config = ModelConfig(response_role="assistant", chat_template=None)
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

        body = dict(body)
        body.setdefault("model", SERVED_NAME)
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
            return {"object": "list", "data": [{"id": SERVED_NAME, "object": "model"}]}
        else:
            return {
                "error": (
                    f"Unknown path: {path}. Expected one of "
                    "/v1/chat/completions, /v1/completions, /v1/embeddings, /v1/models"
                )
            }
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}

print("Starting RunPod serverless worker...")
runpod.serverless.start({
    "handler": handler,
    "concurrency_modifier": lambda _: 128,
})