"""
LLM Inference API
-----------------
Tries to load the model via vLLM (AsyncLLMEngine) for high-performance
batched serving.  If vLLM is not installed or CUDA is unavailable, falls
back to a standard HuggingFace `transformers` pipeline with
`device_map="auto"` so the server still works on CPU-only machines.

Default model: HuggingFaceTB/SmolLM-135M
Set the MODEL_ID environment variable to use a different model, e.g.:
    MODEL_ID=TinyLlama/TinyLlama-1.1B-Chat-v1.0 python api/server.py
"""

import os
import sys
import asyncio
import uuid

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Relative path importing
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)

from api.schemas import GenerateRequest, GenerateResponse

# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------
MODEL_ID: str = os.environ.get("MODEL_ID", "HuggingFaceTB/SmolLM-135M")

# Will be set during startup
_backend: str = "none"          # "vllm" | "transformers" | "none"
_vllm_engine = None             # vllm.AsyncLLMEngine instance
_hf_pipeline = None             # transformers.pipeline instance

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="LLM Inference API",
    description="High-performance LLM serving with vLLM or HuggingFace Transformers fallback.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Startup: try vLLM first, fall back to transformers
# ---------------------------------------------------------------------------
@app.on_event("startup")
async def startup_event():
    global _backend, _vllm_engine, _hf_pipeline

    print(f"[API] Loading model: {MODEL_ID}")

    # ---- Attempt 1: vLLM (requires CUDA) ----------------------------------
    if torch.cuda.is_available():
        try:
            from vllm import AsyncLLMEngine, AsyncEngineArgs  # type: ignore
            engine_args = AsyncEngineArgs(
                model=MODEL_ID,
                dtype="auto",
                trust_remote_code=True,
            )
            _vllm_engine = AsyncLLMEngine.from_engine_args(engine_args)
            _backend = "vllm"
            print(f"[API] Backend: vLLM  (model={MODEL_ID})")
            return
        except Exception as exc:
            print(f"[API] vLLM unavailable ({exc}), falling back to transformers.")

    # ---- Attempt 2: HuggingFace transformers pipeline ---------------------
    try:
        from transformers import pipeline as hf_pipeline_fn, AutoTokenizer  # type: ignore

        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        _hf_pipeline = hf_pipeline_fn(
            "text-generation",
            model=MODEL_ID,
            tokenizer=tokenizer,
            device_map="auto",
            trust_remote_code=True,
        )
        _backend = "transformers"
        print(f"[API] Backend: HuggingFace Transformers  (model={MODEL_ID})")
    except Exception as exc:
        print(f"[API] Failed to load model: {exc}")
        _backend = "none"


# ---------------------------------------------------------------------------
# /generate  – POST
# ---------------------------------------------------------------------------
@app.post("/generate", response_model=GenerateResponse)
async def generate_text(request: GenerateRequest):
    if _backend == "none":
        raise HTTPException(
            status_code=503,
            detail="Model pipeline is currently unavailable. Check server logs.",
        )

    prompt = request.prompt

    # ---- vLLM path --------------------------------------------------------
    if _backend == "vllm":
        try:
            from vllm import SamplingParams  # type: ignore

            sampling_params = SamplingParams(
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_k=request.top_k,
                top_p=request.top_p,
            )
            request_id = str(uuid.uuid4())
            results_generator = _vllm_engine.generate(prompt, sampling_params, request_id)

            final_output = None
            async for request_output in results_generator:
                final_output = request_output

            if final_output is None or not final_output.outputs:
                raise HTTPException(status_code=500, detail="vLLM returned no output.")
            generated_text = final_output.outputs[0].text
            return GenerateResponse(generated_text=generated_text)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    # ---- transformers path ------------------------------------------------
    if _backend == "transformers":
        try:
            # Run the blocking pipeline in a thread so we don't block the event loop
            loop = asyncio.get_event_loop()
            outputs = await loop.run_in_executor(
                None,
                lambda: _hf_pipeline(
                    prompt,
                    max_new_tokens=request.max_tokens,
                    temperature=request.temperature,
                    top_k=request.top_k,
                    top_p=request.top_p,
                    do_sample=True,
                    return_full_text=False,
                    pad_token_id=_hf_pipeline.tokenizer.eos_token_id,
                ),
            )
            generated_text = outputs[0]["generated_text"].strip()
            return GenerateResponse(generated_text=generated_text)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    raise HTTPException(status_code=500, detail="Unexpected backend state.")


# ---------------------------------------------------------------------------
# /health  – GET
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    return {
        "status": "online",
        "model_ready": _backend != "none",
        "backend": _backend,
        "model_id": MODEL_ID,
    }


# ---------------------------------------------------------------------------
# Direct execution
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
