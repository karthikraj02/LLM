from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import sys
import os
import argparse

# Relative path importing
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)

from api.schemas import GenerateRequest, GenerateResponse
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
import uuid

app = FastAPI(title="High-Speed vLLM API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = None

@app.on_event("startup")
async def startup_event():
    global engine
    print("Initializing vLLM Async Engine for blazing fast generation...")
    try:
        # Use our LoRA fine-tuned model or the base model if LoRA isn't available
        model_path = "./lora-adapters" if os.path.exists("./lora-adapters") else "meta-llama/Meta-Llama-3-8B-Instruct"
        
        engine_args = AsyncEngineArgs(
            model=model_path,
            quantization="awq", # Ensure awq models are fetched if quantization used
            tensor_parallel_size=1, # Change if you have multiple GPUs for inference
            gpu_memory_utilization=0.90,
            max_model_len=4096,
            enforce_eager=False, 
        )
        engine = AsyncLLMEngine.from_engine_args(engine_args)
        print("vLLM Engine loaded successfully.")
    except Exception as e:
        print(f"Failed to load vLLM model on startup: {e}")

@app.post("/generate", response_model=GenerateResponse)
async def generate_text(request: GenerateRequest):
    if engine is None:
        raise HTTPException(status_code=503, detail="vLLM engine is currently unavailable.")
        
    request_id = str(uuid.uuid4())
    sampling_params = SamplingParams(
        temperature=request.temperature,
        top_k=request.top_k,
        top_p=request.top_p,
        max_tokens=request.max_tokens
    )
    
    try:
        # Async generation with vLLM
        results_generator = engine.generate(request.prompt, sampling_params, request_id)
        
        final_output = None
        async for request_output in results_generator:
            final_output = request_output;
            
        return GenerateResponse(generated_text=final_output.outputs[0].text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "online", "engine_ready": engine is not None}
    
if __name__ == "__main__":
    import uvicorn
    # Allow arguments to be passed from run.py
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", type=str, default="vllm")
    parser.add_argument("--quantization", type=str, default="awq")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    args = parser.parse_args()
    
    uvicorn.run(app, host="0.0.0.0", port=8000)