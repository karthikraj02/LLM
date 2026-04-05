from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import argparse
import uvicorn
import os

from pydantic import BaseModel

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 150
    temperature: float = 0.7
    top_k: int = 40
    top_p: float = 0.9

class GenerateResponse(BaseModel):
    generated_text: str

app = FastAPI(title="vLLM Optimized API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

llm_engine = None
sampling_params_class = None

@app.on_event("startup")
async def startup_event():
    global llm_engine, sampling_params_class
    print("Initializing vLLM Engine...")
    try:
        from vllm import LLM, SamplingParams
        sampling_params_class = SamplingParams
        
        # Load the base model and dynamically apply the LoRA adapters
        llm_engine = LLM(
            model="meta-llama/Meta-Llama-3-8B-Instruct",
            enable_lora=True,
            max_lora_rank=16,
            gpu_memory_utilization=0.9,
            quantization="awq" 
        )
        print("vLLM Engine loaded successfully.")
    except Exception as e:
        print(f"Failed to load vLLM model: {e}")

@app.post("/generate", response_model=GenerateResponse)
async def generate_text(request: GenerateRequest):
    if llm_engine is None:
        raise HTTPException(status_code=503, detail="vLLM engine is currently unavailable.")
        
    try:
        # Define the generation parameters
        sampling_params = sampling_params_class(
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            max_tokens=request.max_tokens
        )
        
        # Check if LoRA adapters exist
        lora_path = "./lora-adapters"
        lora_request = None
        if os.path.exists(lora_path):
            from vllm.lora.request import LoRARequest
            lora_request = LoRARequest("custom_lora", 1, lora_path)
            
        outputs = llm_engine.generate(
            [request.prompt],
            sampling_params=sampling_params,
            lora_request=lora_request
        )
        
        response_text = outputs[0].outputs[0].text
        return GenerateResponse(generated_text=response_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "online", "model_ready": llm_engine is not None}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", type=str, default="vllm")
    parser.add_argument("--quantization", type=str, default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    args = parser.parse_args()
    
    uvicorn.run(app, host="0.0.0.0", port=8000)