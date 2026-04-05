from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import sys
import os

# Relative path importing
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)

from api.schemas import GenerateRequest, GenerateResponse
from inference.generate import LLMPipeline

app = FastAPI(title="Custom LLM API")

# Setup CORS for local UI testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline = None

@app.on_event("startup")
async def startup_event():
    global pipeline
    print("Initializing Model Pipeline within API...")
    try:
        pipeline = LLMPipeline()
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load model on startup: {e}")
        # Allows API to stay alive for healthchecks even without a trained model

@app.post("/generate", response_model=GenerateResponse)
async def generate_text(request: GenerateRequest):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model pipeline is currently unavailable.")
        
    try:
        response_text = pipeline.generate(
            prompt=request.prompt,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p
        )
        return GenerateResponse(generated_text=response_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "online", "model_ready": pipeline is not None}
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
