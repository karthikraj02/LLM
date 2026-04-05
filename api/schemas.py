from pydantic import BaseModel, Field

class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="The input text to generate from")
    max_tokens: int = Field(50, description="Maximum number of tokens to generate")
    temperature: float = Field(0.8, description="Sampling temperature")
    top_k: int = Field(40, description="Top-k sampling parameter")
    top_p: float = Field(0.9, description="Nucleus sampling threshold")

class GenerateResponse(BaseModel):
    generated_text: str
