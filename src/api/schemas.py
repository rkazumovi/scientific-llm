"""
scientific-llm - Step 8a: FastAPI request/response models.

Kept in their own file (rather than inline in main.py) because FastAPI
uses these Pydantic models for two things at once - request validation
(a malformed request body is rejected with a 422 before your handler
code ever runs) and the auto-generated OpenAPI schema at /docs - and
both read better when the shapes are named and separate from the route
logic.
"""

from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="The user prompt to send to the model.")
    max_new_tokens: int = Field(200, ge=1, le=1024, description="Generation length cap.")


class GenerateResponse(BaseModel):
    prompt: str
    generated_text: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    adapter_loaded: bool
    gpu_available: bool
