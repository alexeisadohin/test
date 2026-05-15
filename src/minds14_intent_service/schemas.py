from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Response schema for service readiness checks."""

    status: str
    intent_model_loaded: bool
    asr_model_loaded: bool


class TextRequest(BaseModel):
    """Request schema for text intent prediction."""

    text: str = Field(..., min_length=1)


class TextPredictionResponse(BaseModel):
    """Response schema for text intent prediction."""

    label: str
    score: float


class AudioPredictionResponse(TextPredictionResponse):
    """Response schema for audio intent prediction."""

    transcript: str
