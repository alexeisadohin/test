from __future__ import annotations

import tempfile
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Dict, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from loguru import logger
from sklearn.pipeline import Pipeline

from minds14_intent_service.asr import AsrTranscriber
from minds14_intent_service.config import Settings
from minds14_intent_service.schemas import (
    AudioPredictionResponse,
    HealthResponse,
    TextPredictionResponse,
    TextRequest,
)
from minds14_intent_service.text_model import load_artifact, predict_intent
from minds14_intent_service.training import train_and_save


class AppState:
    """Hold loaded model objects for the running FastAPI process."""

    intent_model: Optional[Pipeline] = None
    intent_metadata: Dict[str, Any] = {}
    asr: Optional[AsrTranscriber] = None


state = AppState()


def configure_logging(log_file: Path) -> None:
    """Configure loguru sinks for console and file logging."""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add(log_file, rotation="10 MB", retention=5, level="INFO", enqueue=True)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Initialize models once at FastAPI startup and release on shutdown."""
    settings = Settings()
    configure_logging(settings.log_file)
    logger.info("Starting service")

    model_path = settings.intent_model_path
    if not model_path.exists():
        if not settings.auto_train:
            raise RuntimeError(
                f"Intent model not found at {model_path}. "
                "Run scripts/train_intent_model.py or set AUTO_TRAIN=true."
            )
        logger.info("Intent model not found; training baseline model")
        train_and_save(
            output_path=model_path,
            dataset_name=settings.dataset_name,
            language_config=settings.language_config,
            test_size=settings.test_size,
            random_state=settings.random_state,
        )

    state.intent_model, state.intent_metadata = load_artifact(model_path)
    logger.info("Loaded intent model from {}", model_path)

    state.asr = AsrTranscriber(
        settings.asr_model_name,
        settings.asr_device,
        language=settings.asr_language,
        task=settings.asr_task,
    )
    logger.info(
        "Loaded ASR model {} with language={} task={}",
        settings.asr_model_name,
        settings.asr_language,
        settings.asr_task,
    )

    yield

    logger.info("Stopping service")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="MINDS-14 Intent Service",
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        """Return service and model loading status."""
        return HealthResponse(
            status="ok",
            intent_model_loaded=state.intent_model is not None,
            asr_model_loaded=state.asr is not None,
        )

    @app.post("/predict/text", response_model=TextPredictionResponse)
    def predict_text(payload: TextRequest) -> TextPredictionResponse:
        """Predict intent class from a text request."""
        if state.intent_model is None:
            raise HTTPException(status_code=503, detail="Intent model is not loaded.")
        try:
            prediction = predict_intent(state.intent_model, payload.text)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        logger.info("Text prediction label={} score={:.4f}", prediction.label, prediction.score)
        return TextPredictionResponse(label=prediction.label, score=prediction.score)

    @app.post("/predict/audio", response_model=AudioPredictionResponse)
    async def predict_audio(file: UploadFile = File(...)) -> AudioPredictionResponse:
        """Transcribe an uploaded audio file and predict its intent class."""
        if state.intent_model is None:
            raise HTTPException(status_code=503, detail="Intent model is not loaded.")
        if state.asr is None:
            raise HTTPException(status_code=503, detail="ASR model is not loaded.")

        suffix = Path(file.filename or "audio.wav").suffix or ".wav"
        content = await file.read()
        if not content:
            raise HTTPException(status_code=422, detail="Audio file must not be empty.")

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(content)
            tmp_path = Path(tmp.name)

        try:
            transcript = state.asr.transcribe_path(str(tmp_path))
            if not transcript:
                raise HTTPException(status_code=422, detail="ASR returned an empty transcript.")
            prediction = predict_intent(state.intent_model, transcript)
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("Audio prediction failed")
            raise HTTPException(status_code=500, detail="Audio prediction failed.") from exc
        finally:
            tmp_path.unlink(missing_ok=True)

        logger.info("Audio prediction label={} score={:.4f}", prediction.label, prediction.score)
        return AudioPredictionResponse(
            label=prediction.label,
            score=prediction.score,
            transcript=transcript,
        )

    return app
