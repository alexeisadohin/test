from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Read service configuration from defaults, env file, and environment."""

    dataset_name: str = "PolyAI/minds14"
    language_config: str = "ru-RU"
    test_size: float = 0.2
    random_state: int = 42

    model_dir: Path = Path("models")
    intent_model_filename: str = "intent_classifier.joblib"
    auto_train: bool = False

    asr_model_name: str = "openai/whisper-small"
    asr_device: str = "auto"
    asr_language: str = "russian"
    asr_task: str = "transcribe"

    log_file: Path = Path("logs/service.log")

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    @property
    def intent_model_path(self) -> Path:
        """Возвращает полный путь к артефакту классификатора интентов."""
        return self.model_dir / self.intent_model_filename
