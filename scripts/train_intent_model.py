from pathlib import Path

from loguru import logger

from minds14_intent_service.config import Settings
from minds14_intent_service.training import train_and_save


def main() -> None:
    """Train and save the baseline intent classifier."""
    settings = Settings()
    metadata = train_and_save(
        output_path=settings.intent_model_path,
        dataset_name=settings.dataset_name,
        language_config=settings.language_config,
        test_size=settings.test_size,
        random_state=settings.random_state,
    )

    logger.info("Saved model to {}", Path(settings.intent_model_path).resolve())
    logger.info("Metrics are computed from the local holdout split: {}", metadata["metrics"])


if __name__ == "__main__":
    main()
