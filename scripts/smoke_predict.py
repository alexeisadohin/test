from minds14_intent_service.config import Settings
from minds14_intent_service.text_model import load_artifact, predict_intent


def main() -> None:
    """Выполняет один локальный прогноз с сохраненным артефактом модели."""
    settings = Settings()
    model, _ = load_artifact(settings.intent_model_path)
    prediction = predict_intent(model, "я хочу заблокировать карту")
    print({"label": prediction.label, "score": prediction.score})


if __name__ == "__main__":
    main()
