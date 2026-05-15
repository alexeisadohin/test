from minds14_intent_service.config import Settings
from minds14_intent_service.text_model import load_artifact, predict_intent


def main() -> None:
    """Run a single local prediction with the saved model artifact."""
    settings = Settings()
    model, _ = load_artifact(settings.intent_model_path)
    prediction = predict_intent(model, "я хочу заблокировать карту")
    print({"label": prediction.label, "score": prediction.score})


if __name__ == "__main__":
    main()
