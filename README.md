# MINDS-14 Intent Service

Проект для тестового задания: классификация `intent_class` по текстовой транскрипции и по аудио через ASR + текстовый классификатор.

## Структура

```text
.
|-- notebooks/
|   |-- notebook.ipynb
|   `-- notebook_with_outputs.ipynb
|-- Dockerfile
|-- pyproject.toml
|-- requirements.txt
|-- README.md
|-- models/
|   `-- .gitkeep
|-- logs/
|   `-- .gitkeep
|-- scripts/
|   |-- train_intent_model.py
|   `-- smoke_predict.py
`-- src/
    `-- minds14_intent_service/
        |-- app.py
        |-- asr.py
        |-- config.py
        |-- data.py
        |-- schemas.py
        |-- text_model.py
        `-- training.py
```

## Что реализовано

- `notebooks/notebook.ipynb`: EDA, баланс классов, `train_test_split(..., stratify=...)`, текстовый baseline, `weighted F1`, `macro F1`, ASR pipeline, `WER`/`CER`, сравнение качества на оригинальных и ASR-транскриптах, LoRA.
- `scripts/train_intent_model.py`: обучение TF-IDF + Logistic Regression и сохранение артефакта в `models/intent_classifier.joblib`.
- `src/minds14_intent_service/app.py`: FastAPI-сервис с ручками `/health`, `/predict/text`, `/predict/audio`.
- Модели загружаются один раз при старте сервиса через FastAPI `lifespan`.
- Логирование сделано через `loguru`: в консоль контейнера и в файл `logs/service.log`.

## Схема итогового pipeline

```text
POST /predict/audio
        |
        v
  аудиофайл (.wav/.mp3/...)
        |
        v
  Whisper ASR
        |
        v
  текстовая транскрипция
        |
        v
  TF-IDF word/char признаки
        |
        v
  Logistic Regression
        |
        v
  intent_class + score
```

Для текстового endpoint схема короче:

```text
POST /predict/text -> TF-IDF word/char признаки -> Logistic Regression -> intent_class + score
```

## Данные и модели

- Датасет: `PolyAI/minds14`, конфигурация `ru-RU`.
- Текстовая модель: `TfidfVectorizer` по word/char n-граммам + `LogisticRegression`.
- ASR: `openai/whisper-small`.

`openai/whisper-small` выбран как multilingual ASR с поддержкой русского языка. Для коротких пользовательских запросов это разумный баланс качества и скорости на GPU уровня RTX 3090. Качество ASR оценивается через `WER` и `CER`; влияние ASR на бизнес-задачу оценивается сравнением intent-метрик на оригинальных транскрипциях и ASR-транскрипциях.

## Локальный запуск

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
python scripts/train_intent_model.py
uvicorn minds14_intent_service.app:create_app --factory --host 0.0.0.0 --port 8000
```

Если нужно обучать baseline автоматически при старте сервиса:

```bash
set AUTO_TRAIN=true
uvicorn minds14_intent_service.app:create_app --factory --host 0.0.0.0 --port 8000
```

## Docker

```bash
docker build -t test-task .
docker run -p 8000:8000 -e AUTO_TRAIN=true test-task
```

Если `models/intent_classifier.joblib` уже создан локально до сборки образа, можно запускать без `AUTO_TRAIN=true`.

## Endpoints

Health check:

```bash
curl http://localhost:8000/health
```

Классификация текста:

```bash
curl -X POST http://localhost:8000/predict/text ^
  -H "Content-Type: application/json" ^
  -d "{\"text\":\"я хочу заблокировать карту\"}"
```

Классификация аудио:

```bash
curl -X POST http://localhost:8000/predict/audio ^
  -F "file=@sample.wav"
```

Пример ответа `/predict/text`:

```json
{
  "label": "freeze",
  "score": 0.83
}
```

Пример ответа `/predict/audio`:

```json
{
  "label": "freeze",
  "score": 0.83,
  "transcript": "я хочу заблокировать карту"
}
```

## Конфигурация

Настройки задаются через переменные окружения:

- `DATASET_NAME`, по умолчанию `PolyAI/minds14`
- `LANGUAGE_CONFIG`, по умолчанию `ru-RU`
- `TEST_SIZE`, по умолчанию `0.2`
- `MODEL_DIR`, по умолчанию `models`
- `INTENT_MODEL_FILENAME`, по умолчанию `intent_classifier.joblib`
- `AUTO_TRAIN`, по умолчанию `false`
- `ASR_MODEL_NAME`, по умолчанию `openai/whisper-small`
- `ASR_DEVICE`, по умолчанию `auto`
- `ASR_LANGUAGE`, по умолчанию `russian`
- `ASR_TASK`, по умолчанию `transcribe`
- `LOG_FILE`, по умолчанию `logs/service.log`

## Интерпретация качества

В ноутбуке считаются:

- `precision_weighted`
- `recall_weighted`
- `f1_weighted`
- `f1_macro`
- отчет по каждому классу
- `WER` и `CER` для ASR

Пороговая логика для вывода о применимости:

- `recall_weighted >= 0.70`
- `precision_weighted >= 0.75`
- `f1_weighted >= 0.80`

Если метрики ниже порогов, решение с текущими данными и выбранной моделью нельзя считать готовым к применению. Что улучшать: добавить размеченные данные, добрать примеры по слабым или редким intent-классам, почистить транскрипции, попробовать более сильный текстовый encoder или LoRA-дообучение, отдельно улучшать ASR по `WER`/`CER`.
