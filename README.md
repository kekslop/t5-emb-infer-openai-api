# API Векторной Модели

Контейнеризированное FastAPI приложение для запуска моделей векторных представлений текста.

## Функциональность

- Аутентификация на основе токенов
- Совместимость с API OpenAI embeddings
- Очередь запросов для обработки параллельных запросов
- Поддержка GPU-ускорения
- Настройка Docker и docker-compose для простого развертывания

## Требования

- Docker
- Docker Compose
- Файлы модели (не включены в этот репозиторий)

## Настройка

1. Клонируйте этот репозиторий
2. Поместите файлы модели в директорию `./FRIDA`
3. Создайте файл `.env` на основе шаблона `env.sample`:
   ```
   cp env.sample .env
   ```
4. Отредактируйте файл `.env`, чтобы установить токен API и другие настройки
5. Соберите и запустите контейнер:
   ```
   docker-compose up -d
   ```

## Переменные окружения

- `API_TOKEN`: Токен аутентификации для доступа к API
- `MODEL_PATH`: Путь к файлам модели (по умолчанию: ./FRIDA)
- `MODEL_NAME`: Имя модели, публикуемое в API (по умолчанию: ai-forever/FRIDA)
- `MAX_QUEUE_SIZE`: Максимальный размер очереди запросов (по умолчанию: 10)

## Использование API

### Аутентификация

Все эндпоинты требуют Bearer токен:

```
Authorization: Bearer your_secret_token_here
```

### Эндпоинты

- `/v1/embedding`: Основной эндпоинт для получения векторных представлений (совместимый с OpenAI API)
- `/v1/embeddings`: Эндпоинт, совместимый с OpenAI embeddings API
- `/v1/models`: Получение информации о доступных моделях
- `/health`: Проверка состояния сервиса

### Пример запроса

```python
import requests

# Пустой список как значение по умолчанию для input
headers = {
    "Authorization": "Bearer your_secret_token_here",
    "Content-Type": "application/json"
}

response = requests.post(
    "http://localhost:8000/v1/embedding",
    headers=headers,
    json={
        "model": "ai-forever/FRIDA",
        "input": ["Текст для векторизации"]
    }
)

print(response.json())
```

## Параллельные запросы

Система реализует очередь запросов для обработки нескольких параллельных запросов без перегрузки GPU. Когда очередь заполнена, новые запросы получат ответ 503 Service Unavailable.

## Команды Docker

- Сборка и запуск: `docker-compose up -d`
- Просмотр логов: `docker-compose logs -f`
- Остановка: `docker-compose down`

## Загрузка моделей с помощью Hugging Face CLI

Для загрузки моделей с Hugging Face Hub можно использовать Hugging Face CLI. Это удобный способ загрузить необходимые файлы модели в директорию `./FRIDA`.

### Установка Hugging Face CLI

```bash
pip install -U "huggingface_hub[cli]"
```

### Аутентификация (опционально)

Для доступа к приватным моделям необходимо авторизоваться:

```bash
huggingface-cli login
```

### Примеры загрузки моделей

#### Загрузка всей модели

```bash
huggingface-cli download ai-forever/FRIDA --local-dir ./FRIDA
```

#### Загрузка отдельных файлов модели

```bash
huggingface-cli download ai-forever/FRIDA model.safetensors --local-dir ./FRIDA
huggingface-cli download ai-forever/FRIDA config.json --local-dir ./FRIDA
```

#### Загрузка определенной версии модели

```bash
huggingface-cli download ai-forever/FRIDA --revision v1.0 --local-dir ./FRIDA
```

#### Загрузка файлов по шаблону

```bash
huggingface-cli download ai-forever/FRIDA --include "*.safetensors" --exclude "*.fp16.*" --local-dir ./FRIDA
```

---

# Embedding Model API

A containerized FastAPI application for running embedding models.

## Features

- Token-based authentication
- Compatible with OpenAI embeddings API
- Request queueing for handling concurrent requests
- GPU acceleration support
- Docker and docker-compose setup for easy deployment

## Requirements

- Docker
- Docker Compose
- Model files (not included in this repository)

## Setup

1. Clone this repository
2. Place your model files in the `./FRIDA` directory
3. Create a `.env` file using the `env.sample` as a template:
   ```
   cp env.sample .env
   ```
4. Edit the `.env` file to set your API token and other settings
5. Build and start the container:
   ```
   docker-compose up -d
   ```

## Environment Variables

- `API_TOKEN`: Authentication token for accessing the API
- `MODEL_PATH`: Path to the model files (default: ./FRIDA)
- `MODEL_NAME`: Name of the model published in the API (default: ai-forever/FRIDA)
- `MAX_QUEUE_SIZE`: Maximum number of requests in the queue (default: 10)

## API Usage

### Authentication

All endpoints require a Bearer token:

```
Authorization: Bearer your_secret_token_here
```

### Endpoints

- `/v1/embedding`: Main embedding endpoint (compatible with OpenAI's API)
- `/v1/embeddings`: OpenAI-compatible embeddings endpoint
- `/v1/models`: Get information about available models
- `/health`: Health check endpoint

### Example Request

```python
import requests

# Empty list as default input
headers = {
    "Authorization": "Bearer your_secret_token_here",
    "Content-Type": "application/json"
}

response = requests.post(
    "http://localhost:8000/v1/embedding",
    headers=headers,
    json={
        "model": "ai-forever/FRIDA",
        "input": ["Text to embed"]
    }
)

print(response.json())
```

## Concurrent Requests

The system implements a request queue to handle multiple concurrent requests without overwhelming the GPU. When the queue is full, new requests will receive a 503 Service Unavailable response.

## Docker Commands

- Build and start: `docker-compose up -d`
- View logs: `docker-compose logs -f`
- Stop: `docker-compose down`

## Downloading Models with Hugging Face CLI

You can use the Hugging Face CLI to download models from the Hugging Face Hub. This is a convenient way to download the required model files to the `./FRIDA` directory.

### Installing Hugging Face CLI

```bash
pip install -U "huggingface_hub[cli]"
```

### Authentication (optional)

For access to private models, you need to authenticate:

```bash
huggingface-cli login
```

### Examples of Downloading Models

#### Download the Entire Model

```bash
huggingface-cli download ai-forever/FRIDA --local-dir ./FRIDA
```

#### Download Individual Model Files

```bash
huggingface-cli download ai-forever/FRIDA model.safetensors --local-dir ./FRIDA
huggingface-cli download ai-forever/FRIDA config.json --local-dir ./FRIDA
```

#### Download a Specific Model Version

```bash
huggingface-cli download ai-forever/FRIDA --revision v1.0 --local-dir ./FRIDA
```

#### Download Files Using Patterns

```bash
huggingface-cli download ai-forever/FRIDA --include "*.safetensors" --exclude "*.fp16.*" --local-dir ./FRIDA
```