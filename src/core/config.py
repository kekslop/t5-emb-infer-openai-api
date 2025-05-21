import os
from pathlib import Path
from dotenv import load_dotenv

# Загрузка переменных окружения из .env файла
dotenv_path = Path('.env')
if dotenv_path.exists():
    load_dotenv(dotenv_path=dotenv_path)
else:
    # Попробуем загрузить из env в корне проекта
    alt_dotenv_path = Path('env')
    if alt_dotenv_path.exists():
        load_dotenv(dotenv_path=alt_dotenv_path)
    else:
        print("Warning: Neither .env nor env file found, using default values or system environment variables")

# Constants - using environment variables
LOCAL_MODEL_PATH = os.environ.get("MODEL_PATH", "./FRIDA")  # Local path to your existing model
MODEL_NAME = os.environ.get("MODEL_NAME", "ai-forever/FRIDA")  # Model name can be customized via env
API_TOKEN = os.environ.get("API_TOKEN", "default_token_change_me")
MAX_QUEUE_SIZE = int(os.environ.get("MAX_QUEUE_SIZE", 10))

# Для отладки
print(f"Loaded configuration:")
print(f"MODEL_PATH: {LOCAL_MODEL_PATH}")
print(f"MODEL_NAME: {MODEL_NAME}")
print(f"API_TOKEN: {'***' + API_TOKEN[-4:] if len(API_TOKEN) > 4 else 'Not set properly'}")
print(f"MAX_QUEUE_SIZE: {MAX_QUEUE_SIZE}") 