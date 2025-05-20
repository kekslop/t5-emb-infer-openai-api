import os

# Constants - using environment variables
LOCAL_MODEL_PATH = os.environ.get("MODEL_PATH", "./FRIDA")  # Local path to your existing model
MODEL_NAME = os.environ.get("MODEL_NAME", "ai-forever/FRIDA")  # Model name can be customized via env
API_TOKEN = os.environ.get("API_TOKEN", "default_token_change_me")
MAX_QUEUE_SIZE = int(os.environ.get("MAX_QUEUE_SIZE", 10)) 