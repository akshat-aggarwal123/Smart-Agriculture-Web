import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./test.db")
    ML_MODEL_PATH: str = os.path.join(os.path.dirname(__file__), "ml-models")

settings = Settings()