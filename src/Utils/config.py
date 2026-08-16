from pathlib import Path
from typing import Any
from dotenv import load_dotenv

from pydantic_settings import BaseSettings, SettingsConfigDict

# Resolve the absolute path to the .env file at the project root
BASE_DIR = Path(__file__).resolve().parent.parent.parent
ENV_FILE_PATH = BASE_DIR / ".env"

# Load the environment variables into the OS environment so LangChain can see them
load_dotenv(ENV_FILE_PATH)


class Settings(BaseSettings):
    APP_ENV: str = "development"

    GROQ_API_KEY: str

    GENERATION_BACKEND: str
    GOOGLE_API_KEY: str
    COHERE_API_KEY: str
    GENERATION_BACKEND: str
    
    model_config = SettingsConfigDict(
        env_file=ENV_FILE_PATH,
        env_file_encoding="utf-8",
        extra="ignore",
    )


# Instantiate settings once so it can be shared across modules
settings = Settings()


def get_settings() -> Settings:
    return settings


