import logging
from typing import Any, Dict, List, Optional

from pydantic import AnyUrl, BaseSettings, Field, PostgresDsn, validator


class Settings(BaseSettings):
    LOG_LEVEL: int = Field(default=logging.INFO, env="LOG_LEVEL")

    VERSION: str = Field(default="v1", env="VERSION")
    DEBUG: bool = Field(default=True, env="DEBUG")

    BACKEND_CORS_ORIGINS: List[AnyUrl] = [
        AnyUrl("http://localhost:5173", scheme="http"),
        AnyUrl(
            "https://fruit-cards-ts.pages.dev",
            scheme="https",
        ),
    ]

    OPENAI_API_KEY: str = Field(
        default="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",
        env="OPENAI_API_KEY",
    )
    GPT_INDEX_QUERY: str = """As a professor generate 5 Questions and Answer in the following format: [{"question": "{Question}", "answer": "{Answer}"}] don't include the number of questions in the query."""
    MENU_QUERY: str = """As a dietician identify menu items that can meet the following dietary restrictions: """

    SUPABASE_URL: str = Field(
        default="https://vfxgbbsngynctbktxjjt.supabase.co", env="SUPABASE_URL"
    )
    SUPABASE_KEY: str = Field(
        default="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InZmeGdiYnNuZ3luY3Ria3R4amp0Iiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTY3NDczOTcwOSwiZXhwIjoxOTkwMzE1NzA5fQ.Eu0AByv4jWD852nfGynBdLKLGmjFV6OBsna3zNITmUE",
        env="SUPABASE_KEY",
    )


settings = Settings()
