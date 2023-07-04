import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from app.api import routes
from app.core.config import settings


logger = logging.getLogger(__name__)

tags_metadata = [
    {
        "name": "health",
        "description": "Health check for api",
    }
]

app = FastAPI(
    title="fruit-cards-api",
    description="base project for fastapi backend",
)


async def on_startup() -> None:
    logger.info("FastAPI app running...")


app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_event_handler("startup", on_startup)


@app.get("/")
def get_root():
    return {"message": "Fruit Cards running on AWS"}


app.include_router(routes.api_router, prefix=f"/{settings.VERSION}")
