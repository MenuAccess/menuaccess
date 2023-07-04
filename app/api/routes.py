from fastapi import APIRouter, status
from fastapi.responses import PlainTextResponse, Response

from app.api.v1 import health, upload


api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(upload.router)
