from fastapi import APIRouter
from src.api.v1.endpoints import asr, chat, health

api_router = APIRouter()

api_router.include_router(health.router, prefix="/health", tags=["Health Check"])
api_router.include_router(chat.router, prefix="/chat", tags=["Chatbot Interaction"])
api_router.include_router(asr.router, prefix="/asr", tags=["Speech Recognition"])
