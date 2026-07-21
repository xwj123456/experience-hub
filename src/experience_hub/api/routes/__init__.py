"""Versioned Experience Hub HTTP route composition."""

from fastapi import APIRouter

from experience_hub.api.routes.agents import router as agents_router
from experience_hub.api.routes.experiences import router as experiences_router
from experience_hub.api.routes.inspiration import router as inspiration_router
from experience_hub.api.routes.lifecycle import router as lifecycle_router
from experience_hub.api.routes.sharing import router as sharing_router

api_router = APIRouter()
api_router.include_router(agents_router)
api_router.include_router(experiences_router)
api_router.include_router(lifecycle_router)
api_router.include_router(sharing_router)
api_router.include_router(inspiration_router)

__all__ = ["api_router"]
