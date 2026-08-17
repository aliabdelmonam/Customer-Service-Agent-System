from fastapi import APIRouter

router = APIRouter()

@router.get("", status_code=200)
async def check_health():
    """
    Simple health check endpoint to verify the service status.
    """
    return {
        "status": "healthy",
        "message": "Customer service agent API is operational."
    }
