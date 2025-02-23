import torch
from fastapi import APIRouter
from fastapi import Response

router = APIRouter(prefix="/api")

# Health check endpoint
# 健康检查端点
@router.get("/health")
async def healthcheck() -> Response:
    return Response(status_code=200)

# GPU status endpoint
# GPU状态端点
@router.get("/gpu-status")
async def gpu_status() -> dict[str, bool | str]:
    if torch.cuda.is_available():
        return {"gpu_available": True, "type": "cuda"}
    elif torch.backends.mps.is_available():
        return {"gpu_available": True, "type": "mps"}
    else:
        return {"gpu_available": False, "type": "none"}
