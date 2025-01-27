"""
此文件用于组合和管理 OpenAI Assistants API 的所有相关路由
包括助手(Assistants)、消息(Messages)、运行(Runs)和线程(Threads)等功能的路由管理
"""

from fastapi import APIRouter

from onyx.server.openai_assistants_api.asssistants_api import (
    router as assistants_router,
)
from onyx.server.openai_assistants_api.messages_api import router as messages_router
from onyx.server.openai_assistants_api.runs_api import router as runs_router
from onyx.server.openai_assistants_api.threads_api import router as threads_router


def get_full_openai_assistants_api_router() -> APIRouter:
    """
    创建并返回一个包含所有 OpenAI Assistants API 相关路由的主路由器
    
    Returns:
        APIRouter: 配置完成的FastAPI路由器，包含所有Assistants API相关的子路由
    """
    # 创建主路由器，设置统一的URL前缀
    router = APIRouter(prefix="/openai-assistants")

    # 包含各个功能模块的路由
    router.include_router(assistants_router)    # 包含助手相关的路由
    router.include_router(runs_router)          # 包含运行相关的路由
    router.include_router(threads_router)       # 包含线程相关的路由
    router.include_router(messages_router)      # 包含消息相关的路由

    return router
