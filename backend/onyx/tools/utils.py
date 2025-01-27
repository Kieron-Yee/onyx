"""
工具函数模块
本模块提供了一系列用于处理工具调用、令牌计算和功能可用性检查的实用函数。
"""

import json

from sqlalchemy.orm import Session

from onyx.configs.app_configs import AZURE_DALLE_API_KEY
from onyx.db.connector import check_connectors_exist
from onyx.db.document import check_docs_exist
from onyx.db.models import LLMProvider
from onyx.natural_language_processing.utils import BaseTokenizer
from onyx.tools.tool import Tool


# 支持显式工具调用的OpenAI模型列表
OPEN_AI_TOOL_CALLING_MODELS = {
    "gpt-3.5-turbo",
    "gpt-4-turbo",
    "gpt-4",
    "gpt-4o",
    "gpt-4o-mini",
}


def explicit_tool_calling_supported(model_provider: str, model_name: str) -> bool:
    """
    检查指定的模型是否支持显式工具调用功能
    
    参数:
        model_provider: str - 模型提供商名称
        model_name: str - 模型名称
    
    返回:
        bool - 如果支持则返回True，否则返回False
    """
    if model_provider == "openai" and model_name in OPEN_AI_TOOL_CALLING_MODELS:
        return True

    return False


def compute_tool_tokens(tool: Tool, llm_tokenizer: BaseTokenizer) -> int:
    """
    计算单个工具的令牌数量
    
    参数:
        tool: Tool - 要计算的工具实例
        llm_tokenizer: BaseTokenizer - 用于令牌化的分词器
    
    返回:
        int - 工具定义的令牌数量
    """
    return len(llm_tokenizer.encode(json.dumps(tool.tool_definition())))


def compute_all_tool_tokens(tools: list[Tool], llm_tokenizer: BaseTokenizer) -> int:
    """
    计算所有工具的总令牌数量
    
    参数:
        tools: list[Tool] - 工具列表
        llm_tokenizer: BaseTokenizer - 用于令牌化的分词器
    
    返回:
        int - 所有工具定义的总令牌数量
    """
    return sum(compute_tool_tokens(tool, llm_tokenizer) for tool in tools)


def is_image_generation_available(db_session: Session) -> bool:
    """
    检查图像生成功能是否可用
    
    参数:
        db_session: Session - 数据库会话
    
    返回:
        bool - 如果图像生成功能可用则返回True，否则返回False
    """
    providers = db_session.query(LLMProvider).all()
    for provider in providers:
        if provider.provider == "openai":
            return True

    return bool(AZURE_DALLE_API_KEY)


def is_document_search_available(db_session: Session) -> bool:
    """
    检查文档搜索功能是否可用
    
    参数:
        db_session: Session - 数据库会话
    
    返回:
        bool - 如果文档搜索功能可用则返回True，否则返回False
    """
    docs_exist = check_docs_exist(db_session)
    connectors_exist = check_connectors_exist(db_session)
    return docs_exist or connectors_exist
