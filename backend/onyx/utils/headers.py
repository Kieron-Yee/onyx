"""
这个模块用于处理HTTP请求头相关的工具函数。
主要功能包括：
1. 清理和转换请求头格式
2. 获取相关的请求头信息
3. 构建LLM额外请求头
"""

from typing import TypedDict

from fastapi.datastructures import Headers

from onyx.configs.model_configs import LITELLM_EXTRA_HEADERS
from onyx.configs.model_configs import LITELLM_PASS_THROUGH_HEADERS
from onyx.configs.tool_configs import CUSTOM_TOOL_PASS_THROUGH_HEADERS
from onyx.utils.logger import setup_logger

logger = setup_logger()


class HeaderItemDict(TypedDict):
    """
    定义请求头项的类型字典
    属性:
        key: 请求头的键
        value: 请求头的值
    """
    key: str
    value: str


def clean_header_list(headers_to_clean: list[HeaderItemDict]) -> dict[str, str]:
    """
    清理请求头列表，去除重复的请求头
    
    参数:
        headers_to_clean: 需要清理的请求头列表
    
    返回:
        清理后的请求头字典
    """
    cleaned_headers: dict[str, str] = {}
    for item in headers_to_clean:
        key = item["key"]
        value = item["value"]
        if key in cleaned_headers:
            logger.warning(
                f"Duplicate header {key} found in custom headers, ignoring..."
            )
            continue
        cleaned_headers[key] = value
    return cleaned_headers


def header_dict_to_header_list(header_dict: dict[str, str]) -> list[HeaderItemDict]:
    """
    将请求头字典转换为请求头列表格式
    
    参数:
        header_dict: 请求头字典
    
    返回:
        转换后的请求头列表
    """
    return [{"key": key, "value": value} for key, value in header_dict.items()]


def header_list_to_header_dict(header_list: list[HeaderItemDict]) -> dict[str, str]:
    """
    将请求头列表转换为请求头字典格式
    
    参数:
        header_list: 请求头列表
    
    返回:
        转换后的请求头字典
    """
    return {header["key"]: header["value"] for header in header_list}


def get_relevant_headers(
    headers: dict[str, str] | Headers, desired_headers: list[str] | None
) -> dict[str, str]:
    """
    从请求头中获取指定的相关请求头
    
    参数:
        headers: 原始请求头
        desired_headers: 需要获取的请求头列表
    
    返回:
        过滤后的相关请求头字典
    """
    if not desired_headers:
        return {}

    pass_through_headers: dict[str, str] = {}
    for key in desired_headers:
        if key in headers:
            pass_through_headers[key] = headers[key]
        else:
            # fastapi makes all header keys lowercase, handling that here
            lowercase_key = key.lower()
            if lowercase_key in headers:
                pass_through_headers[lowercase_key] = headers[lowercase_key]

    return pass_through_headers


def get_litellm_additional_request_headers(
    headers: dict[str, str] | Headers
) -> dict[str, str]:
    """
    获取LiteLLM额外的请求头
    
    参数:
        headers: 原始请求头
    
    返回:
        LiteLLM相关的请求头
    """
    return get_relevant_headers(headers, LITELLM_PASS_THROUGH_HEADERS)


def build_llm_extra_headers(
    additional_headers: dict[str, str] | None = None
) -> dict[str, str]:
    """
    构建LLM额外的请求头
    
    参数:
        additional_headers: 额外的请求头字典
    
    返回:
        合并后的额外请求头字典
    """
    extra_headers: dict[str, str] = {}
    if additional_headers:
        extra_headers.update(additional_headers)
    if LITELLM_EXTRA_HEADERS:
        extra_headers.update(LITELLM_EXTRA_HEADERS)
    return extra_headers


def get_custom_tool_additional_request_headers(
    headers: dict[str, str] | Headers
) -> dict[str, str]:
    """
    获取自定义工具的额外请求头
    
    参数:
        headers: 原始请求头
    
    返回:
        自定义工具相关的请求头
    """
    return get_relevant_headers(headers, CUSTOM_TOOL_PASS_THROUGH_HEADERS)
