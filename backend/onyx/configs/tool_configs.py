"""
This module contains configuration settings for custom tools and image generation.
本模块包含自定义工具和图像生成的配置设置。
"""

import json
import os


# Image generation output format setting (url or base64)
# 图像生成输出格式设置(url或base64格式)
# 用于指定图像生成工具返回结果的格式，可以是URL链接或base64编码的图像数据
IMAGE_GENERATION_OUTPUT_FORMAT = os.environ.get("IMAGE_GENERATION_OUTPUT_FORMAT", "url")

# If specified, will pass through request headers to the call to API calls made by custom tools
# 如果指定，将把请求头传递给自定义工具的API调用
# 用于配置需要传递到自定义工具API调用中的HTTP请求头列表
CUSTOM_TOOL_PASS_THROUGH_HEADERS: list[str] | None = None
_CUSTOM_TOOL_PASS_THROUGH_HEADERS_RAW = os.environ.get(
    "CUSTOM_TOOL_PASS_THROUGH_HEADERS"
)
if _CUSTOM_TOOL_PASS_THROUGH_HEADERS_RAW:
    try:
        CUSTOM_TOOL_PASS_THROUGH_HEADERS = json.loads(
            _CUSTOM_TOOL_PASS_THROUGH_HEADERS_RAW
        )
    except Exception:
        # need to import here to avoid circular imports
        from onyx.utils.logger import setup_logger

        logger = setup_logger()
        logger.error(
            "Failed to parse CUSTOM_TOOL_PASS_THROUGH_HEADERS, must be a valid JSON object"
        )
