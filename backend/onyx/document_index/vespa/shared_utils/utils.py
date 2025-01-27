"""
此模块提供了与Vespa搜索引擎交互的辅助工具函数。
主要功能包括：
- 文本字符验证
- 文档ID字符处理
- Unicode字符过滤
- Vespa HTTP客户端配置
"""

import re
from typing import cast

import httpx

from onyx.configs.app_configs import MANAGED_VESPA
from onyx.configs.app_configs import VESPA_CLOUD_CERT_PATH
from onyx.configs.app_configs import VESPA_CLOUD_KEY_PATH
from onyx.configs.app_configs import VESPA_REQUEST_TIMEOUT

# NOTE: This does not seem to be used in reality despite the Vespa Docs pointing to this code
# See here for reference: https://docs.vespa.ai/en/documents.html
# https://github.com/vespa-engine/vespa/blob/master/vespajlib/src/main/java/com/yahoo/text/Text.java
# 注：尽管Vespa文档指向此代码，但实际上似乎并未使用

# Define allowed ASCII characters
# 定义允许的ASCII字符
ALLOWED_ASCII_CHARS: list[bool] = [False] * 0x80
ALLOWED_ASCII_CHARS[0x9] = True  # tab
ALLOWED_ASCII_CHARS[0xA] = True  # newline
ALLOWED_ASCII_CHARS[0xD] = True  # carriage return
for i in range(0x20, 0x7F):
    ALLOWED_ASCII_CHARS[i] = True  # printable ASCII chars
ALLOWED_ASCII_CHARS[0x7F] = True  # del - discouraged, but allowed


def is_text_character(codepoint: int) -> bool:
    """Returns whether the given codepoint is a valid text character.
    判断给定的代码点是否是有效的文本字符。

    Args:
        codepoint (int): Unicode代码点值

    Returns:
        bool: 如果是有效的文本字符返回True，否则返回False
    """
    if codepoint < 0x80:
        return ALLOWED_ASCII_CHARS[codepoint]
    if codepoint < 0xD800:
        return True
    if codepoint <= 0xDFFF:
        return False
    if codepoint < 0xFDD0:
        return True
    if codepoint <= 0xFDEF:
        return False
    if codepoint >= 0x10FFFE:
        return False
    return (codepoint & 0xFFFF) < 0xFFFE


def replace_invalid_doc_id_characters(text: str) -> str:
    """Replaces invalid document ID characters in text.
    替换文本中无效的文档ID字符。

    Args:
        text (str): 需要处理的原始文本

    Returns:
        str: 替换无效字符后的文本
    
    说明：
        目前主要处理单引号字符的替换，将其转换为下划线
    """
    # There may be a more complete set of replacements that need to be made but Vespa docs are unclear
    # and users only seem to be running into this error with single quotes
    # 可能还需要更完整的替换集，但Vespa文档不够明确，用户目前主要遇到单引号的问题
    return text.replace("'", "_")


def remove_invalid_unicode_chars(text: str) -> str:
    """Vespa does not take in unicode chars that aren't valid for XML.
    This removes them.
    Vespa不接受XML中无效的Unicode字符，此函数用于移除这些字符。

    Args:
        text (str): 需要处理的原始文本

    Returns:
        str: 移除无效Unicode字符后的文本
    """
    _illegal_xml_chars_RE: re.Pattern = re.compile(
        "[\x00-\x08\x0b\x0c\x0e-\x1F\uD800-\uDFFF\uFFFE\uFFFF]"
    )
    return _illegal_xml_chars_RE.sub("", text)


def get_vespa_http_client(no_timeout: bool = False, http2: bool = True) -> httpx.Client:
    """Configure and return an HTTP client for communicating with Vespa,
    including authentication if needed.
    配置并返回用于与Vespa通信的HTTP客户端，根据需要包含认证信息。

    Args:
        no_timeout (bool, optional): 是否禁用超时设置. Defaults to False.
        http2 (bool, optional): 是否启用HTTP/2. Defaults to True.

    Returns:
        httpx.Client: 配置好的HTTP客户端实例
    """
    return httpx.Client(
        cert=cast(tuple[str, str], (VESPA_CLOUD_CERT_PATH, VESPA_CLOUD_KEY_PATH))
        if MANAGED_VESPA
        else None,
        verify=False if not MANAGED_VESPA else True,
        timeout=None if no_timeout else VESPA_REQUEST_TIMEOUT,
        http2=http2,
    )
