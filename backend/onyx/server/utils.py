"""
工具函数模块
本模块包含了一些通用的工具函数，主要用于处理JSON序列化、凭证掩码和基本认证等功能。
"""

import json
from datetime import datetime
from typing import Any

from fastapi import HTTPException
from fastapi import status

from onyx.connectors.google_utils.shared_constants import (
    DB_CREDENTIALS_AUTHENTICATION_METHOD,
)


class BasicAuthenticationError(HTTPException):
    """
    基本认证错误异常类
    继承自FastAPI的HTTPException，用于处理认证相关的错误
    """
    def __init__(self, detail: str):
        super().__init__(status_code=status.HTTP_403_FORBIDDEN, detail=detail)


class DateTimeEncoder(json.JSONEncoder):
    """Custom JSON encoder that converts datetime objects to ISO format strings.
    自定义JSON编码器，用于将datetime对象转换为ISO格式的字符串
    """

    def default(self, obj: Any) -> Any:
        """
        重写default方法，处理datetime类型的序列化
        
        参数:
            obj: 需要序列化的对象
            
        返回:
            序列化后的数据
        """
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


def get_json_line(
    json_dict: dict[str, Any], encoder: type[json.JSONEncoder] = DateTimeEncoder
) -> str:
    """Convert a dictionary to a JSON string with datetime handling, and add a newline.
    将字典转换为JSON字符串，支持datetime处理，并添加换行符

    Args:
        json_dict: The dictionary to be converted to JSON.
                  要转换成JSON的字典
        encoder: JSON encoder class to use, defaults to DateTimeEncoder.
                使用的JSON编码器类，默认为DateTimeEncoder

    Returns:
        A JSON string representation of the input dictionary with a newline character.
        输入字典的JSON字符串表示，末尾带有换行符
    """
    return json.dumps(json_dict, cls=encoder) + "\n"


def mask_string(sensitive_str: str) -> str:
    """
    对敏感字符串进行掩码处理
    
    参数:
        sensitive_str: 需要掩码的敏感字符串
        
    返回:
        掩码后的字符串，只显示最后4个字符
    """
    return "****...**" + sensitive_str[-4:]


def mask_credential_dict(credential_dict: dict[str, Any]) -> dict[str, str]:
    """
    对凭证字典中的敏感信息进行掩码处理
    
    参数:
        credential_dict: 包含凭证信息的字典
        
    返回:
        处理后的凭证字典，敏感信息被掩码
        
    异常:
        ValueError: 当凭证包含不支持的数据类型时抛出
    """
    masked_creds = {}
    for key, val in credential_dict.items():
        if isinstance(val, str):
            # we want to pass the authentication_method field through so the frontend
            # can disambiguate credentials created by different methods
            # 我们需要保留authentication_method字段，以便前端可以区分不同方法创建的凭证
            if key == DB_CREDENTIALS_AUTHENTICATION_METHOD:
                masked_creds[key] = val
            else:
                masked_creds[key] = mask_string(val)
            continue

        if isinstance(val, int):
            masked_creds[key] = "*****"
            continue

        raise ValueError(
            f"Unable to mask credentials of type other than string, cannot process request."
            f"Recieved type: {type(val)}"
        )

    return masked_creds
