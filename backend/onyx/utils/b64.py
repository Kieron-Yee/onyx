"""
文件功能：用于处理Base64编码的图片数据，包括图片类型的检测和识别
主要提供了两个函数：
1. 从字节数据中获取图片类型
2. 从Base64字符串中获取图片类型
"""

import base64


def get_image_type_from_bytes(raw_b64_bytes: bytes) -> str:
    """
    通过检查图片文件的魔数（Magic Number）来判断图片类型
    
    参数:
        raw_b64_bytes: bytes - 需要检查的图片字节数据
        
    返回:
        str - 返回图片的MIME类型
        
    异常:
        ValueError - 当图片格式不支持时抛出异常
    """
    magic_number = raw_b64_bytes[:4]

    if magic_number.startswith(b"\x89PNG"):
        mime_type = "image/png"
    elif magic_number.startswith(b"\xFF\xD8"):
        mime_type = "image/jpeg"
    elif magic_number.startswith(b"GIF8"):
        mime_type = "image/gif"
    elif magic_number.startswith(b"RIFF") and raw_b64_bytes[8:12] == b"WEBP":
        mime_type = "image/webp"
    else:
        raise ValueError(
            "Unsupported image format - only PNG, JPEG, GIF, and WEBP are supported."
            "不支持的图片格式 - 仅支持 PNG、JPEG、GIF 和 WEBP 格式"
        )

    return mime_type


def get_image_type(raw_b64_string: str) -> str:
    """
    从Base64编码的字符串中获取图片类型
    
    参数:
        raw_b64_string: str - Base64编码的图片数据字符串
        
    返回:
        str - 返回图片的MIME类型
        
    异常:
        ValueError - 当图片格式不支持时抛出异常
        binascii.Error - 当Base64解码失败时抛出异常
    """
    binary_data = base64.b64decode(raw_b64_string)
    return get_image_type_from_bytes(binary_data)
