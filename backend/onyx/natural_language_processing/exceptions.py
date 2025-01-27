"""
This module contains custom exceptions for natural language processing operations.
此模块包含自然语言处理操作的自定义异常。
"""

class ModelServerRateLimitError(Exception):
    """
    Exception raised for rate limiting errors from the model server.
    模型服务器限流错误异常。
    
    这个异常类用于处理当模型服务器因为请求频率过高而触发限流机制时的情况。
    当服务达到其设定的请求限制时，将抛出此异常。
    """
    pass
