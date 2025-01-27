"""
错误处理模块 - 定义了系统中使用的自定义异常类
This module contains custom exception classes used in the system
"""

class EERequiredError(Exception):
    """This error is thrown if an Enterprise Edition feature or API is
    requested but the Enterprise Edition flag is not set.
    当请求企业版功能或API但未设置企业版标志时，将抛出此错误。
    """
    
    """
    企业版功能需求错误
    
    用途：
        当用户尝试使用企业版特性但系统未启用企业版时抛出此异常
    
    继承：
        Exception - Python标准异常基类
    """
