"""
此文件定义了HTML连接器转换链接策略的枚举类型。
主要用于处理HTML文档中链接的不同转换方式。
"""

from enum import Enum


class HtmlBasedConnectorTransformLinksStrategy(str, Enum):
    """
    HTML连接器链接转换策略枚举类。
    定义了处理HTML文档中链接的不同策略选项。
    继承自str和Enum，确保枚举值为字符串类型。
    """
    
    # remove links entirely
    # 完全移除链接
    STRIP = "strip"
    
    # turn HTML links into markdown links
    # 将HTML链接转换为markdown格式的链接
    MARKDOWN = "markdown"
