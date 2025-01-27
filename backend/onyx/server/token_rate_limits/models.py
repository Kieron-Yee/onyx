"""
This file contains models for token rate limiting functionality.
该文件包含令牌速率限制功能相关的模型定义。
"""

from pydantic import BaseModel
from onyx.db.models import TokenRateLimit


class TokenRateLimitArgs(BaseModel):
    """
    Token rate limit arguments model for receiving rate limit settings.
    令牌速率限制参数模型，用于接收速率限制设置。

    Attributes:
        enabled (bool): 是否启用速率限制
        token_budget (int): 令牌预算数量
        period_hours (int): 限制周期（小时）
    """
    enabled: bool
    token_budget: int
    period_hours: int


class TokenRateLimitDisplay(BaseModel):
    """
    Token rate limit display model for API responses.
    令牌速率限制显示模型，用于API响应。

    Attributes:
        token_id (int): 令牌ID
        enabled (bool): 是否启用速率限制
        token_budget (int): 令牌预算数量
        period_hours (int): 限制周期（小时）
    """
    token_id: int
    enabled: bool
    token_budget: int
    period_hours: int

    @classmethod
    def from_db(cls, token_rate_limit: TokenRateLimit) -> "TokenRateLimitDisplay":
        """
        Create a display model instance from database model.
        从数据库模型创建显示模型实例。

        Args:
            token_rate_limit (TokenRateLimit): 数据库中的令牌速率限制模型实例

        Returns:
            TokenRateLimitDisplay: 用于显示的令牌速率限制模型实例
        """
        return cls(
            token_id=token_rate_limit.id,
            enabled=token_rate_limit.enabled,
            token_budget=token_rate_limit.token_budget,
            period_hours=token_rate_limit.period_hours,
        )
