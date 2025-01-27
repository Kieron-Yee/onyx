"""
此文件用于管理Redis中的停止信号处理机制。
主要功能：
1. 管理连接器的停止状态
2. 提供围栏机制来控制连接器的启停
3. 提供重置所有停止状态的功能
"""

import redis


class RedisConnectorStop:
    """Manages interactions with redis for stop signaling. Should only be accessed
    through RedisConnector.
    管理Redis停止信号的交互。只能通过RedisConnector访问。
    
    该类实现了一个基于Redis的围栏机制，用于控制连接器的停止状态。
    通过在Redis中存储围栏标记来实现连接器的状态管理。
    """

    # Redis中围栏键的前缀
    FENCE_PREFIX = "connectorstop_fence"

    def __init__(self, tenant_id: str | None, id: int, redis: redis.Redis) -> None:
        """
        初始化RedisConnectorStop实例
        
        参数:
            tenant_id: 租户ID，可以为空
            id: 连接器ID
            redis: Redis客户端实例
        """
        self.tenant_id: str | None = tenant_id  # 租户ID
        self.id: int = id  # 连接器ID
        self.redis = redis  # Redis客户端实例

        # 构建围栏键名
        self.fence_key: str = f"{self.FENCE_PREFIX}_{id}"

    @property
    def fenced(self) -> bool:
        """
        检查连接器是否处于围栏状态
        
        返回值:
            bool: True表示连接器已被围栏（停止状态），False表示正常状态
        """
        if self.redis.exists(self.fence_key):
            return True

        return False

    def set_fence(self, value: bool) -> None:
        """
        设置连接器的围栏状态
        
        参数:
            value: True表示设置围栏（停止状态），False表示移除围栏
        """
        if not value:
            self.redis.delete(self.fence_key)
            return

        self.redis.set(self.fence_key, 0)

    @staticmethod
    def reset_all(r: redis.Redis) -> None:
        """
        重置所有连接器的围栏状态
        
        参数:
            r: Redis客户端实例
        """
        for key in r.scan_iter(RedisConnectorStop.FENCE_PREFIX + "*"):
            r.delete(key)
