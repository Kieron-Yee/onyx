"""
此文件实现了Redis对象助手基类，用于处理与Redis相关的对象操作。
主要功能包括:
1. 提供Redis键值管理
2. 处理任务ID和围栏键的生成和解析
3. 定义任务生成的抽象接口
"""

from abc import ABC
from abc import abstractmethod

from celery import Celery
from redis import Redis
from redis.lock import Lock as RedisLock
from sqlalchemy.orm import Session

from onyx.redis.redis_pool import get_redis_client


class RedisObjectHelper(ABC):
    """
    Redis对象助手基类，提供Redis对象操作的基本功能
    """
    # Redis键前缀
    PREFIX = "base"
    # 围栏键前缀
    FENCE_PREFIX = PREFIX + "_fence"
    # 任务集合键前缀
    TASKSET_PREFIX = PREFIX + "_taskset"

    def __init__(self, tenant_id: str | None, id: str):
        """
        初始化Redis对象助手
        
        Args:
            tenant_id: 租户ID，可以为空
            id: 对象ID
        """
        self._tenant_id: str | None = tenant_id
        self._id: str = id
        self.redis = get_redis_client(tenant_id=tenant_id)

    @property
    def task_id_prefix(self) -> str:
        """获取任务ID前缀"""
        return f"{self.PREFIX}_{self._id}"

    @property
    def fence_key(self) -> str:
        """
        获取围栏键
        example: documentset_fence_1
        """
        return f"{self.FENCE_PREFIX}_{self._id}"

    @property
    def taskset_key(self) -> str:
        """
        获取任务集合键
        example: documentset_taskset_1
        """
        return f"{self.TASKSET_PREFIX}_{self._id}"

    @staticmethod
    def get_id_from_fence_key(key: str) -> str | None:
        """
        从围栏键中提取对象ID
        
        Extracts the object ID from a fence key in the format `PREFIX_fence_X`.
        从格式为"PREFIX_fence_X"的围栏键中提取对象ID。

        Args:
            key: 围栏键字符串

        Returns:
            提取的ID（如果键格式正确），否则返回None
        """
        parts = key.split("_")
        if len(parts) != 3:
            return None

        object_id = parts[2]
        return object_id

    @staticmethod
    def get_id_from_task_id(task_id: str) -> str | None:
        """
        从任务ID字符串中提取对象ID
        
        Extracts the object ID from a task ID string.
        从任务ID字符串中提取对象ID。

        This method assumes the task ID is formatted as `prefix_objectid_suffix`, where:
        此方法假设任务ID格式为"prefix_objectid_suffix"，其中：
        - `prefix` is an arbitrary string (e.g., the name of the task or entity),
        - prefix是任意字符串（例如任务或实体的名称）
        - `objectid` is the ID you want to extract,
        - objectid是要提取的ID
        - `suffix` is another arbitrary string (e.g., a UUID).
        - suffix是另一个任意字符串（例如UUID）

        Args:
            task_id: 需要提取对象ID的任务ID字符串

        Returns:
            如果任务ID格式正确则返回提取的对象ID，否则返回None
        """
        parts = task_id.split("_")
        if len(parts) != 3:
            return None

        object_id = parts[1]
        return object_id

    @abstractmethod
    def generate_tasks(
        self,
        celery_app: Celery,
        db_session: Session,
        redis_client: Redis,
        lock: RedisLock,
        tenant_id: str | None,
    ) -> tuple[int, int] | None:
        """
        生成任务的抽象方法
        
        First element should be the number of actual tasks generated, second should
        be the number of docs that were candidates to be synced for the cc pair.
        返回的第一个元素应该是实际生成的任务数量，第二个元素应该是CC对需要同步的文档候选数量。

        The need for this is when we are syncing stale docs referenced by multiple
        connectors. In a single pass across multiple cc pairs, we only want a task
        for be created for a particular document id the first time we see it.
        The rest can be skipped.
        这是因为当我们同步被多个连接器引用的过期文档时，在跨多个CC对的单次遍历中，
        我们只希望在第一次看到特定文档ID时为其创建任务，其余的可以跳过。

        Args:
            celery_app: Celery应用实例
            db_session: 数据库会话
            redis_client: Redis客户端
            lock: Redis锁
            tenant_id: 租户ID，可以为空

        Returns:
            返回一个元组(已生成任务数, 候选文档数)或None
        """
