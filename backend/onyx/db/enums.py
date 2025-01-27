"""
此文件包含了系统中使用的所有枚举类型定义。
这些枚举类型用于标识各种状态、模式和访问类型等系统配置。
"""

from enum import Enum as PyEnum


# 索引状态枚举类，用于标识文档索引的不同状态
class IndexingStatus(str, PyEnum):
    NOT_STARTED = "not_started"        # 未开始
    IN_PROGRESS = "in_progress"        # 进行中
    SUCCESS = "success"                # 成功
    CANCELED = "canceled"              # 已取消
    FAILED = "failed"                  # 失败
    COMPLETED_WITH_ERRORS = "completed_with_errors"  # 完成但有错误

    # 判断当前状态是否为终止状态的方法
    def is_terminal(self) -> bool:
        terminal_states = {
            IndexingStatus.SUCCESS,
            IndexingStatus.COMPLETED_WITH_ERRORS,
            IndexingStatus.CANCELED,
            IndexingStatus.FAILED,
        }
        return self in terminal_states


# 索引模式枚举类，用于定义索引操作的模式
class IndexingMode(str, PyEnum):
    UPDATE = "update"    # 更新模式
    REINDEX = "reindex"  # 重新索引模式


# 删除状态枚举类，用于标识删除操作的不同状态
# 这些状态可能在将来会与其他状态有所不同，因此保留这个独立的枚举类
class DeletionStatus(str, PyEnum):
    NOT_STARTED = "not_started"  # 未开始
    IN_PROGRESS = "in_progress"  # 进行中
    SUCCESS = "success"          # 成功
    FAILED = "failed"           # 失败


# Celery任务状态枚举类，与Celery任务状态保持一致
class TaskStatus(str, PyEnum):
    PENDING = "PENDING"   # 等待中
    STARTED = "STARTED"   # 已开始
    SUCCESS = "SUCCESS"   # 成功
    FAILURE = "FAILURE"   # 失败


# 索引模型状态枚举类，用于标识模型的时态状态
class IndexModelStatus(str, PyEnum):
    PAST = "PAST"        # 过去
    PRESENT = "PRESENT"  # 现在
    FUTURE = "FUTURE"    # 未来


# 聊天会话共享状态枚举类，用于定义会话的可见性
class ChatSessionSharedStatus(str, PyEnum):
    PUBLIC = "public"    # 公开
    PRIVATE = "private"  # 私密


# 连接器凭证对状态枚举类，用于标识连接器凭证的状态
class ConnectorCredentialPairStatus(str, PyEnum):
    ACTIVE = "ACTIVE"     # 活跃状态
    PAUSED = "PAUSED"     # 暂停状态
    DELETING = "DELETING" # 删除中状态

    # 判断当前状态是否为活跃状态的方法
    def is_active(self) -> bool:
        return self == ConnectorCredentialPairStatus.ACTIVE


# 访问类型枚举类，用于定义资源的访问权限类型
class AccessType(str, PyEnum):
    PUBLIC = "public"   # 公开访问
    PRIVATE = "private" # 私有访问
    SYNC = "sync"      # 同步访问
