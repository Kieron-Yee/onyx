"""NOTE: this needs to be separate from models.py because of circular imports.
Both search/models.py and db/models.py import enums from this file AND
search/models.py imports from db/models.py.
注意：由于循环导入的问题，此文件需要与models.py分开。
search/models.py和db/models.py都从此文件导入枚举，同时search/models.py还导入了db/models.py。
"""

"""
此文件包含了搜索系统中使用的各种枚举类型定义，
用于定义搜索行为、偏好设置和评估方式等重要参数。
"""

from enum import Enum


class RecencyBiasSetting(str, Enum):
    """
    定义搜索结果的时间衰减偏好设置
    """
    FAVOR_RECENT = "favor_recent"  # 2x decay rate / 双倍衰减率，更强烈地偏好最近的内容
    BASE_DECAY = "base_decay"      # 使用基础衰减率
    NO_DECAY = "no_decay"          # 不使用时间衰减
    AUTO = "auto"                  # Determine based on query if to use base_decay or favor_recent
                                  # 根据查询自动决定使用基础衰减还是偏好最近内容


class OptionalSearchSetting(str, Enum):
    """
    定义可选搜索功能的启用设置
    """
    ALWAYS = "always"   # 始终启用
    NEVER = "never"     # 从不启用
    AUTO = "auto"       # Determine whether to run search based on history and latest query
                       # 根据历史记录和最新查询自动决定是否执行搜索


class SearchType(str, Enum):
    """
    定义搜索类型
    """
    KEYWORD = "keyword"    # 关键词搜索
    SEMANTIC = "semantic"  # 语义搜索


class LLMEvaluationType(str, Enum):
    """
    定义大语言模型的评估类型
    """
    AGENTIC = "agentic"        # applies agentic evaluation / 使用主动式评估
    BASIC = "basic"            # applies boolean evaluation / 使用布尔评估
    SKIP = "skip"              # skips evaluation / 跳过评估
    UNSPECIFIED = "unspecified"  # reverts to default / 使用默认评估方式


class QueryFlow(str, Enum):
    """
    定义查询流程类型
    """
    SEARCH = "search"                    # 普通搜索流程
    QUESTION_ANSWER = "question-answer"  # 问答流程
