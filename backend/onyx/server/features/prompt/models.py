"""
此模块定义了与提示(Prompt)相关的Pydantic模型。
包含创建提示请求的模型定义和提示快照的数据模型。
这些模型用于API请求验证和响应序列化。
"""

from pydantic import BaseModel
from onyx.db.models import Prompt


class CreatePromptRequest(BaseModel):
    """
    创建提示请求的数据模型
    
    属性说明：
        name (str): 提示的名称
        description (str): 提示的描述信息
        system_prompt (str): 系统级提示内容
        task_prompt (str): 任务级提示内容
        include_citations (bool): 是否包含引用，默认为False
        datetime_aware (bool): 是否启用日期时间感知，默认为False
        persona_ids (list[int] | None): 关联的角色ID列表，可为空
    """
    name: str
    description: str
    system_prompt: str
    task_prompt: str
    include_citations: bool = False
    datetime_aware: bool = False
    persona_ids: list[int] | None = None


class PromptSnapshot(BaseModel):
    """
    提示快照的数据模型，用于记录提示的当前状态
    
    属性说明：
        id (int): 提示的唯一标识符
        name (str): 提示的名称
        description (str): 提示的描述信息
        system_prompt (str): 系统级提示内容
        task_prompt (str): 任务级提示内容
        include_citations (bool): 是否包含引用
        datetime_aware (bool): 是否启用日期时间感知
        default_prompt (bool): 是否为默认提示
    """
    id: int
    name: str
    description: str
    system_prompt: str
    task_prompt: str
    include_citations: bool
    datetime_aware: bool
    default_prompt: bool

    @classmethod
    def from_model(cls, prompt: Prompt) -> "PromptSnapshot":
        """
        从数据库模型创建提示快照实例

        参数：
            prompt (Prompt): 数据库中的提示模型实例

        返回：
            PromptSnapshot: 创建的提示快照实例

        异常：
            ValueError: 当提示已被删除时抛出
        """
        if prompt.deleted:
            raise ValueError("Prompt has been deleted")  # 提示已被删除

        return PromptSnapshot(
            id=prompt.id,
            name=prompt.name,
            description=prompt.description,
            system_prompt=prompt.system_prompt,
            task_prompt=prompt.task_prompt,
            include_citations=prompt.include_citations,
            datetime_aware=prompt.datetime_aware,
            default_prompt=prompt.default_prompt,
        )
