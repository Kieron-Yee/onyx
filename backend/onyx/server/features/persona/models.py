"""
This file contains the data models related to personas and their configurations
此文件包含与personas及其配置相关的数据模型

主要功能：
1. 定义了persona相关的请求和响应模型
2. 提供了persona数据的序列化和反序列化功能
3. 包含了persona类别管理的相关模型
"""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel
from pydantic import Field

from onyx.context.search.enums import RecencyBiasSetting
from onyx.db.models import Persona
from onyx.db.models import PersonaCategory
from onyx.db.models import StarterMessage
from onyx.server.features.document_set.models import DocumentSet
from onyx.server.features.prompt.models import PromptSnapshot
from onyx.server.features.tool.models import ToolSnapshot
from onyx.server.models import MinimalUserSnapshot
from onyx.utils.logger import setup_logger

logger = setup_logger()


# More minimal request for generating a persona prompt
class GenerateStarterMessageRequest(BaseModel):
    """
    用于生成persona初始消息的请求模型
    
    属性：
        name: persona的名称
        description: persona的描述
        instructions: 指令内容
        document_set_ids: 文档集ID列表
    """
    name: str
    description: str
    instructions: str
    document_set_ids: list[int]


class CreatePersonaRequest(BaseModel):
    """
    创建persona的请求模型
    
    属性：
        name: persona名称
        description: persona描述
        num_chunks: 块数
        llm_relevance_filter: 是否启用LLM相关性过滤
        is_public: 是否公开
        llm_filter_extraction: 是否启用LLM过滤提取
        recency_bias: 时间偏好设置
        prompt_ids: 提示词ID列表
        document_set_ids: 文档集ID列表
        tool_ids: 工具ID列表
        llm_model_provider_override: LLM提供商覆盖设置
        llm_model_version_override: LLM版本覆盖设置
        starter_messages: 初始消息列表
        users: 可访问的用户UUID列表
        groups: 可访问的用户组ID列表
        icon_color: 图标颜色
        icon_shape: 图标形状
        uploaded_image_id: 上传的图片ID
        remove_image: 是否移除图片
        is_default_persona: 是否为默认persona
        display_priority: 显示优先级
        search_start_date: 搜索起始日期
        category_id: 类别ID
    """
    name: str
    description: str
    num_chunks: float
    llm_relevance_filter: bool
    is_public: bool
    llm_filter_extraction: bool
    recency_bias: RecencyBiasSetting
    prompt_ids: list[int]
    document_set_ids: list[int]
    # e.g. ID of SearchTool or ImageGenerationTool or <USER_DEFINED_TOOL>
    tool_ids: list[int]
    llm_model_provider_override: str | None = None
    llm_model_version_override: str | None = None
    starter_messages: list[StarterMessage] | None = None
    # For Private Personas, who should be able to access these
    users: list[UUID] = Field(default_factory=list)
    groups: list[int] = Field(default_factory=list)
    icon_color: str | None = None
    icon_shape: int | None = None
    uploaded_image_id: str | None = None  # New field for uploaded image
    remove_image: bool | None = None
    is_default_persona: bool = False
    display_priority: int | None = None
    search_start_date: datetime | None = None
    category_id: int | None = None


class PersonaSnapshot(BaseModel):
    """
    Persona快照模型，用于序列化和展示Persona信息
    
    属性：
        [属性列表与CreatePersonaRequest类似，此处省略]
    """
    id: int
    owner: MinimalUserSnapshot | None
    name: str
    is_visible: bool
    is_public: bool
    display_priority: int | None
    description: str
    num_chunks: float | None
    llm_relevance_filter: bool
    llm_filter_extraction: bool
    llm_model_provider_override: str | None
    llm_model_version_override: str | None
    starter_messages: list[StarterMessage] | None
    builtin_persona: bool
    prompts: list[PromptSnapshot]
    tools: list[ToolSnapshot]
    document_sets: list[DocumentSet]
    users: list[MinimalUserSnapshot]
    groups: list[int]
    icon_color: str | None
    icon_shape: int | None
    uploaded_image_id: str | None = None
    is_default_persona: bool
    search_start_date: datetime | None = None
    category_id: int | None = None

    @classmethod
    def from_model(
        cls, persona: Persona, allow_deleted: bool = False
    ) -> "PersonaSnapshot":
        """
        从数据库模型创建Persona快照
        
        参数：
            persona: Persona数据库模型实例
            allow_deleted: 是否允许已删除的Persona
            
        返回：
            PersonaSnapshot实例
            
        异常：
            ValueError: 当persona已删除且allow_deleted为False时抛出
        """
        if persona.deleted:
            error_msg = f"Persona with ID {persona.id} has been deleted"
            if not allow_deleted:
                raise ValueError(error_msg)
            else:
                logger.warning(error_msg)

        return PersonaSnapshot(
            id=persona.id,
            name=persona.name,
            owner=(
                MinimalUserSnapshot(id=persona.user.id, email=persona.user.email)
                if persona.user
                else None
            ),
            is_visible=persona.is_visible,
            is_public=persona.is_public,
            display_priority=persona.display_priority,
            description=persona.description,
            num_chunks=persona.num_chunks,
            llm_relevance_filter=persona.llm_relevance_filter,
            llm_filter_extraction=persona.llm_filter_extraction,
            llm_model_provider_override=persona.llm_model_provider_override,
            llm_model_version_override=persona.llm_model_version_override,
            starter_messages=persona.starter_messages,
            builtin_persona=persona.builtin_persona,
            is_default_persona=persona.is_default_persona,
            prompts=[PromptSnapshot.from_model(prompt) for prompt in persona.prompts],
            tools=[ToolSnapshot.from_model(tool) for tool in persona.tools],
            document_sets=[
                DocumentSet.from_model(document_set_model)
                for document_set_model in persona.document_sets
            ],
            users=[
                MinimalUserSnapshot(id=user.id, email=user.email)
                for user in persona.users
            ],
            groups=[user_group.id for user_group in persona.groups],
            icon_color=persona.icon_color,
            icon_shape=persona.icon_shape,
            uploaded_image_id=persona.uploaded_image_id,
            search_start_date=persona.search_start_date,
            category_id=persona.category_id,
        )


class PromptTemplateResponse(BaseModel):
    """
    提示模板响应模型
    
    属性：
        final_prompt_template: 最终的提示模板字符串
    """
    final_prompt_template: str


class PersonaSharedNotificationData(BaseModel):
    """
    Persona共享通知数据模型
    
    属性：
        persona_id: 被共享的PersonaID
    """
    persona_id: int


class ImageGenerationToolStatus(BaseModel):
    """
    图片生成工具状态模型
    
    属性：
        is_available: 工具是否可用
    """
    is_available: bool


class PersonaCategoryCreate(BaseModel):
    """
    Persona类别创建请求模型
    
    属性：
        name: 类别名称
        description: 类别描述
    """
    name: str
    description: str


class PersonaCategoryResponse(BaseModel):
    """
    Persona类别响应模型
    
    属性：
        id: 类别ID
        name: 类别名称
        description: 类别描述
    """
    id: int
    name: str
    description: str | None

    @classmethod
    def from_model(cls, category: PersonaCategory) -> "PersonaCategoryResponse":
        """
        从数据库模型创建类别响应对象
        
        参数：
            category: PersonaCategory数据库模型实例
            
        返回：
            PersonaCategoryResponse实例
        """
        return PersonaCategoryResponse(
            id=category.id,
            name=category.name,
            description=category.description,
        )
