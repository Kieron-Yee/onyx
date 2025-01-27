"""
该模块定义了查询和聊天功能相关的数据模型。
包含了聊天会话、消息、搜索反馈等功能的数据结构定义。
主要用于处理聊天系统的请求和响应数据。
"""

from datetime import datetime
from typing import Any
from typing import TYPE_CHECKING
from uuid import UUID

from pydantic import BaseModel
from pydantic import model_validator

from onyx.chat.models import PersonaOverrideConfig
from onyx.chat.models import RetrievalDocs
from onyx.configs.constants import DocumentSource
from onyx.configs.constants import MessageType
from onyx.configs.constants import SearchFeedbackType
from onyx.configs.constants import SessionType
from onyx.context.search.models import BaseFilters
from onyx.context.search.models import ChunkContext
from onyx.context.search.models import RerankingDetails
from onyx.context.search.models import RetrievalDetails
from onyx.context.search.models import SearchDoc
from onyx.context.search.models import Tag
from onyx.db.enums import ChatSessionSharedStatus
from onyx.file_store.models import FileDescriptor
from onyx.llm.override_models import LLMOverride
from onyx.llm.override_models import PromptOverride
from onyx.tools.models import ToolCallFinalResult

if TYPE_CHECKING:
    pass


class SourceTag(Tag):
    """
    文档来源标签模型，继承自基础Tag类。
    定义了文档的来源信息。
    """
    source: DocumentSource


class TagResponse(BaseModel):
    """
    标签响应模型，包含文档来源标签列表。
    """
    tags: list[SourceTag]


class UpdateChatSessionThreadRequest(BaseModel):
    """
    更新聊天会话线程的请求模型。
    
    属性:
        chat_session_id: 聊天会话ID
        new_alternate_model: 新的替代模型名称
    """
    # 如果未指定，使用Onyx默认角色 / If not specified, use Onyx default persona
    chat_session_id: UUID
    new_alternate_model: str


class ChatSessionCreationRequest(BaseModel):
    """
    创建聊天会话的请求模型。
    
    属性:
        persona_id: 角色ID，默认为0
        description: 会话描述，可选
    """
    persona_id: int = 0
    description: str | None = None


class CreateChatSessionID(BaseModel):
    chat_session_id: UUID


class ChatFeedbackRequest(BaseModel):
    """
    聊天反馈请求模型。
    
    属性:
        chat_message_id: 聊天消息ID
        is_positive: 是否为正面反馈
        feedback_text: 反馈文本
        predefined_feedback: 预定义反馈
    """
    chat_message_id: int
    is_positive: bool | None = None
    feedback_text: str | None = None
    predefined_feedback: str | None = None

    @model_validator(mode="after")
    def check_is_positive_or_feedback_text(self) -> "ChatFeedbackRequest":
        """
        验证是否提供了正面反馈或反馈文本。
        
        返回:
            ChatFeedbackRequest: 验证后的请求对象
        """
        if self.is_positive is None and self.feedback_text is None:
            raise ValueError("Empty feedback received.")
        return self


"""
当前不同分支是通过更改搜索查询生成的

                 [空根消息] 这允许第一个消息也可以被分支
              /           |           \
[第一条消息] [第一条消息编辑1] [第一条消息编辑2]
       |                  |
[第二条消息]  [编辑1分支的第二条消息]
"""


class CreateChatMessageRequest(ChunkContext):
    """
    创建聊天消息的请求模型。
    在创建消息前，需要先创建聊天会话并获取ID。
    Before creating messages, be sure to create a chat_session and get an id
    
    属性:
        chat_session_id: 聊天会话ID
        parent_message_id: 父消息ID
        message: 新消息内容
        file_descriptors: 附加文件描述符列表
        ...
    """
    chat_session_id: UUID
    # 这是树中前一条消息的主键（唯一标识符）
    parent_message_id: int | None
    # 新消息内容
    message: str
    # 我们应该附加到此消息的文件
    file_descriptors: list[FileDescriptor]

    # 如果未提供提示，则使用聊天会话中最大的提示
    # 但实际上这应该明确指定，仅在简化的API中推断
    # 使用prompt_id 0使用系统默认提示，即问答
    prompt_id: int | None
    # 如果提供了search_doc_ids，则不使用检索选项
    search_doc_ids: list[int] | None
    retrieval_options: RetrievalDetails | None
    # 可通过API使用，但不推荐用于大多数流程
    rerank_settings: RerankingDetails | None = None
    # 允许调用者指定他们想要使用的确切搜索查询
    # 如果指定，将禁用查询重写
    query_override: str | None = None

    # 启用额外处理以确保我们使用给定的用户消息ID重新生成
    regenerate: bool | None = None

    # 允许调用者覆盖角色/提示
    # 这些不会在聊天线程详细信息中持久化
    llm_override: LLMOverride | None = None
    prompt_override: PromptOverride | None = None

    # 允许用户指定替代助手
    alternate_assistant_id: int | None = None

    # 这优先于prompt_override
    # 这不会是直接从API传递的类型
    persona_override_config: PersonaOverrideConfig | None = None

    # 用于种子聊天以启动AI答案的生成
    use_existing_user_message: bool = False

    # 用于“OpenAI Assistants API”
    existing_assistant_message_id: int | None = None

    # 强制LLM返回结构化响应，请参阅
    # https://platform.openai.com/docs/guides/structured-outputs/introduction
    structured_response_format: dict | None = None

    @model_validator(mode="after")
    def check_search_doc_ids_or_retrieval_options(self) -> "CreateChatMessageRequest":
        """
        验证搜索文档ID和检索选项的有效性。
        必须提供其中之一，但不能同时提供两者或都不提供。
        
        返回:
            CreateChatMessageRequest: 验证后的请求对象
        """
        if self.search_doc_ids is None and self.retrieval_options is None:
            raise ValueError(
                "必须提供search_doc_ids或retrieval_options之一，但不能同时提供两者或都不提供。"
                "Either search_doc_ids or retrieval_options must be provided, but not both or neither."
            )
        return self

    def model_dump(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        data = super().model_dump(*args, **kwargs)
        data["chat_session_id"] = str(data["chat_session_id"])
        return data


class ChatMessageIdentifier(BaseModel):
    """
    聊天消息标识符模型。
    
    属性:
        message_id: 消息ID
    """
    message_id: int


class ChatRenameRequest(BaseModel):
    """
    聊天会话重命名请求模型。
    
    属性:
        chat_session_id: 聊天会话ID
        name: 新名称，可选
    """
    chat_session_id: UUID
    name: str | None = None


class ChatSessionUpdateRequest(BaseModel):
    """
    聊天会话更新请求模型。
    
    属性:
        sharing_status: 会话共享状态
    """
    sharing_status: ChatSessionSharedStatus


class DeleteAllSessionsRequest(BaseModel):
    """
    删除所有会话请求模型。
    
    属性:
        session_type: 会话类型
    """
    session_type: SessionType


class RenameChatSessionResponse(BaseModel):
    """
    重命名聊天会话响应模型。
    
    属性:
        new_name: 新名称(仅在名称是自动生成时有用)
    """
    new_name: str


class ChatSessionDetails(BaseModel):
    """
    聊天会话详情模型。
    
    属性:
        id: 会话UUID
        name: 会话名称
        persona_id: 角色ID
        time_created: 创建时间
        shared_status: 共享状态
        folder_id: 文件夹ID
        current_alternate_model: 当前替代模型
    """
    id: UUID
    name: str | None
    persona_id: int | None = None
    time_created: str
    shared_status: ChatSessionSharedStatus
    folder_id: int | None = None
    current_alternate_model: str | None = None


class ChatSessionsResponse(BaseModel):
    """
    聊天会话列表响应模型。
    
    属性:
        sessions: 会话详情列表
    """
    sessions: list[ChatSessionDetails]


class SearchFeedbackRequest(BaseModel):
    """
    搜索反馈请求模型。
    
    属性:
        message_id: 消息ID
        document_id: 文档ID
        document_rank: 文档排名
        click: 是否点击
        search_feedback: 搜索反馈类型
    """
    message_id: int
    document_id: str
    document_rank: int
    click: bool
    search_feedback: SearchFeedbackType | None = None

    @model_validator(mode="after")
    def check_click_or_search_feedback(self) -> "SearchFeedbackRequest":
        click, feedback = self.click, self.search_feedback

        if click is False and feedback is None:
            raise ValueError("Empty feedback received.")
        return self


class ChatMessageDetail(BaseModel):
    """
    聊天消息详情模型。
    
    属性:
        message_id: 消息ID
        parent_message: 父消息ID
        latest_child_message: 最新子消息ID
        message: 消息内容
        rephrased_query: 重述的查询
        context_docs: 上下文文档
        message_type: 消息类型
        time_sent: 发送时间
        overridden_model: 覆盖的模型
        alternate_assistant_id: 替代助手ID
        chat_session_id: 聊天会话ID
        citations: 引用映射
        files: 文件列表
        tool_call: 工具调用结果
    """
    message_id: int
    parent_message: int | None = None
    latest_child_message: int | None = None
    message: str
    rephrased_query: str | None = None
    context_docs: RetrievalDocs | None = None
    message_type: MessageType
    time_sent: datetime
    overridden_model: str | None
    alternate_assistant_id: int | None = None
    # 字典映射引用编号到db_doc_id
    chat_session_id: UUID | None = None
    citations: dict[int, int] | None = None
    files: list[FileDescriptor]
    tool_call: ToolCallFinalResult | None

    def model_dump(self, *args: list, **kwargs: dict[str, Any]) -> dict[str, Any]:  # type: ignore
        initial_dict = super().model_dump(mode="json", *args, **kwargs)  # type: ignore
        initial_dict["time_sent"] = self.time_sent.isoformat()
        return initial_dict


class SearchSessionDetailResponse(BaseModel):
    """
    搜索会话详情响应模型。
    
    属性:
        search_session_id: 搜索会话ID
        description: 描述
        documents: 文档列表
        messages: 消息列表
    """
    search_session_id: UUID
    description: str | None
    documents: list[SearchDoc]
    messages: list[ChatMessageDetail]


class ChatSessionDetailResponse(BaseModel):
    """
    聊天会话详情响应模型。
    
    属性:
        chat_session_id: 聊天会话ID
        description: 描述
        persona_id: 角色ID
        persona_name: 角色名称
        persona_icon_color: 角色图标颜色
        persona_icon_shape: 角色图标形状
        messages: 消息列表
        time_created: 创建时间
        shared_status: 共享状态
        current_alternate_model: 当前替代模型
    """
    chat_session_id: UUID
    description: str | None
    persona_id: int | None = None
    persona_name: str | None
    persona_icon_color: str | None
    persona_icon_shape: int | None
    messages: list[ChatMessageDetail]
    time_created: datetime
    shared_status: ChatSessionSharedStatus
    current_alternate_model: str | None


# 这个不再使用
class QueryValidationResponse(BaseModel):
    reasoning: str
    answerable: bool


class AdminSearchRequest(BaseModel):
    """
    管理员搜索请求模型。
    
    属性:
        query: 查询字符串
        filters: 基础过滤条件
    """
    query: str
    filters: BaseFilters


class AdminSearchResponse(BaseModel):
    """
    管理员搜索响应模型。
    
    属性:
        documents: 搜索到的文档列表
    """
    documents: list[SearchDoc]
