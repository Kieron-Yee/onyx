"""
This module implements the prompt building functionality for chat interactions.
It provides classes and functions to construct prompts for LLM interactions,
handling message history, system prompts, and tool integrations.

本模块实现了聊天交互的提示构建功能。
它提供了用于构建LLM交互提示的类和函数，处理消息历史、系统提示和工具集成。
"""

from collections.abc import Callable
from typing import cast

from langchain_core.messages import BaseMessage
from langchain_core.messages import HumanMessage
from langchain_core.messages import SystemMessage
from pydantic.v1 import BaseModel as BaseModel__v1

from onyx.chat.models import PromptConfig
from onyx.chat.prompt_builder.citations_prompt import compute_max_llm_input_tokens
from onyx.chat.prompt_builder.utils import translate_history_to_basemessages
from onyx.file_store.models import InMemoryChatFile
from onyx.llm.interfaces import LLMConfig
from onyx.llm.models import PreviousMessage
from onyx.llm.utils import build_content_with_imgs
from onyx.llm.utils import check_message_tokens
from onyx.llm.utils import message_to_prompt_and_imgs
from onyx.natural_language_processing.utils import get_tokenizer
from onyx.prompts.chat_prompts import CHAT_USER_CONTEXT_FREE_PROMPT
from onyx.prompts.prompt_utils import add_date_time_to_prompt
from onyx.prompts.prompt_utils import drop_messages_history_overflow
from onyx.tools.force import ForceUseTool
from onyx.tools.models import ToolCallFinalResult
from onyx.tools.models import ToolCallKickoff
from onyx.tools.models import ToolResponse
from onyx.tools.tool import Tool


def default_build_system_message(
    prompt_config: PromptConfig,
) -> SystemMessage | None:
    """
    Build a system message based on the provided prompt configuration.
    
    Args:
        prompt_config (PromptConfig): Configuration for the prompt including system prompt and settings
        
    Returns:
        SystemMessage | None: Constructed system message or None if no system prompt is provided
    
    构建基于提供的提示配置的系统消息。
    
    参数:
        prompt_config: 包含系统提示和设置的提示配置
        
    返回:
        SystemMessage | None: 构建的系统消息，如果没有系统提示则返回None
    """
    system_prompt = prompt_config.system_prompt.strip()
    if prompt_config.datetime_aware:
        system_prompt = add_date_time_to_prompt(prompt_str=system_prompt)

    if not system_prompt:
        return None

    system_msg = SystemMessage(content=system_prompt)

    return system_msg


def default_build_user_message(
    user_query: str, prompt_config: PromptConfig, files: list[InMemoryChatFile] = []
) -> HumanMessage:
    """
    Build a user message incorporating the user query and any task-specific prompts.
    
    Args:
        user_query (str): The user's input query
        prompt_config (PromptConfig): Configuration for the prompt
        files (list[InMemoryChatFile]): List of files to be included in the message
    
    Returns:
        HumanMessage: Constructed user message
        
    构建包含用户查询和任务特定提示的用户消息。
    
    参数:
        user_query: 用户输入的查询
        prompt_config: 提示配置
        files: 要包含在消息中的文件列表
        
    返回:
        HumanMessage: 构建的用户消息
    """
    user_prompt = (
        CHAT_USER_CONTEXT_FREE_PROMPT.format(
            task_prompt=prompt_config.task_prompt, user_query=user_query
        )
        if prompt_config.task_prompt
        else user_query
    )
    user_prompt = user_prompt.strip()
    user_msg = HumanMessage(
        content=build_content_with_imgs(user_prompt, files) if files else user_prompt
    )
    return user_msg


class AnswerPromptBuilder:
    """
    A builder class for constructing complete prompt sequences for LLM interactions.
    
    提示构建器类，用于构建LLM交互的完整提示序列。
    """
    
    def __init__(
        self,
        user_message: HumanMessage,
        message_history: list[PreviousMessage],
        llm_config: LLMConfig,
        raw_user_text: str,
        single_message_history: str | None = None,
    ) -> None:
        """
        Initialize the prompt builder with necessary components.
        
        Args:
            user_message: The current user message
            message_history: Previous message history
            llm_config: LLM configuration settings
            raw_user_text: Original user input text
            single_message_history: Condensed chat history (optional)
            
        初始化提示构建器及其必要组件。
        
        参数:
            user_message: 当前用户消息
            message_history: 之前的消息历史
            llm_config: LLM配置设置
            raw_user_text: 原始用户输入文本
            single_message_history: 压缩的聊天历史（可选）
        """
        self.max_tokens = compute_max_llm_input_tokens(llm_config)

        llm_tokenizer = get_tokenizer(
            provider_type=llm_config.model_provider,
            model_name=llm_config.model_name,
        )
        self.llm_tokenizer_encode_func = cast(
            Callable[[str], list[int]], llm_tokenizer.encode
        )

        self.raw_message_history = message_history
        (
            self.message_history,
            self.history_token_cnts,
        ) = translate_history_to_basemessages(message_history)

        # for cases where like the QA flow where we want to condense the chat history
        # into a single message rather than a sequence of User / Assistant messages
        self.single_message_history = single_message_history

        self.system_message_and_token_cnt: tuple[SystemMessage, int] | None = None
        self.user_message_and_token_cnt = (
            user_message,
            check_message_tokens(user_message, self.llm_tokenizer_encode_func),
        )

        self.new_messages_and_token_cnts: list[tuple[BaseMessage, int]] = []

        self.raw_user_message = raw_user_text

    def update_system_prompt(self, system_message: SystemMessage | None) -> None:
        """
        Update the system prompt and recalculate token count.
        
        Args:
            system_message: New system message to set
            
        更新系统提示并重新计算令牌数。
        
        参数:
            system_message: 要设置的新系统消息
        """
        if not system_message:
            self.system_message_and_token_cnt = None
            return

        self.system_message_and_token_cnt = (
            system_message,
            check_message_tokens(system_message, self.llm_tokenizer_encode_func),
        )

    def update_user_prompt(self, user_message: HumanMessage) -> None:
        """
        Update the user prompt and recalculate token count.
        
        Args:
            user_message: New user message to set
            
        更新用户提示并重新计算令牌数。
        
        参数:
            user_message: 要设置的新用户消息
        """
        self.user_message_and_token_cnt = (
            user_message,
            check_message_tokens(user_message, self.llm_tokenizer_encode_func),
        )

    def append_message(self, message: BaseMessage) -> None:
        """
        Append a new message to the message history and calculate its tokens.
        
        Args:
            message: Message to append
            
        将新消息追加到消息历史并计算其令牌数。
        
        参数:
            message: 要追加的消息
        """
        token_count = check_message_tokens(message, self.llm_tokenizer_encode_func)
        self.new_messages_and_token_cnts.append((message, token_count))

    def get_user_message_content(self) -> str:
        """
        Get the content of the current user message.
        
        Returns:
            str: Content of the user message
            
        获取当前用户消息的内容。
        
        返回:
            str: 用户消息的内容
        """
        query, _ = message_to_prompt_and_imgs(self.user_message_and_token_cnt[0])
        return query

    def build(self) -> list[BaseMessage]:
        """
        Build the final sequence of messages for the LLM prompt.
        
        Returns:
            list[BaseMessage]: Final sequence of messages
            
        构建LLM提示的最终消息序列。
        
        返回:
            list[BaseMessage]: 最终的消息序列
        """
        if not self.user_message_and_token_cnt:
            raise ValueError("User message must be set before building prompt")

        final_messages_with_tokens: list[tuple[BaseMessage, int]] = []
        if self.system_message_and_token_cnt:
            final_messages_with_tokens.append(self.system_message_and_token_cnt)

        final_messages_with_tokens.extend(
            [
                (self.message_history[i], self.history_token_cnts[i])
                for i in range(len(self.message_history))
            ]
        )

        final_messages_with_tokens.append(self.user_message_and_token_cnt)

        if self.new_messages_and_token_cnts:
            final_messages_with_tokens.extend(self.new_messages_and_token_cnts)

        return drop_messages_history_overflow(
            final_messages_with_tokens, self.max_tokens
        )


class LLMCall(BaseModel__v1):
    """
    Model representing an LLM interaction call with all necessary components.
    
    Attributes:
        prompt_builder: Builder for constructing prompts
        tools: List of available tools
        force_use_tool: Tool forcing mechanism
        files: List of files involved in the interaction
        tool_call_info: Information about tool calls
        using_tool_calling_llm: Flag indicating if using tool-calling LLM
        
    表示LLM交互调用的模型，包含所有必要组件。
    
    属性:
        prompt_builder: 用于构建提示的构建器
        tools: 可用工具列表
        force_use_tool: 工具强制使用机制
        files: 交互中涉及的文件列表
        tool_call_info: 工具调用信息
        using_tool_calling_llm: 指示是否使用工具调用LLM的标志
    """
    prompt_builder: AnswerPromptBuilder
    tools: list[Tool]
    force_use_tool: ForceUseTool
    files: list[InMemoryChatFile]
    tool_call_info: list[ToolCallKickoff | ToolResponse | ToolCallFinalResult]
    using_tool_calling_llm: bool

    class Config:
        arbitrary_types_allowed = True
