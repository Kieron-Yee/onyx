"""
此文件实现了一个通用的大语言模型(LLM)接口封装,主要功能包括:
1. 提供统一的LLM调用接口,支持多种LLM服务提供商
2. 处理消息转换、流式响应等功能
3. 实现错误处理和日志记录
4. 支持工具调用(Tool Calls)功能

主要基于litellm库实现,可以方便地配置和使用各种LLM服务。
"""

import json
import os
import traceback
from collections.abc import Iterator
from collections.abc import Sequence
from typing import Any
from typing import cast

import litellm  # type: ignore
from httpx import RemoteProtocolError
from langchain.schema.language_model import LanguageModelInput
from langchain_core.messages import AIMessage
from langchain_core.messages import AIMessageChunk
from langchain_core.messages import BaseMessage
from langchain_core.messages import BaseMessageChunk
from langchain_core.messages import ChatMessage
from langchain_core.messages import ChatMessageChunk
from langchain_core.messages import FunctionMessage
from langchain_core.messages import FunctionMessageChunk
from langchain_core.messages import HumanMessage
from langchain_core.messages import HumanMessageChunk
from langchain_core.messages import SystemMessage
from langchain_core.messages import SystemMessageChunk
from langchain_core.messages.tool import ToolCallChunk
from langchain_core.messages.tool import ToolMessage
from langchain_core.prompt_values import PromptValue

from onyx.configs.app_configs import LOG_DANSWER_MODEL_INTERACTIONS
from onyx.configs.model_configs import (
    DISABLE_LITELLM_STREAMING,
)
from onyx.configs.model_configs import GEN_AI_TEMPERATURE
from onyx.configs.model_configs import LITELLM_EXTRA_BODY
from onyx.llm.interfaces import LLM
from onyx.llm.interfaces import LLMConfig
from onyx.llm.interfaces import ToolChoiceOptions
from onyx.server.utils import mask_string
from onyx.utils.logger import setup_logger
from onyx.utils.long_term_log import LongTermLogger


logger = setup_logger()

# If a user configures a different model and it doesn't support all the same
# parameters like frequency and presence, just ignore them
litellm.drop_params = True
litellm.telemetry = False

_LLM_PROMPT_LONG_TERM_LOG_CATEGORY = "llm_prompt"


def _base_msg_to_role(msg: BaseMessage) -> str:
    """
    将BaseMessage对象转换为对应的角色字符串
    
    参数:
        msg: BaseMessage - 需要转换的消息对象
    
    返回:
        str - 对应的角色字符串(user/assistant/system/function/unknown)
    """
    if isinstance(msg, HumanMessage) or isinstance(msg, HumanMessageChunk):
        return "user"
    if isinstance(msg, AIMessage) or isinstance(msg, AIMessageChunk):
        return "assistant"
    if isinstance(msg, SystemMessage) or isinstance(msg, SystemMessageChunk):
        return "system"
    if isinstance(msg, FunctionMessage) or isinstance(msg, FunctionMessageChunk):
        return "function"
    return "unknown"


def _convert_litellm_message_to_langchain_message(
    litellm_message: litellm.Message,
) -> BaseMessage:
    """
    将litellm的消息格式转换为langchain的消息格式
    
    参数:
        litellm_message: 来自litellm的消息对象
        
    返回:
        BaseMessage: 转换后的langchain消息对象
        
    抛出:
        ValueError: 当收到未知的角色类型时
    """
    # Extracting the basic attributes from the litellm message
    content = litellm_message.content or ""
    role = litellm_message.role

    # Handling function calls and tool calls if present
    tool_calls = (
        cast(
            list[litellm.ChatCompletionMessageToolCall],
            litellm_message.tool_calls,
        )
        if hasattr(litellm_message, "tool_calls")
        else []
    )

    # Create the appropriate langchain message based on the role
    if role == "user":
        return HumanMessage(content=content)
    elif role == "assistant":
        return AIMessage(
            content=content,
            tool_calls=[
                {
                    "name": tool_call.function.name or "",
                    "args": json.loads(tool_call.function.arguments),
                    "id": tool_call.id,
                }
                for tool_call in tool_calls
            ]
            if tool_calls
            else [],
        )
    elif role == "system":
        return SystemMessage(content=content)
    else:
        raise ValueError(f"Unknown role type received: {role}")


def _convert_message_to_dict(message: BaseMessage) -> dict:
    """
    将langchain的消息对象转换为字典格式
    
    # Adapted from langchain_community.chat_models.litellm._convert_message_to_dict
    # 改编自langchain_community.chat_models.litellm._convert_message_to_dict
    
    参数:
        message: BaseMessage - 需要转换的langchain消息对象
    
    返回:
        dict - 转换后的消息字典
        
    抛出:
        ValueError: 当收到未知的消息类型时
    """
    if isinstance(message, ChatMessage):
        message_dict = {"role": message.role, "content": message.content}
    elif isinstance(message, HumanMessage):
        message_dict = {"role": "user", "content": message.content}
    elif isinstance(message, AIMessage):
        message_dict = {"role": "assistant", "content": message.content}
        if message.tool_calls:
            message_dict["tool_calls"] = [
                {
                    "id": tool_call.get("id"),
                    "function": {
                        "name": tool_call["name"],
                        "arguments": json.dumps(tool_call["args"]),
                    },
                    "type": "function",
                    "index": tool_call.get("index", 0),
                }
                for tool_call in message.tool_calls
            ]
        if "function_call" in message.additional_kwargs:
            message_dict["function_call"] = message.additional_kwargs["function_call"]
    elif isinstance(message, SystemMessage):
        message_dict = {"role": "system", "content": message.content}
    elif isinstance(message, FunctionMessage):
        message_dict = {
            "role": "function",
            "content": message.content,
            "name": message.name,
        }
    elif isinstance(message, ToolMessage):
        message_dict = {
            "tool_call_id": message.tool_call_id,
            "role": "tool",
            "name": message.name or "",
            "content": message.content,
        }
    else:
        raise ValueError(f"Got unknown type {message}")
    if "name" in message.additional_kwargs:
        message_dict["name"] = message.additional_kwargs["name"]
    return message_dict


def _convert_delta_to_message_chunk(
    _dict: dict[str, Any],
    curr_msg: BaseMessage | None,
    stop_reason: str | None = None,
) -> BaseMessageChunk:
    """
    将增量消息转换为消息块
    
    # Adapted from langchain_community.chat_models.litellm._convert_delta_to_message_chunk
    # 改编自langchain_community.chat_models.litellm._convert_delta_to_message_chunk
    
    参数:
        _dict: dict - 增量消息字典
        curr_msg: BaseMessage | None - 当前消息对象
        stop_reason: str | None - 停止原因
    
    返回:
        BaseMessageChunk - 转换后的消息块
        
    抛出:
        ValueError: 当收到未知的角色类型时
    """
    role = _dict.get("role") or (_base_msg_to_role(curr_msg) if curr_msg else None)
    content = _dict.get("content") or ""
    additional_kwargs = {}
    if _dict.get("function_call"):
        additional_kwargs.update({"function_call": dict(_dict["function_call"])})
    tool_calls = cast(
        list[litellm.utils.ChatCompletionDeltaToolCall] | None, _dict.get("tool_calls")
    )

    if role == "user":
        return HumanMessageChunk(content=content)
    # NOTE: if tool calls are present, then it's an assistant.
    # In Ollama, the role will be None for tool-calls
    elif role == "assistant" or tool_calls:
        if tool_calls:
            tool_call = tool_calls[0]
            tool_name = tool_call.function.name or (curr_msg and curr_msg.name) or ""
            idx = tool_call.index

            tool_call_chunk = ToolCallChunk(
                name=tool_name,
                id=tool_call.id,
                args=tool_call.function.arguments,
                index=idx,
            )

            return AIMessageChunk(
                content=content,
                tool_call_chunks=[tool_call_chunk],
                additional_kwargs={
                    "usage_metadata": {"stop": stop_reason},
                    **additional_kwargs,
                },
            )

        return AIMessageChunk(
            content=content,
            additional_kwargs={
                "usage_metadata": {"stop": stop_reason},
                **additional_kwargs,
            },
        )
    elif role == "system":
        return SystemMessageChunk(content=content)
    elif role == "function":
        return FunctionMessageChunk(content=content, name=_dict["name"])
    elif role:
        return ChatMessageChunk(content=content, role=role)

    raise ValueError(f"Unknown role: {role}")


def _prompt_to_dict(
    prompt: LanguageModelInput,
) -> Sequence[str | list[str] | dict[str, Any] | tuple[str, str]]:
    """
    将提示输入转换为字典序列
    
    参数:
        prompt: LanguageModelInput - 输入提示
        
    返回:
        Sequence - 转换后的字典序列
    """
    # NOTE: this must go first, since it is also a Sequence
    # 注意：这必须放在首位，因为它也是一个Sequence
    if isinstance(prompt, str):
        return [_convert_message_to_dict(HumanMessage(content=prompt))]

    if isinstance(prompt, (list, Sequence)):
        return [
            _convert_message_to_dict(msg) if isinstance(msg, BaseMessage) else msg
            for msg in prompt
        ]

    if isinstance(prompt, PromptValue):
        return [_convert_message_to_dict(message) for message in prompt.to_messages()]


class DefaultMultiLLM(LLM):
    """
    基于Litellm库实现的通用LLM接口类
    
    支持多种LLM提供商的统一接口实现,包括:
    - 消息处理和转换
    - 流式响应
    - 错误处理
    - 日志记录
    - 工具调用
    """

    def __init__(
        self,
        api_key: str | None,
        timeout: int,
        model_provider: str,
        model_name: str,
        api_base: str | None = None,
        api_version: str | None = None,
        deployment_name: str | None = None,
        max_output_tokens: int | None = None,
        custom_llm_provider: str | None = None,
        temperature: float = GEN_AI_TEMPERATURE,
        custom_config: dict[str, str] | None = None,
        extra_headers: dict[str, str] | None = None,
        extra_body: dict | None = LITELLM_EXTRA_BODY,
        model_kwargs: dict[str, Any] | None = None,
        long_term_logger: LongTermLogger | None = None,
    ):
        """
        初始化LLM接口
        
        参数:
            api_key: API密钥
            timeout: 超时时间(秒)
            model_provider: 模型提供商
            model_name: 模型名称
            api_base: API基础URL
            api_version: API版本
            deployment_name: 部署名称
            max_output_tokens: 最大输出token数
            custom_llm_provider: 自定义LLM提供商
            temperature: 采样温度
            custom_config: 自定义配置
            extra_headers: 额外的HTTP头
            extra_body: 额外的请求体参数
            model_kwargs: 模型特定的参数
            long_term_logger: 长期日志记录器
        """
        self._timeout = timeout
        self._model_provider = model_provider
        self._model_version = model_name
        self._temperature = temperature
        self._api_key = api_key
        self._deployment_name = deployment_name
        self._api_base = api_base
        self._api_version = api_version
        self._custom_llm_provider = custom_llm_provider
        self._long_term_logger = long_term_logger

        # This can be used to store the maximum output tokens for this model.
        # self._max_output_tokens = (
        #     max_output_tokens
        #     if max_output_tokens is not None
        #     else get_llm_max_output_tokens(
        #         model_map=litellm.model_cost,
        #         model_name=model_name,
        #         model_provider=model_provider,
        #     )
        # )
        self._custom_config = custom_config

        # Create a dictionary for model-specific arguments if it's None
        # 如果model_kwargs为None，则创建一个用于模型特定参数的字典
        model_kwargs = model_kwargs or {}

        # NOTE: have to set these as environment variables for Litellm since
        # not all are able to passed in but they always support them set as env
        # variables. We'll also try passing them in, since litellm just ignores
        # addtional kwargs (and some kwargs MUST be passed in rather than set as
        # env variables)
        # 注意：必须将这些设置为Litellm的环境变量，因为不是所有参数都能直接传入，
        # 但它们都支持通过环境变量设置。我们也会尝试直接传入这些参数，因为litellm会忽略
        # 额外的kwargs（有些kwargs必须传入而不是设置为环境变量）
        if custom_config:
            # Specifically pass in "vertex_credentials" as a model_kwarg to the
            # completion call for vertex AI. More details here:
            # https://docs.litellm.ai/docs/providers/vertex
            # 专门将"vertex_credentials"作为model_kwarg传递给vertex AI的completion调用。
            # 更多详情请参见：https://docs.litellm.ai/docs/providers/vertex
            vertex_credentials_key = "vertex_credentials"
            vertex_credentials = custom_config.get(vertex_credentials_key)
            if vertex_credentials and model_provider == "vertex_ai":
                model_kwargs[vertex_credentials_key] = vertex_credentials
            else:
                # standard case
                # 标准情况
                for k, v in custom_config.items():
                    os.environ[k] = v

        if extra_headers:
            model_kwargs.update({"extra_headers": extra_headers})
        if extra_body:
            model_kwargs.update({"extra_body": extra_body})

        self._model_kwargs = model_kwargs

    def log_model_configs(self) -> None:
        """
        记录模型配置到日志
        """
        logger.debug(f"Config: {self.config}")

    def _safe_model_config(self) -> dict:
        """
        返回安全的模型配置(隐藏敏感信息)
        
        返回:
            dict - 处理后的配置字典
        """
        dump = self.config.model_dump()
        dump["api_key"] = mask_string(dump.get("api_key", ""))
        return dump

    def _record_call(self, prompt: LanguageModelInput) -> None:
        if self._long_term_logger:
            self._long_term_logger.record(
                {"prompt": _prompt_to_dict(prompt), "model": self._safe_model_config()},
                category=_LLM_PROMPT_LONG_TERM_LOG_CATEGORY,
            )

    def _record_result(
        self, prompt: LanguageModelInput, model_output: BaseMessage
    ) -> None:
        if self._long_term_logger:
            self._long_term_logger.record(
                {
                    "prompt": _prompt_to_dict(prompt),
                    "content": model_output.content,
                    "tool_calls": (
                        model_output.tool_calls
                        if hasattr(model_output, "tool_calls")
                        else []
                    ),
                    "model": self._safe_model_config(),
                },
                category=_LLM_PROMPT_LONG_TERM_LOG_CATEGORY,
            )

    def _record_error(self, prompt: LanguageModelInput, error: Exception) -> None:
        if self._long_term_logger:
            self._long_term_logger.record(
                {
                    "prompt": _prompt_to_dict(prompt),
                    "error": str(error),
                    "traceback": "".join(
                        traceback.format_exception(
                            type(error), error, error.__traceback__
                        )
                    ),
                    "model": self._safe_model_config(),
                },
                category=_LLM_PROMPT_LONG_TERM_LOG_CATEGORY,
            )

    # def _calculate_max_output_tokens(self, prompt: LanguageModelInput) -> int:
    #     # NOTE: This method can be used for calculating the maximum tokens for the stream,
    #     # but it isn't used in practice due to the computational cost of counting tokens
    #     # and because LLM providers automatically cut off at the maximum output.
    #     # The implementation is kept for potential future use or debugging purposes.

    #     # Get max input tokens for the model
    #     max_context_tokens = get_max_input_tokens(
    #         model_name=self.config.model_name, model_provider=self.config.model_provider
    #     )

    #     llm_tokenizer = get_tokenizer(
    #         model_name=self.config.model_name,
    #         provider_type=self.config.model_provider,
    #     )
    #     # Calculate tokens in the input prompt
    #     input_tokens = sum(len(llm_tokenizer.encode(str(m))) for m in prompt)

    #     # Calculate available tokens for output
    #     available_output_tokens = max_context_tokens - input_tokens

    #     # Return the lesser of available tokens or configured max
    #     return min(self._max_output_tokens, available_output_tokens)

    def _completion(
        self,
        prompt: LanguageModelInput,
        tools: list[dict] | None,
        tool_choice: ToolChoiceOptions | None,
        stream: bool,
        structured_response_format: dict | None = None,
    ) -> litellm.ModelResponse | litellm.CustomStreamWrapper:
        """
        执行LLM完成请求
        
        参数:
            prompt: 输入提示
            tools: 可用工具列表
            tool_choice: 工具选择选项
            stream: 是否使用流式响应
            structured_response_format: 结构化响应格式
            
        返回:
            litellm.ModelResponse | litellm.CustomStreamWrapper: 模型响应
            
        抛出:
            Exception: 当调用失败时记录并重新抛出异常
        """
        # litellm doesn't accept LangChain BaseMessage objects, so we need to convert them
        # to a dict representation
        # litellm不接受LangChain BaseMessage对象，所以需要将其转换为字典表示
        processed_prompt = _prompt_to_dict(prompt)
        self._record_call(processed_prompt)

        try:
            return litellm.completion(
                # model choice
                # 模型选择
                model=f"{self.config.model_provider}/{self.config.deployment_name or self.config.model_name}",
                # NOTE: have to pass in None instead of empty string for these
                # otherwise litellm can have some issues with bedrock
                # 注意：这些参数必须传入None而不是空字符串，否则litellm在bedrock上可能会有问题
                api_key=self._api_key or None,
                base_url=self._api_base or None,
                api_version=self._api_version or None,
                custom_llm_provider=self._custom_llm_provider or None,
                # actual input
                # 实际输入
                messages=processed_prompt,
                tools=tools,
                tool_choice=tool_choice if tools else None,
                # streaming choice
                # 流式选择
                stream=stream,
                # model params
                # 模型参数
                temperature=self._temperature,
                timeout=self._timeout,
                # For now, we don't support parallel tool calls
                # NOTE: we can't pass this in if tools are not specified
                # or else OpenAI throws an error
                # 当前不支持并行工具调用
                # 注意：如果没有指定tools，就不能传入这个参数，否则OpenAI会抛出错误
                **({"parallel_tool_calls": False} if tools else {}),
                **(
                    {"response_format": structured_response_format}
                    if structured_response_format
                    else {}
                ),
                **self._model_kwargs,
            )
        except Exception as e:
            self._record_error(processed_prompt, e)
            # for break pointing
            # 用于断点调试
            raise e

    @property
    def config(self) -> LLMConfig:
        """
        获取当前LLM配置
        
        返回:
            LLMConfig: 当前配置信息
        """
        return LLMConfig(
            model_provider=self._model_provider,
            model_name=self._model_version,
            temperature=self._temperature,
            api_key=self._api_key,
            api_base=self._api_base,
            api_version=self._api_version,
            deployment_name=self._deployment_name,
        )

    def _invoke_implementation(
        self,
        prompt: LanguageModelInput,
        tools: list[dict] | None = None,
        tool_choice: ToolChoiceOptions | None = None,
        structured_response_format: dict | None = None,
    ) -> BaseMessage:
        """
        实现LLM的直接调用
        
        参数:
            prompt: 输入提示
            tools: 可用工具列表
            tool_choice: 工具选择选项
            structured_response_format: 结构化响应格式
            
        返回:
            BaseMessage: 模型响应消息
            
        抛出:
            ValueError: 当响应格式不符合预期时
        """
        if LOG_DANSWER_MODEL_INTERACTIONS:
            self.log_model_configs()

        response = cast(
            litellm.ModelResponse,
            self._completion(
                prompt, tools, tool_choice, False, structured_response_format
            ),
        )
        choice = response.choices[0]
        if hasattr(choice, "message"):
            output = _convert_litellm_message_to_langchain_message(choice.message)
            if output:
                self._record_result(prompt, output)
            return output
        else:
            raise ValueError("Unexpected response choice type")

    def _stream_implementation(
        self,
        prompt: LanguageModelInput,
        tools: list[dict] | None = None,
        tool_choice: ToolChoiceOptions | None = None,
        structured_response_format: dict | None = None,
    ) -> Iterator[BaseMessage]:
        """
        实现LLM的流式调用
        
        参数:
            prompt: 输入提示
            tools: 可用工具列表
            tool_choice: 工具选择选项
            structured_response_format: 结构化响应格式
            
        返回:
            Iterator[BaseMessage]: 消息流迭代器
            
        抛出:
            RuntimeError: 当AI模型在生成过程中失败时
        """
        if LOG_DANSWER_MODEL_INTERACTIONS:
            self.log_model_configs()

        if (
            DISABLE_LITELLM_STREAMING or self.config.model_name == "o1-2024-12-17"
        ):  # TODO: remove once litellm supports streaming
            yield self.invoke(prompt, tools, tool_choice, structured_response_format)
            return

        output = None
        response = cast(
            litellm.CustomStreamWrapper,
            self._completion(
                prompt, tools, tool_choice, True, structured_response_format
            ),
        )
        try:
            for part in response:
                if not part["choices"]:
                    continue

                choice = part["choices"][0]
                message_chunk = _convert_delta_to_message_chunk(
                    choice["delta"],
                    output,
                    stop_reason=choice["finish_reason"],
                )

                if output is None:
                    output = message_chunk
                else:
                    output += message_chunk

                yield message_chunk

        except RemoteProtocolError:
            raise RuntimeError(
                "The AI model failed partway through generation, please try again."
            )

        if output:
            self._record_result(prompt, output)

        if LOG_DANSWER_MODEL_INTERACTIONS and output:
            content = output.content or ""
            if isinstance(output, AIMessage):
                if content:
                    log_msg = content
                elif output.tool_calls:
                    log_msg = "Tool Calls: " + str(
                        [
                            {
                                key: value
                                for key, value in tool_call.items()
                                if key != "index"
                            }
                            for tool_call in output.tool_calls
                        ]
                    )
                else:
                    log_msg = ""
                logger.debug(f"Raw Model Output:\n{log_msg}")
            else:
                logger.debug(f"Raw Model Output:\n{content}")
