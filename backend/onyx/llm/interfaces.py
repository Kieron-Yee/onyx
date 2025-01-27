"""
此文件定义了与语言模型(LLM)交互的接口和基础类。
主要包含：
- LLM配置类(LLMConfig)
- LLM基础抽象类
- 日志记录功能
- 类型定义
"""

import abc
from collections.abc import Iterator
from typing import Literal

from langchain.schema.language_model import LanguageModelInput
from langchain_core.messages import AIMessageChunk
from langchain_core.messages import BaseMessage
from pydantic import BaseModel

from onyx.configs.app_configs import DISABLE_GENERATIVE_AI
from onyx.configs.app_configs import LOG_DANSWER_MODEL_INTERACTIONS
from onyx.configs.app_configs import LOG_INDIVIDUAL_MODEL_TOKENS
from onyx.utils.logger import setup_logger


logger = setup_logger()

# 定义工具选择选项的字面量类型
ToolChoiceOptions = Literal["required"] | Literal["auto"] | Literal["none"]


class LLMConfig(BaseModel):
    """
    LLM配置模型类，用于存储语言模型的配置信息
    
    属性说明：
    model_provider: 模型提供商
    model_name: 模型名称
    temperature: 温度参数
    api_key: API密钥(可选)
    api_base: API基础URL(可选)
    api_version: API版本(可选)
    deployment_name: 部署名称(可选)
    """
    model_provider: str
    model_name: str
    temperature: float
    api_key: str | None = None
    api_base: str | None = None
    api_version: str | None = None
    deployment_name: str | None = None
    # This disables the "model_" protected namespace for pydantic
    # 这里禁用pydantic的"model_"保护命名空间
    model_config = {"protected_namespaces": ()}


def log_prompt(prompt: LanguageModelInput) -> None:
    """
    记录模型输入提示的日志函数
    
    参数：
    prompt: 语言模型的输入，可以是字符串或消息列表
    """
    if isinstance(prompt, list):
        for ind, msg in enumerate(prompt):
            if isinstance(msg, AIMessageChunk):
                if msg.content:
                    log_msg = msg.content
                elif msg.tool_call_chunks:
                    log_msg = "Tool Calls: " + str(
                        [
                            {
                                key: value
                                for key, value in tool_call.items()
                                if key != "index"
                            }
                            for tool_call in msg.tool_call_chunks
                        ]
                    )
                else:
                    log_msg = ""
                logger.debug(f"Message {ind}:\n{log_msg}")
            else:
                logger.debug(f"Message {ind}:\n{msg.content}")
    if isinstance(prompt, str):
        logger.debug(f"Prompt:\n{prompt}")


class LLM(abc.ABC):
    """
    Mimics the LangChain LLM / BaseChatModel interfaces to make it easy
    to use these implementations to connect to a variety of LLM providers.
    模仿LangChain LLM/BaseChatModel接口，使其易于连接到各种LLM提供商
    """

    @property
    def requires_warm_up(self) -> bool:
        """
        Is this model running in memory and needs an initial call to warm it up?
        判断该模型是否需要预热
        
        返回：
        bool: 是否需要预热
        """
        return False

    @property
    def requires_api_key(self) -> bool:
        """
        判断该模型是否需要API密钥
        
        返回：
        bool: 是否需要API密钥
        """
        return True

    @property
    @abc.abstractmethod
    def config(self) -> LLMConfig:
        """
        获取LLM配置信息的抽象方法
        
        返回：
        LLMConfig: LLM配置对象
        """
        raise NotImplementedError

    @abc.abstractmethod
    def log_model_configs(self) -> None:
        """
        记录模型配置信息的抽象方法
        """
        raise NotImplementedError

    def _precall(self, prompt: LanguageModelInput) -> None:
        """
        调用模型前的预处理方法
        
        参数：
        prompt: 模型输入
        """
        if DISABLE_GENERATIVE_AI:
            raise Exception("Generative AI is disabled")
        if LOG_DANSWER_MODEL_INTERACTIONS:
            log_prompt(prompt)

    def invoke(
        self,
        prompt: LanguageModelInput,
        tools: list[dict] | None = None,
        tool_choice: ToolChoiceOptions | None = None,
        structured_response_format: dict | None = None,
    ) -> BaseMessage:
        """
        调用模型进行推理的方法
        
        参数：
        prompt: 模型输入
        tools: 可用工具列表
        tool_choice: 工具选择选项
        structured_response_format: 结构化响应格式
        
        返回：
        BaseMessage: 模型响应消息
        """
        self._precall(prompt)
        # TODO add a postcall to log model outputs independent of concrete class implementation
        # TODO 添加后处理来记录模型输出，独立于具体类实现
        return self._invoke_implementation(
            prompt, tools, tool_choice, structured_response_format
        )

    @abc.abstractmethod
    def _invoke_implementation(
        self,
        prompt: LanguageModelInput,
        tools: list[dict] | None = None,
        tool_choice: ToolChoiceOptions | None = None,
        structured_response_format: dict | None = None,
    ) -> BaseMessage:
        """
        具体的模型调用实现的抽象方法
        
        参数：
        prompt: 模型输入
        tools: 可用工具列表
        tool_choice: 工具选择选项
        structured_response_format: 结构化响应格式
        
        返回：
        BaseMessage: 模型响应消息
        """
        raise NotImplementedError

    def stream(
        self,
        prompt: LanguageModelInput,
        tools: list[dict] | None = None,
        tool_choice: ToolChoiceOptions | None = None,
        structured_response_format: dict | None = None,
    ) -> Iterator[BaseMessage]:
        """
        流式调用模型的方法
        
        参数：
        prompt: 模型输入
        tools: 可用工具列表
        tool_choice: 工具选择选项
        structured_response_format: 结构化响应格式
        
        返回：
        Iterator[BaseMessage]: 模型响应消息的迭代器
        """
        self._precall(prompt)
        # TODO add a postcall to log model outputs independent of concrete class implementation
        # TODO 添加后处理来记录模型输出，独立于具体类实现
        messages = self._stream_implementation(
            prompt, tools, tool_choice, structured_response_format
        )

        tokens = []
        for message in messages:
            if LOG_INDIVIDUAL_MODEL_TOKENS:
                tokens.append(message.content)
            yield message

        if LOG_INDIVIDUAL_MODEL_TOKENS and tokens:
            logger.debug(f"Model Tokens: {tokens}")

    @abc.abstractmethod
    def _stream_implementation(
        self,
        prompt: LanguageModelInput,
        tools: list[dict] | None = None,
        tool_choice: ToolChoiceOptions | None = None,
        structured_response_format: dict | None = None,
    ) -> Iterator[BaseMessage]:
        """
        流式调用的具体实现抽象方法
        
        参数：
        prompt: 模型输入
        tools: 可用工具列表
        tool_choice: 工具选择选项
        structured_response_format: 结构化响应格式
        
        返回：
        Iterator[BaseMessage]: 模型响应消息的迭代器
        """
        raise NotImplementedError
