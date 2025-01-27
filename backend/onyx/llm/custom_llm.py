"""
此文件实现了自定义LLM模型服务器的接口类。
主要功能：
- 提供与自定义LLM服务器交互的标准接口
- 处理模型请求和响应
- 支持同步和流式输出
"""

import json
from collections.abc import Iterator

import requests
from langchain.schema.language_model import LanguageModelInput
from langchain_core.messages import AIMessage
from langchain_core.messages import BaseMessage
from requests import Timeout

from onyx.configs.model_configs import GEN_AI_NUM_RESERVED_OUTPUT_TOKENS
from onyx.llm.interfaces import LLM
from onyx.llm.interfaces import ToolChoiceOptions
from onyx.llm.utils import convert_lm_input_to_basic_string
from onyx.utils.logger import setup_logger


logger = setup_logger()


class CustomModelServer(LLM):
    """This class is to provide an example for how to use Onyx
    with any LLM, even servers with custom API definitions.
    To use with your own model server, simply implement the functions
    below to fit your model server expectation

    The implementation below works against the custom FastAPI server from the blog:
    https://medium.com/@yuhongsun96/how-to-augment-llms-with-private-data-29349bd8ae9f
    
    此类提供了一个如何将Onyx与任何LLM配合使用的示例，
    即使是具有自定义API定义的服务器也可以使用。
    要与您自己的模型服务器一起使用，只需实现下面的函数以适应您的模型服务器要求即可。
    
    以下实现适用于博客中的自定义FastAPI服务器。
    """

    @property
    def requires_api_key(self) -> bool:
        """
        判断是否需要API密钥
        返回值：
            bool: 是否需要API密钥，默认为False
        """
        return False

    def __init__(
        self,
        api_key: str | None,
        timeout: int,
        endpoint: str,
        max_output_tokens: int = GEN_AI_NUM_RESERVED_OUTPUT_TOKENS,
    ):
        """
        初始化自定义模型服务器
        参数：
            api_key: API密钥（此示例中未使用）
            timeout: 超时时间（秒）
            endpoint: 模型服务器的端点URL
            max_output_tokens: 最大输出token数
        """
        if not endpoint:
            raise ValueError(
                "Cannot point Onyx to a custom LLM server without providing the "
                "endpoint for the model server."
                "无法在不提供模型服务器端点的情况下将Onyx指向自定义LLM服务器。"
            )

        self._endpoint = endpoint
        self._max_output_tokens = max_output_tokens
        self._timeout = timeout

    def _execute(self, input: LanguageModelInput) -> AIMessage:
        """
        执行模型推理
        参数：
            input: 语言模型输入
        返回值：
            AIMessage: 模型生成的响应消息
        """
        headers = {
            "Content-Type": "application/json",
        }

        data = {
            "inputs": convert_lm_input_to_basic_string(input),
            "parameters": {
                "temperature": 0.0,
                "max_tokens": self._max_output_tokens,
            },
        }
        try:
            response = requests.post(
                self._endpoint, headers=headers, json=data, timeout=self._timeout
            )
        except Timeout as error:
            raise Timeout(f"Model inference to {self._endpoint} timed out") from error

        response.raise_for_status()
        response_content = json.loads(response.content).get("generated_text", "")
        return AIMessage(content=response_content)

    def log_model_configs(self) -> None:
        """
        记录模型配置信息
        """
        logger.debug(f"Custom model at: {self._endpoint}")

    def _invoke_implementation(
        self,
        prompt: LanguageModelInput,
        tools: list[dict] | None = None,
        tool_choice: ToolChoiceOptions | None = None,
        structured_response_format: dict | None = None,
    ) -> BaseMessage:
        """
        实现同步调用模型
        参数：
            prompt: 输入提示
            tools: 可用工具列表
            tool_choice: 工具选择选项
            structured_response_format: 结构化响应格式
        返回值：
            BaseMessage: 模型生成的响应消息
        """
        return self._execute(prompt)

    def _stream_implementation(
        self,
        prompt: LanguageModelInput,
        tools: list[dict] | None = None,
        tool_choice: ToolChoiceOptions | None = None,
        structured_response_format: dict | None = None,
    ) -> Iterator[BaseMessage]:
        """
        实现流式调用模型
        参数：
            prompt: 输入提示
            tools: 可用工具列表
            tool_choice: 工具选择选项
            structured_response_format: 结构化响应格式
        返回值：
            Iterator[BaseMessage]: 模型生成的响应消息迭代器
        """
        yield self._execute(prompt)
