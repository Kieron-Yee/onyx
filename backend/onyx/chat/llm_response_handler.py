"""
此文件主要用于处理LLM（大型语言模型）的响应流。
主要功能包括：
1. 处理工具调用响应
2. 处理回答响应
3. 管理响应流的取消操作
"""

from collections.abc import Callable
from collections.abc import Generator
from collections.abc import Iterator

from langchain_core.messages import BaseMessage

from onyx.chat.models import ResponsePart
from onyx.chat.models import StreamStopInfo
from onyx.chat.models import StreamStopReason
from onyx.chat.prompt_builder.build import LLMCall
from onyx.chat.stream_processing.answer_response_handler import AnswerResponseHandler
from onyx.chat.tool_handling.tool_response_handler import ToolResponseHandler


class LLMResponseHandlerManager:
    """
    LLM响应处理管理器
    负责协调工具响应处理器和回答响应处理器，管理整个LLM响应的处理流程
    """

    def __init__(
        self,
        tool_handler: ToolResponseHandler,
        answer_handler: AnswerResponseHandler,
        is_cancelled: Callable[[], bool],
    ):
        """
        初始化LLM响应处理管理器
        
        参数：
            tool_handler: 工具响应处理器，用于处理工具相关的响应
            answer_handler: 回答响应处理器，用于处理普通回答响应
            is_cancelled: 用于检查是否取消处理的回调函数
        """
        self.tool_handler = tool_handler
        self.answer_handler = answer_handler
        self.is_cancelled = is_cancelled

    def handle_llm_response(
        self,
        stream: Iterator[BaseMessage],
    ) -> Generator[ResponsePart, None, None]:
        """
        处理LLM的响应流
        
        参数：
            stream: BaseMessage的迭代器，包含LLM的响应消息流
            
        返回：
            生成器，产生ResponsePart类型的响应部分
            
        说明：
        - 如果处理被取消，将返回取消状态
        - tool handler doesn't do anything until the full message is received
          工具处理器在收到完整消息之前不会执行任何操作
        """
        all_messages: list[BaseMessage] = []
        for message in stream:
            if self.is_cancelled():
                yield StreamStopInfo(stop_reason=StreamStopReason.CANCELLED)
                return
            list(self.tool_handler.handle_response_part(message, all_messages))
            yield from self.answer_handler.handle_response_part(message, all_messages)
            all_messages.append(message)

        yield from self.tool_handler.handle_response_part(None, all_messages)
        yield from self.answer_handler.handle_response_part(None, all_messages)

    def next_llm_call(self, llm_call: LLMCall) -> LLMCall | None:
        """
        获取下一个LLM调用
        
        参数：
            llm_call: 当前的LLM调用对象
            
        返回：
            下一个LLM调用对象，如果没有下一个调用则返回None
        """
        return self.tool_handler.next_llm_call(llm_call)
