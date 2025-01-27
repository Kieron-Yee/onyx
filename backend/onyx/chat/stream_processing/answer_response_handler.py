"""
此文件实现了处理LLM响应流的处理器类。
主要功能包括：
1. 定义响应处理的抽象基类
2. 实现引用处理和响应内容处理的具体类
3. 提供流式处理LLM响应的功能
"""

import abc
from collections.abc import Generator

from langchain_core.messages import BaseMessage

from onyx.chat.llm_response_handler import ResponsePart
from onyx.chat.models import CitationInfo
from onyx.chat.models import LlmDoc
from onyx.chat.stream_processing.citation_processing import CitationProcessor
from onyx.chat.stream_processing.utils import DocumentIdOrderMapping
from onyx.utils.logger import setup_logger

logger = setup_logger()


class AnswerResponseHandler(abc.ABC):
    """
    响应处理器的抽象基类，定义了处理响应部分的接口。
    """
    @abc.abstractmethod
    def handle_response_part(
        self,
        response_item: BaseMessage | None,
        previous_response_items: list[BaseMessage],
    ) -> Generator[ResponsePart, None, None]:
        """
        处理响应片段的抽象方法。

        参数:
            response_item: 当前需要处理的响应项，可能为None
            previous_response_items: 之前已处理的响应项列表

        返回:
            生成器，产生ResponsePart对象
        """
        raise NotImplementedError


class DummyAnswerResponseHandler(AnswerResponseHandler):
    """
    空响应处理器，用于测试或作为占位符。
    """
    def handle_response_part(
        self,
        response_item: BaseMessage | None,
        previous_response_items: list[BaseMessage],
    ) -> Generator[ResponsePart, None, None]:
        """
        空实现的响应处理方法，不执行任何操作。

        参数:
            response_item: 当前需要处理的响应项
            previous_response_items: 之前已处理的响应项列表

        返回:
            空生成器
        """
        # This is a dummy handler that returns nothing
        # 这是一个不返回任何内容的空处理器
        yield from []


class CitationResponseHandler(AnswerResponseHandler):
    """
    引用响应处理器，负责处理和管理文档引用。
    """
    def __init__(
        self,
        context_docs: list[LlmDoc],
        final_doc_id_to_rank_map: DocumentIdOrderMapping,
        display_doc_id_to_rank_map: DocumentIdOrderMapping,
    ):
        """
        初始化引用响应处理器。

        参数:
            context_docs: 上下文文档列表
            final_doc_id_to_rank_map: 最终文档ID到排名的映射
            display_doc_id_to_rank_map: 显示用的文档ID到排名的映射
        """
        self.context_docs = context_docs
        self.final_doc_id_to_rank_map = final_doc_id_to_rank_map
        self.display_doc_id_to_rank_map = display_doc_id_to_rank_map
        self.citation_processor = CitationProcessor(
            context_docs=self.context_docs,
            final_doc_id_to_rank_map=self.final_doc_id_to_rank_map,
            display_doc_id_to_rank_map=self.display_doc_id_to_rank_map,
        )
        self.processed_text = ""
        self.citations: list[CitationInfo] = []

        # TODO remove this after citation issue is resolved
        # TODO 解决引用问题后移除此处
        logger.debug(f"Document to ranking map {self.final_doc_id_to_rank_map}")

    def handle_response_part(
        self,
        response_item: BaseMessage | None,
        previous_response_items: list[BaseMessage],
    ) -> Generator[ResponsePart, None, None]:
        """
        处理包含引用的响应片段。

        参数:
            response_item: 当前需要处理的响应项
            previous_response_items: 之前已处理的响应项列表

        返回:
            处理后的ResponsePart生成器
        """
        if response_item is None:
            return

        content = (
            response_item.content if isinstance(response_item.content, str) else ""
        )

        # Process the new content through the citation processor
        # 通过引用处理器处理新的内容
        yield from self.citation_processor.process_token(content)


# No longer in use, remove later
# class QuotesResponseHandler(AnswerResponseHandler):
#     def __init__(
#         self,
#         context_docs: list[LlmDoc],
#         is_json_prompt: bool = True,
#     ):
#         self.quotes_processor = QuotesProcessor(
#             context_docs=context_docs,
#             is_json_prompt=is_json_prompt,
#         )

#     def handle_response_part(
#         self,
#         response_item: BaseMessage | None,
#         previous_response_items: list[BaseMessage],
#     ) -> Generator[ResponsePart, None, None]:
#         if response_item is None:
#             yield from self.quotes_processor.process_token(None)
#             return

#         content = (
#             response_item.content if isinstance(response_item.content, str) else ""
#         )

#         yield from self.quotes_processor.process_token(content)
