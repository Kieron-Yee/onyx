"""
此模块用于评估文档片段对用户查询的相关性和有用性。
主要功能包括：
1. 使用LLM评估单个文档片段的有用性
2. 批量评估多个文档片段的有用性
"""

from collections.abc import Callable

from onyx.configs.chat_configs import DISABLE_LLM_DOC_RELEVANCE
from onyx.llm.interfaces import LLM
from onyx.llm.utils import dict_based_prompt_to_langchain_prompt
from onyx.llm.utils import message_to_string
from onyx.prompts.llm_chunk_filter import NONUSEFUL_PAT
from onyx.prompts.llm_chunk_filter import SECTION_FILTER_PROMPT
from onyx.utils.logger import setup_logger
from onyx.utils.threadpool_concurrency import run_functions_tuples_in_parallel

logger = setup_logger()


def llm_eval_section(
    query: str,
    section_content: str,
    llm: LLM,
    title: str,
    metadata: dict[str, str | list[str]],
) -> bool:
    """
    使用LLM评估文档片段是否与用户查询相关。
    
    参数:
        query: 用户的查询字符串
        section_content: 需要评估的文档片段内容
        llm: LLM实例
        title: 文档标题
        metadata: 文档元数据字典
    
    返回:
        bool: 如果片段有用返回True，否则返回False
    """

    def _get_metadata_str(metadata: dict[str, str | list[str]]) -> str:
        """
        将元数据字典转换为格式化字符串。
        
        参数:
            metadata: 包含元数据的字典
        
        返回:
            str: 格式化后的元数据字符串
        """
        metadata_str = "\nMetadata:\n"
        for key, value in metadata.items():
            value_str = ", ".join(value) if isinstance(value, list) else value
            metadata_str += f"{key} - {value_str}\n"
        return metadata_str

    def _get_usefulness_messages() -> list[dict[str, str]]:
        """
        构造用于评估有用性的消息列表。
        
        返回:
            list[dict[str, str]]: 包含提示信息的消息列表
        """
        metadata_str = _get_metadata_str(metadata) if metadata else ""
        messages = [
            {
                "role": "user",
                "content": SECTION_FILTER_PROMPT.format(
                    title=title.replace("\n", " "),
                    chunk_text=section_content,
                    user_query=query,
                    optional_metadata=metadata_str,
                ),
            },
        ]
        return messages

    def _extract_usefulness(model_output: str) -> bool:
        """
        从模型输出中提取有用性判断。
        Default useful if the LLM doesn't match pattern exactly
        This is because it's better to trust the (re)ranking if LLM fails
        默认有用如果LLM不能完全匹配模式，这是因为如果LLM失败时最好相信(重新)排名
        
        参数:
            model_output: LLM的输出字符串
            
        返回:
            bool: 文档片段是否有用的判断结果
        """
        if model_output.strip().strip('"').lower() == NONUSEFUL_PAT.lower():
            return False
        return True

    messages = _get_usefulness_messages()
    filled_llm_prompt = dict_based_prompt_to_langchain_prompt(messages)
    model_output = message_to_string(llm.invoke(filled_llm_prompt))
    logger.debug(model_output)

    return _extract_usefulness(model_output)


def llm_batch_eval_sections(
    query: str,
    section_contents: list[str],
    llm: LLM,
    titles: list[str],
    metadata_list: list[dict[str, str | list[str]]],
    use_threads: bool = True,
) -> list[bool]:
    """
    批量评估多个文档片段的有用性。
    
    参数:
        query: 用户查询字符串
        section_contents: 文档片段内容列表
        llm: LLM实例
        titles: 文档标题列表
        metadata_list: 元数据字典列表
        use_threads: 是否使用多线程处理
        
    返回:
        list[bool]: 每个文档片段的有用性评估结果列表
    """
    
    if DISABLE_LLM_DOC_RELEVANCE:
        raise RuntimeError(
            "LLM Doc Relevance is globally disabled, " # LLM文档相关性已全局禁用
            "this should have been caught upstream."   # 这应该在上游被捕获
        )

    if use_threads:
        # 使用多线程并行处理多个文档片段的评估
        functions_with_args: list[tuple[Callable, tuple]] = [
            (llm_eval_section, (query, section_content, llm, title, metadata))
            for section_content, title, metadata in zip(
                section_contents, titles, metadata_list
            )
        ]

        logger.debug(
            "Running LLM usefulness eval in parallel (following logging may be out of order)"
            # 并行运行LLM有用性评估（后续日志可能顺序混乱）
        )
        parallel_results = run_functions_tuples_in_parallel(
            functions_with_args, allow_failures=True
        )

        # In case of failure/timeout, don't throw out the section
        # 如果发生失败/超时，不要丢弃该片段
        return [True if item is None else item for item in parallel_results]

    else:
        # 顺序处理多个文档片段的评估
        return [
            llm_eval_section(query, section_content, llm, title, metadata)
            for section_content, title, metadata in zip(
                section_contents, titles, metadata_list
            )
        ]
