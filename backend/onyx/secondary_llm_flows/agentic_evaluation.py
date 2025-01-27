"""
此文件实现了代理评估(Agentic Evaluation)相关的功能。
主要用于评估推理章节与查询的相关性，并生成分析结果。
"""

import re

from onyx.chat.models import SectionRelevancePiece
from onyx.context.search.models import InferenceSection
from onyx.llm.interfaces import LLM
from onyx.llm.utils import dict_based_prompt_to_langchain_prompt
from onyx.llm.utils import message_to_string
from onyx.prompts.agentic_evaluation import AGENTIC_SEARCH_SYSTEM_PROMPT
from onyx.prompts.agentic_evaluation import AGENTIC_SEARCH_USER_PROMPT
from onyx.utils.logger import setup_logger

logger = setup_logger()


def _get_agent_eval_messages(
    title: str, content: str, query: str, center_metadata: str
) -> list[dict[str, str]]:
    """
    构建代理评估所需的消息列表
    
    参数:
        title: str - 文档标题
        content: str - 文档内容
        query: str - 查询字符串
        center_metadata: str - 中心元数据
        
    返回:
        list[dict[str, str]] - 包含系统提示和用户提示的消息列表
    """
    messages = [
        {
            "role": "system",
            "content": AGENTIC_SEARCH_SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": AGENTIC_SEARCH_USER_PROMPT.format(
                title=title,
                content=content,
                query=query,
                optional_metadata=center_metadata,
            ),
        },
    ]
    return messages


def evaluate_inference_section(
    document: InferenceSection, query: str, llm: LLM
) -> SectionRelevancePiece:
    """
    评估推理章节与查询的相关性
    
    参数:
        document: InferenceSection - 待评估的推理章节
        query: str - 查询字符串
        llm: LLM - 语言模型接口
        
    返回:
        SectionRelevancePiece - 包含评估结果的相关性片段对象
    """
    def _get_metadata_str(metadata: dict[str, str | list[str]]) -> str:
        """
        将元数据转换为字符串格式
        
        参数:
            metadata: dict - 元数据字典
            
        返回:
            str - 格式化后的元数据字符串
        """
        metadata_str = "\n\nMetadata:\n"
        for key, value in metadata.items():
            value_str = ", ".join(value) if isinstance(value, list) else value
            metadata_str += f"{key} - {value_str}\n"

        # Since there is now multiple sections, add this prefix for clarity
        # 由于现在有多个章节，添加此前缀以提高清晰度
        return metadata_str + "\nContent:"

    document_id = document.center_chunk.document_id
    semantic_id = document.center_chunk.semantic_identifier
    contents = document.combined_content
    center_metadata = document.center_chunk.metadata
    center_metadata_str = _get_metadata_str(center_metadata) if center_metadata else ""

    messages = _get_agent_eval_messages(
        title=semantic_id,
        content=contents,
        query=query,
        center_metadata=center_metadata_str,
    )
    filled_llm_prompt = dict_based_prompt_to_langchain_prompt(messages)
    try:
        model_output = message_to_string(llm.invoke(filled_llm_prompt))

        # Search for the "Useful Analysis" section in the model output
        # This regex looks for "2. Useful Analysis" (case-insensitive) followed by an optional colon,
        # then any text up to "3. Final Relevance"
        # The (?i) flag makes it case-insensitive, and re.DOTALL allows the dot to match newlines
        # If no match is found, the entire model output is used as the analysis
        # 在模型输出中查找"Useful Analysis"部分
        # 这个正则表达式查找"2. Useful Analysis"(不区分大小写)后面跟着可选的冒号
        # 然后匹配直到"3. Final Relevance"之前的所有文本
        # (?i)标志使其不区分大小写，re.DOTALL允许点号匹配换行符
        # 如果没有找到匹配，则使用整个模型输出作为分析结果
        analysis_match = re.search(
            r"(?i)2\.\s*useful analysis:?\s*(.+?)\n\n3\.\s*final relevance",
            model_output,
            re.DOTALL,
        )
        analysis = analysis_match.group(1).strip() if analysis_match else model_output

        # Get the last non-empty line
        # 获取最后一个非空行
        last_line = next(
            (line for line in reversed(model_output.split("\n")) if line.strip()), ""
        )
        relevant = last_line.strip().lower().startswith("true")
    except Exception as e:
        logger.exception(f"An issue occured during the agentic evaluation process. {e}")
        # 在代理评估过程中发生异常
        relevant = False
        analysis = ""

    return SectionRelevancePiece(
        document_id=document_id,
        chunk_id=document.center_chunk.chunk_id,
        relevant=relevant,
        content=analysis,
    )
