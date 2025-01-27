"""
搜索工具实现的工具函数模块
此模块提供了将不同类型的文档对象转换为统一字典格式的工具函数，
用于处理搜索结果的标准化输出。
"""

from onyx.chat.models import LlmDoc
from onyx.context.search.models import InferenceSection
from onyx.prompts.prompt_utils import clean_up_source


def llm_doc_to_dict(llm_doc: LlmDoc, doc_num: int) -> dict:
    """
    将LlmDoc对象转换为标准化的字典格式
    
    参数:
        llm_doc (LlmDoc): 需要转换的LLM文档对象
        doc_num (int): 文档的序号
        
    返回:
        dict: 包含文档信息的字典，包括文档编号、标题、内容、来源和元数据等
    """
    doc_dict = {
        "document_number": doc_num + 1,
        "title": llm_doc.semantic_identifier,
        "content": llm_doc.content,
        "source": clean_up_source(llm_doc.source_type),
        "metadata": llm_doc.metadata,
    }
    if llm_doc.updated_at:
        doc_dict["updated_at"] = llm_doc.updated_at.strftime("%B %d, %Y %H:%M")
    return doc_dict


def section_to_dict(section: InferenceSection, section_num: int) -> dict:
    """
    将InferenceSection对象转换为标准化的字典格式
    
    参数:
        section (InferenceSection): 需要转换的推理部分对象
        section_num (int): 部分的序号
        
    返回:
        dict: 包含段落信息的字典，包括段落编号、标题、内容、来源和元数据等
    """
    doc_dict = {
        "document_number": section_num + 1,
        "title": section.center_chunk.semantic_identifier,
        "content": section.combined_content,
        "source": clean_up_source(section.center_chunk.source_type),
        "metadata": section.center_chunk.metadata,
    }
    if section.center_chunk.updated_at:
        doc_dict["updated_at"] = section.center_chunk.updated_at.strftime(
            "%B %d, %Y %H:%M"
        )
    return doc_dict
