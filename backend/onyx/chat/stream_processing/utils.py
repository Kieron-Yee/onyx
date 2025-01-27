"""
此文件包含用于处理文档流的实用工具函数和类。
主要功能是为文档块建立顺序映射关系，用于追踪和管理文档片段的顺序。
"""

from collections.abc import Sequence
from pydantic import BaseModel
from onyx.chat.models import LlmDoc
from onyx.context.search.models import InferenceChunk


class DocumentIdOrderMapping(BaseModel):
    """
    文档ID顺序映射类
    
    用于存储文档ID与其顺序编号之间的映射关系
    
    属性:
        order_mapping (dict[str, int]): 存储文档ID到顺序编号的字典映射
    """
    order_mapping: dict[str, int]


def map_document_id_order(
    chunks: Sequence[InferenceChunk | LlmDoc], 
    one_indexed: bool = True
) -> DocumentIdOrderMapping:
    """
    为文档块创建顺序映射
    
    参数:
        chunks (Sequence[InferenceChunk | LlmDoc]): 文档块或LLM文档的序列
        one_indexed (bool, optional): 是否使用1为起始索引。默认为True
    
    返回:
        DocumentIdOrderMapping: 包含文档ID到顺序编号映射的对象
    
    功能说明:
        - 接收一系列文档块作为输入
        - 为每个唯一的文档ID分配一个顺序编号
        - 顺序编号可以从0或1开始（由one_indexed参数控制）
        - 相同文档ID只会被分配一次顺序编号
        - 返回包含完整映射关系的DocumentIdOrderMapping对象
    """
    order_mapping = {}
    current = 1 if one_indexed else 0
    for chunk in chunks:
        if chunk.document_id not in order_mapping:
            order_mapping[chunk.document_id] = current
            current += 1

    return DocumentIdOrderMapping(order_mapping=order_mapping)
