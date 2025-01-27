"""
此模块定义了与文档处理相关的数据模型类。
包含了用于文档摄取、处理结果和文档基本信息的Pydantic模型。
"""

from pydantic import BaseModel
from onyx.connectors.models import DocumentBase


class IngestionDocument(BaseModel):
    """
    文档摄取模型类，用于处理文档摄取请求。

    属性:
        document (DocumentBase): 待处理的文档基础信息
        cc_pair_id (int | None): 关联的cc_pair标识，可为空
    """
    document: DocumentBase
    cc_pair_id: int | None = None


class IngestionResult(BaseModel):
    """
    文档摄取结果模型类，用于返回文档处理的结果信息。

    属性:
        document_id (str): 处理后的文档唯一标识符
        already_existed (bool): 标识文档是否已经存在
    """
    document_id: str
    already_existed: bool


class DocMinimalInfo(BaseModel):
    """
    文档最小信息模型类，包含文档的基本信息。

    属性:
        document_id (str): 文档唯一标识符
        semantic_id (str): 文档语义标识符
        link (str | None): 文档链接，可为空
    """
    document_id: str
    semantic_id: str
    link: str | None = None
