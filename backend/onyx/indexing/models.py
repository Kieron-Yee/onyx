"""
此模块定义了文档索引相关的数据模型。
包含了用于文档分块、嵌入向量处理、索引设置等核心功能的类定义。
"""

from typing import TYPE_CHECKING

from pydantic import BaseModel
from pydantic import Field

from onyx.access.models import DocumentAccess
from onyx.connectors.models import Document
from onyx.utils.logger import setup_logger
from shared_configs.enums import EmbeddingProvider
from shared_configs.model_server_models import Embedding

if TYPE_CHECKING:
    from onyx.db.models import SearchSettings


logger = setup_logger()


class ChunkEmbedding(BaseModel):
    """
    文档块的嵌入向量模型，包含完整嵌入向量和迷你块嵌入向量列表
    
    属性:
        full_embedding: 完整文档块的嵌入向量
        mini_chunk_embeddings: 迷你块的嵌入向量列表
    """
    full_embedding: Embedding
    mini_chunk_embeddings: list[Embedding]


class BaseChunk(BaseModel):
    """
    基础文档块模型
    
    属性:
        chunk_id: 文档块ID
        blurb: 文档块第一个段落的开始部分
        content: 文档块的完整内容
        source_links: 原始文本中的链接及其偏移量
        section_continuation: 标识该块是否从段落中间开始
    """
    chunk_id: int
    blurb: str  # The first sentence(s) of the first Section of the chunk
    content: str
    # Holds the link and the offsets into the raw Chunk text
    source_links: dict[int, str] | None
    section_continuation: bool  # True if this Chunk's start is not at the start of a Section


class DocAwareChunk(BaseChunk):
    """
    包含文档信息的文档块模型，继承自BaseChunk
    
    属性:
        source_document: 源文档对象
        title_prefix: 文档标题前缀
        metadata_suffix_semantic: 语义检索用的元数据后缀
        metadata_suffix_keyword: 关键词检索用的元数据后缀
        mini_chunk_texts: 迷你块文本列表
        large_chunk_id: 大块ID
        large_chunk_reference_ids: 大块引用ID列表
    """
    # 在索引流程中，我们可以访问完整的"Document"对象
    # 在推理过程中，我们只能访问文档ID，而不重建Document对象
    source_document: Document

    # 如果标题太长并占用了太多块空间，这可能是一个空字符串
    # 这并不一定意味着文档没有标题
    title_prefix: str

    # 在索引期间，我们还可以（可选地）从元数据字典构建元数据字符串
    # 这也被索引，以便在索引后可以将其剥离，这样可以支持向后兼容的多次元数据表示迭代
    metadata_suffix_semantic: str
    metadata_suffix_keyword: str

    mini_chunk_texts: list[str] | None

    large_chunk_id: int | None

    large_chunk_reference_ids: list[int] = Field(default_factory=list)

    def to_short_descriptor(self) -> str:
        """
        生成文档块的简短描述
        
        返回:
            str: 包含块ID和源文档信息的描述字符串
        """
        return (
            f"Chunk ID: '{self.chunk_id}'; {self.source_document.to_short_descriptor()}"
        )


class IndexChunk(DocAwareChunk):
    """
    用于索引的文档块模型，包含嵌入向量信息
    
    属性:
        embeddings: 文档块的嵌入向量
        title_embedding: 标题的嵌入向量
    """
    embeddings: ChunkEmbedding
    title_embedding: Embedding | None


# TODO(rkuo): 目前，在索引期间发送的这些额外元数据仅用于提高速度，
# 但完整的一致性是在后台同步时实现的
class DocMetadataAwareIndexChunk(IndexChunk):
    """
    An `IndexChunk` that contains all necessary metadata to be indexed. This includes
    the following:
    包含完整元数据的索引文档块，用于索引过程。包含以下信息：

    access: holds all information about which users should have access to the
            source document for this chunk.
    access: 包含用户对该文档块源文档的访问权限信息

    document_sets: all document sets the source document for this chunk is a part
                   of. This is used for filtering / personas.
    document_sets: 该文档块源文档所属的所有文档集合，用于过滤和角色管理

    boost: influences the ranking of this chunk at query time. Positive -> ranked higher,
           negative -> ranked lower.
    boost: 影响查询时的排名，正值提高排名，负值降低排名
    """
    tenant_id: str | None = None
    access: "DocumentAccess"
    document_sets: set[str]
    boost: int

    @classmethod
    def from_index_chunk(
        cls,
        index_chunk: IndexChunk,
        access: "DocumentAccess",
        document_sets: set[str],
        boost: int,
        tenant_id: str | None,
    ) -> "DocMetadataAwareIndexChunk":
        """
        从IndexChunk创建DocMetadataAwareIndexChunk实例
        
        参数:
            index_chunk: 索引文档块
            access: 文档访问权限
            document_sets: 文档集合
            boost: 排名权重
            tenant_id: 租户ID
            
        返回:
            DocMetadataAwareIndexChunk实例
        """
        index_chunk_data = index_chunk.model_dump()
        return cls(
            **index_chunk_data,
            access=access,
            document_sets=document_sets,
            boost=boost,
            tenant_id=tenant_id,
        )


class EmbeddingModelDetail(BaseModel):
    """
    嵌入模型详细信息
    
    属性:
        id: 模型ID
        model_name: 模型名称
        normalize: 是否规范化
        query_prefix: 查询前缀
        passage_prefix: 段落前缀
        api_url: API地址
        provider_type: 提供商类型
        api_key: API密钥
    """
    id: int | None = None
    model_name: str
    normalize: bool
    query_prefix: str | None
    passage_prefix: str | None
    api_url: str | None = None
    provider_type: EmbeddingProvider | None = None
    api_key: str | None = None

    # 这会禁用pydantic的"model_"保护命名空间
    model_config = {"protected_namespaces": ()}

    @classmethod
    def from_db_model(
        cls,
        search_settings: "SearchSettings",
    ) -> "EmbeddingModelDetail":
        """
        从数据库模型创建EmbeddingModelDetail实例
        
        参数:
            search_settings: 搜索设置对象
            
        返回:
            EmbeddingModelDetail实例
        """
        return cls(
            id=search_settings.id,
            model_name=search_settings.model_name,
            normalize=search_settings.normalize,
            query_prefix=search_settings.query_prefix,
            passage_prefix=search_settings.passage_prefix,
            provider_type=search_settings.provider_type,
            api_key=search_settings.api_key,
            api_url=search_settings.api_url,
        )


# Additional info needed for indexing time
class IndexingSetting(EmbeddingModelDetail):
    """
    索引设置模型，继承自EmbeddingModelDetail
    
    属性:
        model_dim: 模型维度
        index_name: 索引名称
        multipass_indexing: 是否启用多通道索引
    """
    model_dim: int
    index_name: str | None
    multipass_indexing: bool

    # 这会禁用pydantic的"model_"保护命名空间
    model_config = {"protected_namespaces": ()}

    @classmethod
    def from_db_model(cls, search_settings: "SearchSettings") -> "IndexingSetting":
        """
        从数据库模型创建IndexingSetting实例
        
        参数:
            search_settings: 搜索设置对象
            
        返回:
            IndexingSetting实例
        """
        return cls(
            model_name=search_settings.model_name,
            model_dim=search_settings.model_dim,
            normalize=search_settings.normalize,
            query_prefix=search_settings.query_prefix,
            passage_prefix=search_settings.passage_prefix,
            provider_type=search_settings.provider_type,
            index_name=search_settings.index_name,
            multipass_indexing=search_settings.multipass_indexing,
        )
