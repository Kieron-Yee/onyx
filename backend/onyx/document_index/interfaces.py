"""
文件功能说明：
该文件定义了文档索引系统的核心接口和数据模型。包含了文档索引、检索、更新、删除等基本操作的抽象基类，
以及文档处理过程中需要用到的各种数据类型定义。这些接口为整个文档管理系统提供了统一的规范。
"""

import abc
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from onyx.access.models import DocumentAccess
from onyx.context.search.models import IndexFilters
from onyx.context.search.models import InferenceChunkUncleaned
from onyx.indexing.models import DocMetadataAwareIndexChunk
from shared_configs.model_server_models import Embedding


@dataclass(frozen=True)
class DocumentInsertionRecord:
    """
    文档插入记录，用于追踪文档的插入状态
    
    属性：
        document_id: 文档唯一标识符
        already_existed: 标识文档是否已经存在
    """
    document_id: str
    already_existed: bool


@dataclass(frozen=True)
class VespaChunkRequest:
    """
    Vespa分块请求数据类，用于指定需要检索的文档块范围
    
    属性：
        document_id: 文档ID
        min_chunk_ind: 最小块索引
        max_chunk_ind: 最大块索引
    """
    document_id: str
    min_chunk_ind: int | None = None
    max_chunk_ind: int | None = None

    @property
    def is_capped(self) -> bool:
        """
        判断是否设置了块范围上限
        
        NOTE: If the max chunk index is not None, then the chunk request is capped
        注释：如果最大块索引不为None，则说明设置了范围上限
        If the min chunk index is None, we can assume the min is 0
        如果最小块索引为None，我们假定最小值为0
        """
        return self.max_chunk_ind is not None

    @property
    def range(self) -> int | None:
        if self.max_chunk_ind is not None:
            return (self.max_chunk_ind - (self.min_chunk_ind or 0)) + 1
        return None


@dataclass
class IndexBatchParams:
    """
    批量索引参数类，包含批量索引文档所需的信息
    
    Information necessary for efficiently indexing a batch of documents

    属性：
        doc_id_to_previous_chunk_cnt: 文档ID到之前块数量的映射
        doc_id_to_new_chunk_cnt: 文档ID到新块数量的映射
        tenant_id: 租户ID
        large_chunks_enabled: 是否启用大块
    """

    doc_id_to_previous_chunk_cnt: dict[str, int | None]
    doc_id_to_new_chunk_cnt: dict[str, int]
    tenant_id: str | None
    large_chunks_enabled: bool


@dataclass
class MinimalDocumentIndexingInfo:
    """
    最小文档索引信息类，包含索引文档所需的最少信息
    
    Minimal information necessary for indexing a document

    属性：
        doc_id: 文档ID
        chunk_start_index: 块起始索引
    """

    doc_id: str
    chunk_start_index: int


@dataclass
class EnrichedDocumentIndexingInfo(MinimalDocumentIndexingInfo):
    """
    丰富的文档索引信息类，包含版本和块范围等信息
    
    Enriched information necessary for indexing a document, including version and chunk range.

    属性：
        old_version: 是否为旧版本
        chunk_end_index: 块结束索引
    """

    old_version: bool
    chunk_end_index: int


@dataclass
class DocumentMetadata:
    """
    文档元数据类，在首次遇到文档时需要插入到Postgres中
    
    Document information that needs to be inserted into Postgres on first time encountering this
    document during indexing across any of the connectors.

    属性：
        connector_id: 连接器ID
        credential_id: 凭证ID
        document_id: 文档ID
        semantic_identifier: 语义标识符
        first_link: 首次链接
        doc_updated_at: 文档更新时间
        primary_owners: 主要所有者
        secondary_owners: 次要所有者
        from_ingestion_api: 是否来自摄取API
    """

    connector_id: int
    credential_id: int
    document_id: str
    semantic_identifier: str
    first_link: str
    doc_updated_at: datetime | None = None
    primary_owners: list[str] | None = None
    secondary_owners: list[str] | None = None
    from_ingestion_api: bool = False


@dataclass
class VespaDocumentFields:
    """
    Vespa文档字段类，指定文档在Vespa中的字段。设置为None的字段将被忽略。
    
    Specifies fields in Vespa for a document. Fields set to None will be ignored.
    Perhaps we should name this in an implementation agnostic fashion, but it's more
    understandable like this for now.

    属性：
        access: 访问控制
        document_sets: 文档集
        boost: 提升值
        hidden: 是否隐藏
    """

    access: DocumentAccess | None = None
    document_sets: set[str] | None = None
    boost: float | None = None
    hidden: bool | None = None


@dataclass
class UpdateRequest:
    """
    更新请求类，用于更新文档的访问控制、文档集、提升值和隐藏状态
    
    For all document_ids, update the allowed_users and the boost to the new values
    Does not update any of the None fields

    属性：
        document_ids: 文档ID列表
        access: 访问控制
        document_sets: 文档集
        boost: 提升值
        hidden: 是否隐藏
    """

    document_ids: list[str]
    access: DocumentAccess | None = None
    document_sets: set[str] | None = None
    boost: float | None = None
    hidden: bool | None = None


class Verifiable(abc.ABC):
    """
    可验证接口类，必须实现文档索引模式验证功能。例如，验证索引、查询、过滤和返回字段的所有必要属性在模式中是否有效。
    
    Class must implement document index schema verification. For example, verify that all of the
    necessary attributes for indexing, querying, filtering, and fields to return from search are
    all valid in the schema.

    参数：
        index_name: 当前用于查询的主索引名称
        secondary_index_name: 后台构建的次索引名称（如果存在）。某些文档索引功能作用于主索引和次索引，某些仅作用于一个索引。
    """

    @abc.abstractmethod
    def __init__(
        self,
        index_name: str,
        secondary_index_name: str | None,
        *args: Any,
        **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self.index_name = index_name
        self.secondary_index_name = secondary_index_name

    @abc.abstractmethod
    def ensure_indices_exist(
        self,
        index_embedding_dim: int,
        secondary_index_embedding_dim: int | None,
    ) -> None:
        """
        验证文档索引是否存在并与代码中的预期一致。
        
        Verify that the document index exists and is consistent with the expectations in the code.

        参数：
            index_embedding_dim: 向量相似性搜索的向量维度
            secondary_index_embedding_dim: 后台构建的次索引的向量维度。次索引应仅在切换嵌入模型时构建，因此该维度应与主索引不同。
        """
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def register_multitenant_indices(
        indices: list[str],
        embedding_dims: list[int],
    ) -> None:
        """
        注册多租户索引。
        
        Register multitenant indices with the document index.
        """
        raise NotImplementedError


class Indexable(abc.ABC):
    """
    可索引接口类，必须实现文档块的索引功能。
    
    Class must implement the ability to index document chunks
    """

    @abc.abstractmethod
    def index(
        self,
        chunks: list[DocMetadataAwareIndexChunk],
        index_batch_params: IndexBatchParams,
    ) -> set[DocumentInsertionRecord]:
        """
        接收文档块列表并在文档索引中对其进行索引。
        
        Takes a list of document chunks and indexes them in the document index

        NOTE: When a document is reindexed/updated here, it must clear all of the existing document
        chunks before reindexing. This is because the document may have gotten shorter since the
        last run. Therefore, upserting the first 0 through n chunks may leave some old chunks that
        have not been written over.

        NOTE: The chunks of a document are never separated into separate index() calls. So there is
        no worry of receiving the first 0 through n chunks in one index call and the next n through
        m chunks of a docu in the next index call.

        NOTE: Due to some asymmetry between the primary and secondary indexing logic, this function
        only needs to index chunks into the PRIMARY index. Do not update the secondary index here,
        it is done automatically outside of this code.

        参数：
            chunks: 包含索引所需的所有信息的文档块。
            tenant_id: 正在索引其块的用户的租户ID
            large_chunks_enabled: 是否启用大块

        返回：
            映射到唯一文档的文档ID列表，用于在更新时去重块，以及文档是新索引的还是已存在并刚刚更新的
        """
        raise NotImplementedError


class Deletable(abc.ABC):
    """
    可删除接口类，必须实现通过给定唯一文档ID删除文档的功能。
    
    Class must implement the ability to delete document by a given unique document id.
    """

    @abc.abstractmethod
    def delete_single(self, doc_id: str) -> int:
        """
        根据单个文档ID，从文档索引中硬删除它。
        
        Given a single document id, hard delete it from the document index

        参数：
            doc_id: 连接器指定的文档ID
        """
        raise NotImplementedError


class Updatable(abc.ABC):
    """
    可更新接口类，必须实现无需更新所有字段即可更新文档某些属性的功能。具体来说，需要能够更新：
    - 访问控制列表
    - 文档集成员资格
    - 提升值（从反馈机制中学习）
    - 文档是否隐藏，隐藏的文档不会从搜索中返回
    
    Class must implement the ability to update certain attributes of a document without needing to
    update all of the fields. Specifically, needs to be able to update:
    - Access Control List
    - Document-set membership
    - Boost value (learning from feedback mechanism)
    - Whether the document is hidden or not, hidden documents are not returned from search
    """

    @abc.abstractmethod
    def update_single(self, doc_id: str, fields: VespaDocumentFields) -> int:
        """
        使用指定字段更新文档的所有块。None值表示该字段不需要更新。
        
        Updates all chunks for a document with the specified fields.
        None values mean that the field does not need an update.

        单个更新函数的理由是，它允许在更高/更战略的层次上进行重试和并行处理，更易于阅读，并允许我们单独处理每个文档的错误情况。

        参数：
            fields: 要更新的文档字段。设置为None的字段不会更改。

        返回：
            None
        """
        raise NotImplementedError

    @abc.abstractmethod
    def update(self, update_requests: list[UpdateRequest]) -> None:
        """
        更新某些块集。要更新的文档和字段在更新请求中指定。列表中的每个更新请求将其更改应用于具有这些ID的所有文档。
        None值表示该字段不需要更新。
        
        Updates some set of chunks. The document and fields to update are specified in the update
        requests. Each update request in the list applies its changes to a list of document ids.
        None values mean that the field does not need an update.

        参数：
            update_requests: 在更新请求中，对于文档ID列表，应用相同的更新。这是为了批量处理效率。许多更新在连接器级别完成，连接器有许多文档
        """
        raise NotImplementedError


class IdRetrievalCapable(abc.ABC):
    """
    可基于ID检索接口类，必须实现以下功能之一：
    - 给定文档ID，按顺序检索文档的所有块。
    - 给定文档ID和块索引（从0开始），检索特定块。
    
    Class must implement the ability to retrieve either:
    - all of the chunks of a document IN ORDER given a document id.
    - a specific chunk given a document id and a chunk index (0 based)
    """

    @abc.abstractmethod
    def id_based_retrieval(
        self,
        chunk_requests: list[VespaChunkRequest],
        filters: IndexFilters,
        batch_retrieval: bool = False,
    ) -> list[InferenceChunkUncleaned]:
        """
        基于文档ID获取块
        
        Fetch chunk(s) based on document id

        NOTE: This is used to reconstruct a full document or an extended (multi-chunk) section
        of a document. Downstream currently assumes that the chunking does not introduce overlaps
        between the chunks. If there are overlaps for the chunks, then the reconstructed document
        or extended section will have duplicate segments.

        参数：
            chunk_requests: 包含文档ID和要检索的块范围的请求
            filters: 检索时应用的过滤器
            batch_retrieval: 如果为True，则执行批量检索

        返回：
            文档ID的块列表或按指定块索引和文档ID的特定块
        """
        raise NotImplementedError


class HybridCapable(abc.ABC):
    """
    混合搜索接口类，必须实现混合（关键词+向量）搜索功能。
    
    Class must implement hybrid (keyword + vector) search functionality
    """

    @abc.abstractmethod
    def hybrid_retrieval(
        self,
        query: str,
        query_embedding: Embedding,
        final_keywords: list[str] | None,
        filters: IndexFilters,
        hybrid_alpha: float,
        time_decay_multiplier: float,
        num_to_retrieve: int,
        offset: int = 0,
    ) -> list[InferenceChunkUncleaned]:
        """
        运行混合搜索并返回推理块列表。
        
        Run hybrid search and return a list of inference chunks.

        NOTE: the query passed in here is the unprocessed plain text query. Preprocessing is
        expected to be handled by this function as it may depend on the index implementation.
        Things like query expansion, synonym injection, stop word removal, lemmatization, etc. are
        done here.

        参数：
            query: 未修改的用户查询。这是获取匹配高亮关键词所需的
            query_embedding: 查询的向量表示，必须具有主索引的正确维度
            final_keywords: 要使用的最终关键词，默认为查询
            filters: 标准过滤器对象
            hybrid_alpha: 关键词和向量搜索结果之间的加权。重要的是两个分数归一化到相同范围，以便进行有意义的比较。1表示100%加权在向量分数上，0表示100%加权在关键词分数上。
            time_decay_multiplier: 随着文档老化，文档分数的衰减倍数。某些查询基于个性化设置，将此值设置为默认值的2倍或3倍
            num_to_retrieve: 要返回的最高匹配块的数量
            offset: 要跳过的最高匹配块的数量（类似于分页）

        返回：
            基于关键词和向量/语义搜索分数加权和的最佳匹配块
        """
        raise NotImplementedError


class AdminCapable(abc.ABC):
    """
    管理员检索接口类，必须实现管理员“资源管理器”页面的搜索功能。假设管理员不是在“搜索”知识，而是已经有某个文档在心中。
    他们要么是因为知道它是一个很好的参考文档而积极提升它，要么是为了“弃用”而消极提升它，或者隐藏文档。
    
    Class must implement a search for the admin "Explorer" page. The assumption here is that the
    admin is not "searching" for knowledge but has some document already in mind. They are either
    looking to positively boost it because they know it's a good reference document, looking to
    negatively boost it as a way of "deprecating", or hiding the document.

    假设管理员知道文档名称，此搜索对标题匹配有很高的重视。

    Suggested implementation:
    Keyword only, BM25 search with 5x weighting on the title field compared to the contents
    """

    @abc.abstractmethod
    def admin_retrieval(
        self,
        query: str,
        filters: IndexFilters,
        num_to_retrieve: int,
        offset: int = 0,
    ) -> list[InferenceChunkUncleaned]:
        """
        运行管理员文档资源管理器页面的特殊搜索
        
        Run the special search for the admin document explorer page

        参数：
            query: 未修改的用户查询。尽管在此流程中，未修改的可能是最好的
            filters: 标准过滤器对象
            num_to_retrieve: 要返回的最高匹配块的数量
            offset: 要跳过的最高匹配块的数量（类似于分页）

        返回：
            资源管理器页面查询的最佳匹配块列表
        """
        raise NotImplementedError


class RandomCapable(abc.ABC):
    """
    随机检索接口类，必须实现随机文档检索功能。
    
    Class must implement random document retrieval capability
    """

    @abc.abstractmethod
    def random_retrieval(
        self,
        filters: IndexFilters,
        num_to_retrieve: int = 10,
    ) -> list[InferenceChunkUncleaned]:
        """
        检索匹配过滤器的随机块
        
        Retrieve random chunks matching the filters
        """
        raise NotImplementedError


class BaseIndex(
    Verifiable,
    Indexable,
    Updatable,
    Deletable,
    AdminCapable,
    IdRetrievalCapable,
    RandomCapable,
    abc.ABC,
):
    """
    基础文档索引类，包含除实际查询方法外的所有基本功能。
    
    All basic document index functionalities excluding the actual querying approach.

    总结来说，文档索引需要能够：
    - 验证模式定义是否有效
    - 索引新文档
    - 更新现有文档的特定属性
    - 删除文档
    - 提供管理员文档资源管理器页面的搜索
    - 基于文档ID检索文档
    """


class DocumentIndex(HybridCapable, BaseIndex, abc.ABC):
    """
    完整的文档索引接口类，继承了所有基础功能接口
    
    A valid document index that can plug into all Onyx flows must implement all of these
    functionalities, though "technically" it does not need to be keyword or vector capable as
    currently all default search flows use Hybrid Search.
    
    一个有效的文档索引必须实现所有这些功能，尽管从技术上讲，由于目前所有默认搜索流程都使用混合搜索，
    因此它不需要具备关键词或向量搜索能力。
    """
