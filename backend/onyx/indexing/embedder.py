"""
本文件实现了文档索引嵌入的核心功能。
主要包含两个类：IndexingEmbedder（抽象基类）和 DefaultIndexingEmbedder（具体实现类）。
这些类负责将文档块转换为向量嵌入表示，用于后续的文档检索和匹配。
"""

from abc import ABC
from abc import abstractmethod

from onyx.db.models import SearchSettings
from onyx.indexing.indexing_heartbeat import IndexingHeartbeatInterface
from onyx.indexing.models import ChunkEmbedding
from onyx.indexing.models import DocAwareChunk
from onyx.indexing.models import IndexChunk
from onyx.natural_language_processing.search_nlp_models import EmbeddingModel
from onyx.utils.logger import setup_logger
from onyx.utils.timing import log_function_time
from shared_configs.configs import INDEXING_MODEL_SERVER_HOST
from shared_configs.configs import INDEXING_MODEL_SERVER_PORT
from shared_configs.enums import EmbeddingProvider
from shared_configs.enums import EmbedTextType
from shared_configs.model_server_models import Embedding


logger = setup_logger()


class IndexingEmbedder(ABC):
    """Converts chunks into chunks with embeddings. Note that one chunk may have
    multiple embeddings associated with it.
    
    将文档块转换为带有嵌入向量的块。注意一个块可能关联多个嵌入向量。
    
    这是一个抽象基类，定义了文档嵌入器的基本接口。负责将文本块转换为向量表示，
    支持不同的嵌入模型和配置选项。
    """

    def __init__(
        self,
        model_name: str,        # 模型名称
        normalize: bool,        # 是否规范化向量
        query_prefix: str | None,    # 查询前缀
        passage_prefix: str | None,  # 段落前缀
        provider_type: EmbeddingProvider | None,  # 提供商类型
        api_key: str | None,    # API密钥
        api_url: str | None,    # API地址
        api_version: str | None,  # API版本
        deployment_name: str | None,  # 部署名称
        callback: IndexingHeartbeatInterface | None,  # 回调接口
    ):
        """
        初始化嵌入器实例
        
        Args:
            model_name: 使用的模型名称
            normalize: 是否对向量进行归一化
            query_prefix: 查询文本的前缀
            passage_prefix: 段落文本的前缀
            provider_type: 嵌入服务提供商类型
            api_key: API访问密钥
            api_url: API服务地址
            api_version: API版本号
            deployment_name: 部署名称
            callback: 用于心跳检测的回调接口
        """
        self.model_name = model_name
        self.normalize = normalize
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix
        self.provider_type = provider_type
        self.api_key = api_key
        self.api_url = api_url
        self.api_version = api_version
        self.deployment_name = deployment_name

        self.embedding_model = EmbeddingModel(
            model_name=model_name,
            query_prefix=query_prefix,
            passage_prefix=passage_prefix,
            normalize=normalize,
            api_key=api_key,
            provider_type=provider_type,
            api_url=api_url,
            api_version=api_version,
            deployment_name=deployment_name,
            # The below are globally set, this flow always uses the indexing one
            server_host=INDEXING_MODEL_SERVER_HOST,
            server_port=INDEXING_MODEL_SERVER_PORT,
            retrim_content=True,
            callback=callback,
        )

    @abstractmethod
    def embed_chunks(
        self,
        chunks: list[DocAwareChunk],
    ) -> list[IndexChunk]:
        """
        将文档块转换为带有嵌入向量的索引块（抽象方法）
        
        Args:
            chunks: 待处理的文档块列表
            
        Returns:
            包含嵌入向量的索引块列表
        """
        raise NotImplementedError


class DefaultIndexingEmbedder(IndexingEmbedder):
    """
    IndexingEmbedder的默认实现类
    
    实现了文档块到向量的转换逻辑，包括：
    1. 处理文档块的标题、内容和元数据
    2. 支持大文档块和小文档块的处理
    3. 缓存标题向量以提高性能
    """

    def __init__(
        self,
        model_name: str,
        normalize: bool,
        query_prefix: str | None,
        passage_prefix: str | None,
        provider_type: EmbeddingProvider | None = None,
        api_key: str | None = None,
        api_url: str | None = None,
        api_version: str | None = None,
        deployment_name: str | None = None,
        callback: IndexingHeartbeatInterface | None = None,
    ):
        """初始化默认嵌入器实例，参数说明同父类"""
        super().__init__(
            model_name,
            normalize,
            query_prefix,
            passage_prefix,
            provider_type,
            api_key,
            api_url,
            api_version,
            deployment_name,
            callback,
        )

    @log_function_time()
    def embed_chunks(
        self,
        chunks: list[DocAwareChunk],
    ) -> list[IndexChunk]:
        """Adds embeddings to the chunks, the title and metadata suffixes are added to the chunk as well
        if they exist. If there is no space for it, it would have been thrown out at the chunking step.
        
        为文档块添加嵌入向量，如果存在标题和元数据后缀也会被添加到块中。
        如果没有足够空间，这些内容在分块步骤就会被丢弃。
        
        Args:
            chunks: 待处理的文档块列表
            
        Returns:
            处理后的索引块列表，每个块包含对应的嵌入向量
            
        Raises:
            ValueError: 当块没有内容时抛出
            RuntimeError: 当大块包含小块时抛出
        """
        # All chunks at this point must have some non-empty content
        # 此时所有的块都必须包含非空内容
        flat_chunk_texts: list[str] = []
        large_chunks_present = False
        for chunk in chunks:
            if (chunk.large_chunk_reference_ids):
                large_chunks_present = True
            chunk_text = (
                f"{chunk.title_prefix}{chunk.content}{chunk.metadata_suffix_semantic}"
            ) or chunk.source_document.get_title_for_document_index()

            if not chunk_text:
                # This should never happen, the document would have been dropped
                # before getting to this point
                raise ValueError(f"Chunk has no content: {chunk.to_short_descriptor()}")

            flat_chunk_texts.append(chunk_text)

            if chunk.mini_chunk_texts:
                if chunk.large_chunk_reference_ids:
                    # A large chunk does not contain mini chunks, if it matches the large chunk
                    # with a high score, then mini chunks would not be used anyway
                    # otherwise it should match the normal chunk
                    raise RuntimeError("Large chunk contains mini chunks")
                flat_chunk_texts.extend(chunk.mini_chunk_texts)

        embeddings = self.embedding_model.encode(
            texts=flat_chunk_texts,
            text_type=EmbedTextType.PASSAGE,
            large_chunks_present=large_chunks_present,
        )

        chunk_titles = {
            chunk.source_document.get_title_for_document_index() for chunk in chunks
        }

        # Drop any None or empty strings
        # 删除所有None或空字符串
        # If there is no title or the title is empty, the title embedding field will be null
        # 这没关系，它不会对评分产生任何影响
        chunk_titles_list = [title for title in chunk_titles if title]

        # Cache the Title embeddings to only have to do it once
        # 缓存标题嵌入向量，只需计算一次
        title_embed_dict: dict[str, Embedding] = {}
        if chunk_titles_list:
            title_embeddings = self.embedding_model.encode(
                chunk_titles_list, text_type=EmbedTextType.PASSAGE
            )
            title_embed_dict.update(
                {
                    title: vector
                    for title, vector in zip(chunk_titles_list, title_embeddings)
                }
            )

        # Mapping embeddings to chunks
        # 将嵌入向量映射到块
        embedded_chunks: list[IndexChunk] = []
        embedding_ind_start = 0
        for chunk in chunks:
            num_embeddings = 1 + (
                len(chunk.mini_chunk_texts) if chunk.mini_chunk_texts else 0
            )
            chunk_embeddings = embeddings[
                embedding_ind_start : embedding_ind_start + num_embeddings
            ]

            title = chunk.source_document.get_title_for_document_index()

            title_embedding = None
            if title:
                if title in title_embed_dict:
                    # Using cached value to avoid recalculating for every chunk
                    # 使用缓存值，避免为每个块重新计算
                    title_embedding = title_embed_dict[title]
                else:
                    logger.error(
                        "Title had to be embedded separately, this should not happen!"
                        # 标题必须单独嵌入，这种情况不应该发生！
                    )
                    title_embedding = self.embedding_model.encode(
                        [title], text_type=EmbedTextType.PASSAGE
                    )[0]
                    title_embed_dict[title] = title_embedding

            new_embedded_chunk = IndexChunk(
                **chunk.model_dump(),
                embeddings=ChunkEmbedding(
                    full_embedding=chunk_embeddings[0],
                    mini_chunk_embeddings=chunk_embeddings[1:],
                ),
                title_embedding=title_embedding,
            )
            embedded_chunks.append(new_embedded_chunk)
            embedding_ind_start += num_embeddings

        return embedded_chunks

    @classmethod
    def from_db_search_settings(
        cls,
        search_settings: SearchSettings,
        callback: IndexingHeartbeatInterface | None = None,
    ) -> "DefaultIndexingEmbedder":
        """
        从数据库搜索设置创建嵌入器实例
        
        Args:
            search_settings: 数据库中的搜索设置
            callback: 心跳检测回调接口
            
        Returns:
            配置好的DefaultIndexingEmbedder实例
        """
        return cls(
            model_name=search_settings.model_name,
            normalize=search_settings.normalize,
            query_prefix=search_settings.query_prefix,
            passage_prefix=search_settings.passage_prefix,
            provider_type=search_settings.provider_type,
            api_key=search_settings.api_key,
            api_url=search_settings.api_url,
            api_version=search_settings.api_version,
            deployment_name=search_settings.deployment_name,
            callback=callback,
        )
