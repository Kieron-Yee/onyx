"""
文件功能说明：
该文件实现了文档分块(chunking)的相关功能，用于将大型文档切分成较小的片段以便于索引和处理。
主要包括：
1. 文档分块的核心逻辑
2. 元数据处理
3. 大块文本和小块文本的生成
4. 文本清理和预处理
"""

from onyx.configs.app_configs import BLURB_SIZE
from onyx.configs.app_configs import LARGE_CHUNK_RATIO
from onyx.configs.app_configs import MINI_CHUNK_SIZE
from onyx.configs.app_configs import SKIP_METADATA_IN_CHUNK
from onyx.configs.constants import DocumentSource
from onyx.configs.constants import RETURN_SEPARATOR
from onyx.configs.constants import SECTION_SEPARATOR
from onyx.configs.model_configs import DOC_EMBEDDING_CONTEXT_SIZE
from onyx.connectors.cross_connector_utils.miscellaneous_utils import (
    get_metadata_keys_to_ignore,
)
from onyx.connectors.models import Document
from onyx.indexing.indexing_heartbeat import IndexingHeartbeatInterface
from onyx.indexing.models import DocAwareChunk
from onyx.natural_language_processing.utils import BaseTokenizer
from onyx.utils.logger import setup_logger
from onyx.utils.text_processing import clean_text
from onyx.utils.text_processing import shared_precompare_cleanup
from shared_configs.configs import STRICT_CHUNK_TOKEN_LIMIT

# Not supporting overlaps, we need a clean combination of chunks and it is unclear if overlaps
# actually help quality at all
# 不支持重叠，我们需要清晰的块组合，目前不清楚重叠是否真的有助于提高质量
CHUNK_OVERLAP = 0
# Fairly arbitrary numbers but the general concept is we don't want the title/metadata to
# overwhelm the actual contents of the chunk
# For example in a rare case, this could be 128 tokens for the 512 chunk and title prefix
# could be another 128 tokens leaving 256 for the actual contents
# 这些是相对任意的数字，但基本概念是我们不希望标题/元数据压倒块的实际内容
# 例如在极少数情况下，512个块中可能有128个标记用于标题前缀，另外128个标记用于元数据，
# 剩下256个用于实际内容
MAX_METADATA_PERCENTAGE = 0.25
CHUNK_MIN_CONTENT = 256


logger = setup_logger()


def _get_metadata_suffix_for_document_index(
    metadata: dict[str, str | list[str]], include_separator: bool = False
) -> tuple[str, str]:
    """
    从文档的元数据生成用于向量嵌入和关键词搜索的字符串表示。

    参数:
        metadata: 包含文档元数据的字典
        include_separator: 是否在结果中包含分隔符

    返回:
        tuple[str, str]: 返回两个字符串，第一个用于向量嵌入，第二个用于关键词搜索

    示例:
        对于以下元数据：
        {
            "author": "John Doe",
            "space": "Engineering"
        }
        向量嵌入字符串应包含键和值之间的关系，而关键词搜索只需要值（John Doe和Engineering）。
        键是重复且更加嘈杂的。
    """
    if not metadata:
        return "", ""

    metadata_str = "Metadata:\n"
    metadata_values = []
    for key, value in metadata.items():
        if key in get_metadata_keys_to_ignore():
            continue

        value_str = ", ".join(value) if isinstance(value, list) else value

        if isinstance(value, list):
            metadata_values.extend(value)
        else:
            metadata_values.append(value)

        metadata_str += f"\t{key} - {value_str}\n"

    metadata_semantic = metadata_str.strip()
    metadata_keyword = " ".join(metadata_values)

    if include_separator:
        return RETURN_SEPARATOR + metadata_semantic, RETURN_SEPARATOR + metadata_keyword
    return metadata_semantic, metadata_keyword


def _combine_chunks(chunks: list[DocAwareChunk], large_chunk_id: int) -> DocAwareChunk:
    """
    合并多个小块为一个大块。

    参数:
        chunks: 要合并的块列表
        large_chunk_id: 大块的ID

    返回:
        DocAwareChunk: 合并后的大块
    """
    merged_chunk = DocAwareChunk(
        source_document=chunks[0].source_document,
        chunk_id=chunks[0].chunk_id,
        blurb=chunks[0].blurb,
        content=chunks[0].content,
        source_links=chunks[0].source_links or {},
        section_continuation=(chunks[0].chunk_id > 0),
        title_prefix=chunks[0].title_prefix,
        metadata_suffix_semantic=chunks[0].metadata_suffix_semantic,
        metadata_suffix_keyword=chunks[0].metadata_suffix_keyword,
        large_chunk_reference_ids=[chunk.chunk_id for chunk in chunks],
        mini_chunk_texts=None,
        large_chunk_id=large_chunk_id,
    )

    offset = 0
    for i in range(1, len(chunks)):
        merged_chunk.content += SECTION_SEPARATOR + chunks[i].content

        offset += len(SECTION_SEPARATOR) + len(chunks[i - 1].content)
        for link_offset, link_text in (chunks[i].source_links or {}).items():
            if merged_chunk.source_links is None:
                merged_chunk.source_links = {}
            merged_chunk.source_links[link_offset + offset] = link_text

    return merged_chunk


def generate_large_chunks(chunks: list[DocAwareChunk]) -> list[DocAwareChunk]:
    """
    根据LARGE_CHUNK_RATIO生成大块文本。

    参数:
        chunks: 小块文本列表

    返回:
        list[DocAwareChunk]: 生成的大块文本列表
    """
    large_chunks = []
    for idx, i in enumerate(range(0, len(chunks), LARGE_CHUNK_RATIO)):
        chunk_group = chunks[i : i + LARGE_CHUNK_RATIO]
        if len(chunk_group) > 1:
            large_chunk = _combine_chunks(chunk_group, idx)
            large_chunks.append(large_chunk)
    return large_chunks


class Chunker:
    """
    文档分块器类，用于将文档分割成更小的块以便索引。

    主要功能：
    1. 将大型文档切分成小块
    2. 处理文档的元数据
    3. 生成文档摘要
    4. 支持多级分块（大块和小块）
    """

    def __init__(
        self,
        tokenizer: BaseTokenizer,
        enable_multipass: bool = False,
        enable_large_chunks: bool = False,
        blurb_size: int = BLURB_SIZE,
        include_metadata: bool = not SKIP_METADATA_IN_CHUNK,
        chunk_token_limit: int = DOC_EMBEDDING_CONTEXT_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
        mini_chunk_size: int = MINI_CHUNK_SIZE,
        callback: IndexingHeartbeatInterface | None = None,
    ) -> None:
        """
        初始化分块器。

        参数:
            tokenizer: 分词器
            enable_multipass: 是否启用多遍处理
            enable_large_chunks: 是否启用大块生成
            blurb_size: 摘要大小
            include_metadata: 是否在块中包含元数据
            chunk_token_limit: 块的标记限制
            chunk_overlap: 块之间的重叠量
            mini_chunk_size: 小块的大小
            callback: 用于处理进度回调的接口
        """
        from llama_index.text_splitter import SentenceSplitter

        self.include_metadata = include_metadata
        self.chunk_token_limit = chunk_token_limit
        self.enable_multipass = enable_multipass
        self.enable_large_chunks = enable_large_chunks
        self.tokenizer = tokenizer
        self.callback = callback

        self.blurb_splitter = SentenceSplitter(
            tokenizer=tokenizer.tokenize,
            chunk_size=blurb_size,
            chunk_overlap=0,
        )

        self.chunk_splitter = SentenceSplitter(
            tokenizer=tokenizer.tokenize,
            chunk_size=chunk_token_limit,
            chunk_overlap=chunk_overlap,
        )

        self.mini_chunk_splitter = (
            SentenceSplitter(
                tokenizer=tokenizer.tokenize,
                chunk_size=mini_chunk_size,
                chunk_overlap=0,
            )
            if enable_multipass
            else None
        )

    def _split_oversized_chunk(self, text: str, content_token_limit: int) -> list[str]:
        """
        将超大文本块按照标记数量限制分割成更小的块。

        参数:
            text: 要分割的文本
            content_token_limit: 内容标记数限制

        返回:
            list[str]: 分割后的文本块列表

        处理过程:
            1. 对文本进行分词
            2. 按照token限制分割文本
            3. 重新组合分割后的文本
        """
        tokens = self.tokenizer.tokenize(text)
        chunks = []
        start = 0
        total_tokens = len(tokens)
        while start < total_tokens:
            end = min(start + content_token_limit, total_tokens)
            token_chunk = tokens[start:end]
            # Join the tokens to reconstruct the text
            # 连接标记以重建文本
            chunk_text = " ".join(token_chunk)
            chunks.append(chunk_text)
            start = end
        return chunks

    def _extract_blurb(self, text: str) -> str:
        """
        从文本中提取摘要。

        参数:
            text: 源文本

        返回:
            str: 提取的摘要文本（通常是第一个句子）
        """
        texts = self.blurb_splitter.split_text(text)
        if not texts:
            return ""
        return texts[0]

    def _get_mini_chunk_texts(self, chunk_text: str) -> list[str] | None:
        """
        生成小型文本块。

        参数:
            chunk_text: 要处理的文本块

        返回:
            list[str] | None: 小型文本块列表，如果未启用多遍处理则返回None
        """
        if self.mini_chunk_splitter and chunk_text.strip():
            return self.mini_chunk_splitter.split_text(chunk_text)
        return None

    def _chunk_document(
        self,
        document: Document,
        title_prefix: str,
        metadata_suffix_semantic: str,
        metadata_suffix_keyword: str,
        content_token_limit: int,
    ) -> list[DocAwareChunk]:
        """
        将文档分割成小块，并添加元数据。

        参数:
            document: 要分割的文档
            title_prefix: 标题前缀
            metadata_suffix_semantic: 用于向量嵌入的元数据后缀
            metadata_suffix_keyword: 用于关键词搜索的元数据后缀
            content_token_limit: 内容标记数限制

        返回:
            list[DocAwareChunk]: 分割后的文档块列表

        处理流程:
            1. 初始化文档块列表
            2. 遍历文档的每个部分
            3. 处理大型部分（超过token限制的部分）
            4. 处理小型部分（合并或创建新块）
            5. 处理剩余内容
        """
        chunks: list[DocAwareChunk] = []
        link_offsets: dict[int, str] = {}
        chunk_text = ""

        def _create_chunk(
            text: str,
            links: dict[int, str],
            is_continuation: bool = False,
        ) -> DocAwareChunk:
            """
            创建新的文档块。

            参数:
                text: 块的内容文本
                links: 链接偏移量和文本的映射
                is_continuation: 是否是前一个块的继续

            返回:
                DocAwareChunk: 创建的文档块
            """
            return DocAwareChunk(
                source_document=document,
                chunk_id=len(chunks),
                blurb=self._extract_blurb(text),
                content=text,
                source_links=links or {0: ""},
                section_continuation=is_continuation,
                title_prefix=title_prefix,
                metadata_suffix_semantic=metadata_suffix_semantic,
                metadata_suffix_keyword=metadata_suffix_keyword,
                mini_chunk_texts=self._get_mini_chunk_texts(text),
                large_chunk_id=None,
            )

        for section_idx, section in enumerate(document.sections):
            section_text = clean_text(section.text)
            section_link_text = section.link or ""
            # If there is no useful content, not even the title, just drop it
            # 如果没有有用的内容（甚至标题都没有），就直接丢弃
            if not section_text and (not document.title or section_idx > 0):
                # If a section is empty and the document has no title, we can just drop it. We return a list of
                # DocAwareChunks where each one contains the necessary information needed down the line for indexing.
                # There is no concern about dropping whole documents from this list, it should not cause any indexing failures.
                # 如果一个部分是空的且文档没有标题，我们可以直接丢弃它。我们返回一个DocAwareChunks列表，
                # 其中每个块都包含后续索引所需的必要信息。
                # 从这个列表中删除整个文档不会导致任何索引失败。
                logger.warning(
                    f"Skipping section {section.text} from document "
                    f"{document.semantic_identifier} due to empty text after cleaning "
                    f"with link {section_link_text}"
                )
                continue

            section_token_count = len(self.tokenizer.tokenize(section_text))

            # Large sections are considered self-contained/unique
            # Therefore, they start a new chunk and are not concatenated
            # at the end by other sections
            # 大型部分被视为独立/唯一的
            # 因此，它们开始一个新的块，不会在末尾与其他部分连接
            if section_token_count > content_token_limit:
                if chunk_text:
                    chunks.append(_create_chunk(chunk_text, link_offsets))
                    link_offsets = {}
                    chunk_text = ""

                split_texts = self.chunk_splitter.split_text(section_text)

                for i, split_text in enumerate(split_texts):
                    if (
                        STRICT_CHUNK_TOKEN_LIMIT
                        and
                        # Tokenizer only runs if STRICT_CHUNK_TOKEN_LIMIT is true
                        # 只有在STRICT_CHUNK_TOKEN_LIMIT为true时才运行分词器
                        len(self.tokenizer.tokenize(split_text)) > content_token_limit
                    ):
                        # If STRICT_CHUNK_TOKEN_LIMIT is true, manually check
                        # the token count of each split text to ensure it is
                        # not larger than the content_token_limit
                        # 如果STRICT_CHUNK_TOKEN_LIMIT为true，手动检查每个分割文本的标记数
                        # 以确保它不大于content_token_limit
                        smaller_chunks = self._split_oversized_chunk(
                            split_text, content_token_limit
                        )
                        for i, small_chunk in enumerate(smaller_chunks):
                            chunks.append(
                                _create_chunk(
                                    text=small_chunk,
                                    links={0: section_link_text},
                                    is_continuation=(i != 0),
                                )
                            )
                    else:
                        chunks.append(
                            _create_chunk(
                                text=split_text,
                                links={0: section_link_text},
                                is_continuation=(i != 0),
                            )
                        )

                continue

            current_token_count = len(self.tokenizer.tokenize(chunk_text))
            current_offset = len(shared_precompare_cleanup(chunk_text))
            # In the case where the whole section is shorter than a chunk, either add
            # to chunk or start a new one
            # 在整个部分短于一个块的情况下，要么添加到当前块，要么开始一个新块
            next_section_tokens = (
                len(self.tokenizer.tokenize(SECTION_SEPARATOR)) + section_token_count
            )
            if next_section_tokens + current_token_count <= content_token_limit:
                if chunk_text:
                    chunk_text += SECTION_SEPARATOR
                chunk_text += section_text
                link_offsets[current_offset] = section_link_text
            else:
                chunks.append(_create_chunk(chunk_text, link_offsets))
                link_offsets = {0: section_link_text}
                chunk_text = section_text

        # Once we hit the end, if we're still in the process of building a chunk, add what we have.
        # If there is only whitespace left then don't include it. If there are no chunks at all
        # from the doc, we can just create a single chunk with the title.
        # 当我们到达末尾时，如果我们仍在构建块的过程中，添加我们所拥有的内容。
        # 如果只剩下空白，则不包括它。如果文档中根本没有块，
        # 我们可以只用标题创建一个块。
        if chunk_text.strip() or not chunks:
            chunks.append(
                _create_chunk(
                    chunk_text,
                    link_offsets or {0: section_link_text},
                )
            )

        # If the chunk does not have any useable content, it will not be indexed
        return chunks

    def _handle_single_document(self, document: Document) -> list[DocAwareChunk]:
        """
        处理单个文档，生成文档块。

        参数:
            document: 需要处理的文档

        返回:
            list[DocAwareChunk]: 处理后的文档块列表

        说明:
            - 提取文档标题
            - 处理元数据
            - 计算token限制
            - 生成普通块和大块（如果启用）
        """
        # Specifically for reproducing an issue with gmail
        # 专门用于重现Gmail相关的问题
        if document.source == DocumentSource.GMAIL:
            logger.debug(f"Chunking {document.semantic_identifier}")

        title = self._extract_blurb(document.get_title_for_document_index() or "")
        title_prefix = title + RETURN_SEPARATOR if title else ""
        title_tokens = len(self.tokenizer.tokenize(title_prefix))

        metadata_suffix_semantic = ""
        metadata_suffix_keyword = ""
        metadata_tokens = 0
        if self.include_metadata:
            (
                metadata_suffix_semantic,
                metadata_suffix_keyword,
            ) = _get_metadata_suffix_for_document_index(
                document.metadata, include_separator=True
            )
            metadata_tokens = len(self.tokenizer.tokenize(metadata_suffix_semantic))

        if metadata_tokens >= self.chunk_token_limit * MAX_METADATA_PERCENTAGE:
            # Note: we can keep the keyword suffix even if the semantic suffix is too long to fit in the model
            # context, there is no limit for the keyword component
            # 注意：即使语义后缀太长而无法适应模型上下文，我们仍可以保留关键词后缀，
            # 因为关键词组件没有限制
            metadata_suffix_semantic = ""
            metadata_tokens = 0

        content_token_limit = self.chunk_token_limit - title_tokens - metadata_tokens
        # If there is not enough context remaining then just index the chunk with no prefix/suffix
        # 如果没有足够的剩余上下文空间，则直接索引没有前缀/后缀的块
        if content_token_limit <= CHUNK_MIN_CONTENT:
            content_token_limit = self.chunk_token_limit
            title_prefix = ""
            metadata_suffix_semantic = ""

        normal_chunks = self._chunk_document(
            document,
            title_prefix,
            metadata_suffix_semantic,
            metadata_suffix_keyword,
            content_token_limit,
        )

        if self.enable_multipass and self.enable_large_chunks:
            large_chunks = generate_large_chunks(normal_chunks)
            normal_chunks.extend(large_chunks)

        return normal_chunks

    def chunk(self, documents: list[Document]) -> list[DocAwareChunk]:
        """
        将文档列表分割成更小的块以进行索引，同时保留文档元数据。

        参数:
            documents: 需要处理的文档列表

        返回:
            list[DocAwareChunk]: 所有生成的文档块列表

        处理流程:
            1. 遍历文档列表
            2. 检查是否需要停止处理
            3. 处理单个文档
            4. 更新处理进度
        """
        final_chunks: list[DocAwareChunk] = []
        for document in documents:
            if self.callback:
                if self.callback.should_stop():
                    raise RuntimeError("Chunker.chunk: Stop signal detected")

            chunks = self._handle_single_document(document)
            final_chunks.extend(chunks)

            if self.callback:
                self.callback.progress("Chunker.chunk", len(chunks))

        return final_chunks
