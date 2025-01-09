import asyncio
import json
import time
from types import TracebackType
from typing import cast
from typing import Optional

import httpx
import openai
import vertexai  # type: ignore
import voyageai  # type: ignore
from cohere import AsyncClient as CohereAsyncClient
from fastapi import APIRouter
from fastapi import HTTPException
from google.oauth2 import service_account  # type: ignore
from litellm import aembedding
from litellm.exceptions import RateLimitError
from retry import retry
from sentence_transformers import CrossEncoder  # type: ignore
from sentence_transformers import SentenceTransformer  # type: ignore
from vertexai.language_models import TextEmbeddingInput  # type: ignore
from vertexai.language_models import TextEmbeddingModel  # type: ignore

from model_server.constants import DEFAULT_COHERE_MODEL
from model_server.constants import DEFAULT_OPENAI_MODEL
from model_server.constants import DEFAULT_VERTEX_MODEL
from model_server.constants import DEFAULT_VOYAGE_MODEL
from model_server.constants import EmbeddingModelTextType
from model_server.constants import EmbeddingProvider
from model_server.utils import simple_log_function_time
from onyx.utils.logger import setup_logger
from shared_configs.configs import API_BASED_EMBEDDING_TIMEOUT
from shared_configs.configs import INDEXING_ONLY
from shared_configs.configs import OPENAI_EMBEDDING_TIMEOUT
from shared_configs.enums import EmbedTextType
from shared_configs.enums import RerankerProvider
from shared_configs.model_server_models import Embedding
from shared_configs.model_server_models import EmbedRequest
from shared_configs.model_server_models import EmbedResponse
from shared_configs.model_server_models import RerankRequest
from shared_configs.model_server_models import RerankResponse
from shared_configs.utils import batch_list


logger = setup_logger()

router = APIRouter(prefix="/encoder")

_GLOBAL_MODELS_DICT: dict[str, "SentenceTransformer"] = {}
_RERANK_MODEL: Optional["CrossEncoder"] = None

# If we are not only indexing, dont want retry very long
# 如果我们不仅仅是索引，不希望重试时间太长
_RETRY_DELAY = 10 if INDEXING_ONLY else 0.1
_RETRY_TRIES = 10 if INDEXING_ONLY else 2

# OpenAI only allows 2048 embeddings to be computed at once
# OpenAI 只允许一次计算2048个嵌入
_OPENAI_MAX_INPUT_LEN = 2048
# Cohere allows up to 96 embeddings in a single embedding calling
# Cohere 允许在一次嵌入调用中最多96个嵌入
_COHERE_MAX_INPUT_LEN = 96


# CloudEmbedding class for handling cloud-based embeddings
# CloudEmbedding 类用于处理基于云的嵌入
class CloudEmbedding:
    def __init__(
        self,
        api_key: str,
        provider: EmbeddingProvider,
        api_url: str | None = None,
        api_version: str | None = None,
        timeout: int = API_BASED_EMBEDDING_TIMEOUT,
    ) -> None:
        self.provider = provider
        self.api_key = api_key
        self.api_url = api_url
        self.api_version = api_version
        self.timeout = timeout
        self.http_client = httpx.AsyncClient(timeout=timeout)
        self._closed = False

    async def _embed_openai(
        self, texts: list[str], model: str | None
    ) -> list[Embedding]:
        if not model:
            model = DEFAULT_OPENAI_MODEL

        # Use the OpenAI specific timeout for this one
        # 使用OpenAI特定的超时
        client = openai.AsyncOpenAI(
            api_key=self.api_key, timeout=OPENAI_EMBEDDING_TIMEOUT
        )

        final_embeddings: list[Embedding] = []
        try:
            for text_batch in batch_list(texts, _OPENAI_MAX_INPUT_LEN):
                response = await client.embeddings.create(input=text_batch, model=model)
                final_embeddings.extend(
                    [embedding.embedding for embedding in response.data]
                )
            return final_embeddings
        except Exception as e:
            error_string = (
                f"Error embedding text with OpenAI: {str(e)} \n"
                f"Model: {model} \n"
                f"Provider: {self.provider} \n"
                f"Texts: {texts}"
            )
            logger.error(error_string)
            # 记录错误信息
            raise RuntimeError(error_string)

    async def _embed_cohere(
        self, texts: list[str], model: str | None, embedding_type: str
    ) -> list[Embedding]:
        if not model:
            model = DEFAULT_COHERE_MODEL

        client = CohereAsyncClient(api_key=self.api_key)

        final_embeddings: list[Embedding] = []
        for text_batch in batch_list(texts, _COHERE_MAX_INPUT_LEN):
            # Does not use the same tokenizer as the Onyx API server but it's approximately the same
            # empirically it's only off by a very few tokens so it's not a big deal
            # 不使用与Onyx API服务器相同的分词器，但大致相同，经验上只差几个标记，所以问题不大
            response = await client.embed(
                texts=text_batch,
                model=model,
                input_type=embedding_type,
                truncate="END",
            )
            final_embeddings.extend(cast(list[Embedding], response.embeddings))
        return final_embeddings

    async def _embed_voyage(
        self, texts: list[str], model: str | None, embedding_type: str
    ) -> list[Embedding]:
        if not model:
            model = DEFAULT_VOYAGE_MODEL

        client = voyageai.AsyncClient(
            api_key=self.api_key, timeout=API_BASED_EMBEDDING_TIMEOUT
        )

        response = await client.embed(
            texts=texts,
            model=model,
            input_type=embedding_type,
            truncation=True,
        )

        return response.embeddings

    async def _embed_azure(
        self, texts: list[str], model: str | None
    ) -> list[Embedding]:
        response = await aembedding(
            model=model,
            input=texts,
            timeout=API_BASED_EMBEDDING_TIMEOUT,
            api_key=self.api_key,
            api_base=self.api_url,
            api_version=self.api_version,
        )
        embeddings = [embedding["embedding"] for embedding in response.data]
        return embeddings

    async def _embed_vertex(
        self, texts: list[str], model: str | None, embedding_type: str
    ) -> list[Embedding]:
        if not model:
            model = DEFAULT_VERTEX_MODEL

        credentials = service_account.Credentials.from_service_account_info(
            json.loads(self.api_key)
        )
        project_id = json.loads(self.api_key)["project_id"]
        vertexai.init(project=project_id, credentials=credentials)
        client = TextEmbeddingModel.from_pretrained(model)

        embeddings = await client.get_embeddings_async(
            [
                TextEmbeddingInput(
                    text,
                    embedding_type,
                )
                for text in texts
            ],
            auto_truncate=True,  # This is the default
            # 自动截断=True，这是默认值
        )
        return [embedding.values for embedding in embeddings]

    async def _embed_litellm_proxy(
        self, texts: list[str], model_name: str | None
    ) -> list[Embedding]:
        if not model_name:
            raise ValueError("Model name is required for LiteLLM proxy embedding.")

        if not self.api_url:
            raise ValueError("API URL is required for LiteLLM proxy embedding.")

        headers = (
            {} if not self.api_key else {"Authorization": f"Bearer {self.api_key}"}
        )

        response = await self.http_client.post(
            self.api_url,
            json={
                "model": model_name,
                "input": texts,
            },
            headers=headers,
        )
        response.raise_for_status()
        result = response.json()
        return [embedding["embedding"] for embedding in result["data"]]

    @retry(tries=_RETRY_TRIES, delay=_RETRY_DELAY)
    async def embed(
        self,
        *,
        texts: list[str],
        text_type: EmbedTextType,
        model_name: str | None = None,
        deployment_name: str | None = None,
    ) -> list[Embedding]:
        if self.provider == EmbeddingProvider.OPENAI:
            return await self._embed_openai(texts, model_name)
        elif self.provider == EmbeddingProvider.AZURE:
            return await self._embed_azure(texts, f"azure/{deployment_name}")
        elif self.provider == EmbeddingProvider.LITELLM:
            return await self._embed_litellm_proxy(texts, model_name)

        embedding_type = EmbeddingModelTextType.get_type(self.provider, text_type)
        if self.provider == EmbeddingProvider.COHERE:
            return await self._embed_cohere(texts, model_name, embedding_type)
        elif self.provider == EmbeddingProvider.VOYAGE:
            return await self._embed_voyage(texts, model_name, embedding_type)
        elif self.provider == EmbeddingProvider.GOOGLE:
            return await self._embed_vertex(texts, model_name, embedding_type)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    @staticmethod
    def create(
        api_key: str,
        provider: EmbeddingProvider,
        api_url: str | None = None,
        api_version: str | None = None,
    ) -> "CloudEmbedding":
        logger.debug(f"Creating Embedding instance for provider: {provider}")  # 创建提供者的嵌入实例
        return CloudEmbedding(api_key, provider, api_url, api_version)

    async def aclose(self) -> None:
        """Explicitly close the client."""
        # 显式关闭客户端
        if not self._closed:
            await self.http_client.aclose()
            self._closed = True

    async def __aenter__(self) -> "CloudEmbedding":
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self.aclose()

    def __del__(self) -> None:
        """Finalizer to warn about unclosed clients."""
        # 终结器，警告未关闭的客户端
        if not self._closed:
            logger.warning(
                "CloudEmbedding was not properly closed. Use 'async with' or call aclose()"
                # CloudEmbedding未正确关闭。使用'async with'或调用aclose()
            )


# Function to get embedding model
# 获取嵌入模型的函数
def get_embedding_model(
    model_name: str,
    max_context_length: int,
) -> "SentenceTransformer":
    from sentence_transformers import SentenceTransformer  # type: ignore

    global _GLOBAL_MODELS_DICT  # A dictionary to store models

    if model_name not in _GLOBAL_MODELS_DICT:
        logger.notice(f"Loading {model_name}")  # 正在加载指定模型
        # 记录加载模型的信息
        # Some model architectures that aren't built into the Transformers or Sentence
        # Transformer need to be downloaded to be loaded locally. This does not mean
        # data is sent to remote servers for inference, however the remote code can
        # be fairly arbitrary so only use trusted models
        # 一些模型架构没有内置在Transformers或Sentence Transformer中，需要下载以在本地加载。
        # 这并不意味着数据会被发送到远程服务器进行推理，然而远程代码可能是相当随意的，所以只使用可信的模型
        model = SentenceTransformer(
            model_name_or_path=model_name,
            trust_remote_code=True,
        )
        model.max_seq_length = max_context_length
        _GLOBAL_MODELS_DICT[model_name] = model
    elif max_context_length != _GLOBAL_MODELS_DICT[model_name].max_seq_length:
        _GLOBAL_MODELS_DICT[model_name].max_seq_length = max_context_length

    return _GLOBAL_MODELS_DICT[model_name]


# Function to get local reranking model
# 获取本地重排序模型的函数
def get_local_reranking_model(
    model_name: str,
) -> CrossEncoder:
    global _RERANK_MODEL
    if _RERANK_MODEL is None:
        logger.notice(f"Loading {model_name}")  # 正在加载指定模型
        # 记录加载模型的信息
        model = CrossEncoder(model_name)
        _RERANK_MODEL = model
    return _RERANK_MODEL


@simple_log_function_time()
# Function to embed text
# 嵌入文本的函数
async def embed_text(
    texts: list[str],
    text_type: EmbedTextType,
    model_name: str | None,
    deployment_name: str | None,
    max_context_length: int,
    normalize_embeddings: bool,
    api_key: str | None,
    provider_type: EmbeddingProvider | None,
    prefix: str | None,
    api_url: str | None,
    api_version: str | None,
) -> list[Embedding]:
    if not all(texts):
        logger.error("Empty strings provided for embedding")  # 提供了空字符串进行嵌入
        # 记录提供了空字符串的错误信息
        raise ValueError("Empty strings are not allowed for embedding.")

    if not texts:
        logger.error("No texts provided for embedding")  # 未提供要嵌入的文本
        # 记录未提供文本的错误信息
        raise ValueError("No texts provided for embedding.")

    start = time.monotonic()

    total_chars = 0
    for text in texts:
        total_chars += len(text)

    if provider_type is not None:
        logger.info(
            f"Embedding {len(texts)} texts with {total_chars} total characters with provider: {provider_type}"
        )  # 使用提供者嵌入指定数量的文本，包含指定数量的字符
        # 记录使用提供者嵌入的信息

        if api_key is None:
            logger.error("API key not provided for cloud model")  # 未为云模型提供API密钥
            # 记录未提供API密钥的错误信息
            raise RuntimeError("API key not provided for cloud model")

        if prefix:
            logger.warning("Prefix provided for cloud model, which is not supported")  # 为云模型提供了前缀，但这不被支持
            # 记录提供了不支持的前缀的警告信息
            raise ValueError(
                "Prefix string is not valid for cloud models. "
                "Cloud models take an explicit text type instead."
            )

        async with CloudEmbedding(
            api_key=api_key,
            provider=provider_type,
            api_url=api_url,
            api_version=api_version,
        ) as cloud_model:
            embeddings = await cloud_model.embed(
                texts=texts,
                model_name=model_name,
                deployment_name=deployment_name,
                text_type=text_type,
            )

        if any(embedding is None for embedding in embeddings):
            error_message = "Embeddings contain None values\n"
            error_message += "Corresponding texts:\n"
            error_message += "\n".join(texts)
            logger.error(error_message)
            # 记录错误信息
            raise ValueError(error_message)

        elapsed = time.monotonic() - start
        logger.info(
            f"Successfully embedded {len(texts)} texts with {total_chars} total characters "
            f"with provider {provider_type} in {elapsed:.2f}"
        )  # 使用提供者成功嵌入指定数量的文本，包含指定数量的字符，耗时指定秒数
        # 记录成功嵌入的信息
    elif model_name is not None:
        logger.info(
            f"Embedding {len(texts)} texts with {total_chars} total characters with local model: {model_name}"
        )  # 使用本地模型嵌入指定数量的文本，包含指定数量的字符
        # 记录使用本地模型嵌入的信息

        prefixed_texts = [f"{prefix}{text}" for text in texts] if prefix else texts

        local_model = get_embedding_model(
            model_name=model_name, max_context_length=max_context_length
        )
        # Run CPU-bound embedding in a thread pool
        embeddings_vectors = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: local_model.encode(
                prefixed_texts, normalize_embeddings=normalize_embeddings
            ),
        )
        embeddings = [
            embedding if isinstance(embedding, list) else embedding.tolist()
            for embedding in embeddings_vectors
        ]

        elapsed = time.monotonic() - start
        logger.info(
            f"Successfully embedded {len(texts)} texts with {total_chars} total characters "
            f"with local model {model_name} in {elapsed:.2f}"
        )
        # 记录成功嵌入的信息
    else:
        logger.error("Neither model name nor provider specified for embedding")  # 未指定模型名称或提供者进行嵌入
        # 记录未指定模型名称或提供者的错误信息
        raise ValueError(
            "Either model name or provider must be provided to run embeddings."
        )

    return embeddings


@simple_log_function_time()
# Function to perform local reranking
# 执行本地重排序的函数
async def local_rerank(query: str, docs: list[str], model_name: str) -> list[float]:
    cross_encoder = get_local_reranking_model(model_name)
    # Run CPU-bound reranking in a thread pool
    return await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: cross_encoder.predict([(query, doc) for doc in docs]).tolist(),  # type: ignore
    )


# Function to perform Cohere reranking
# 执行Cohere重排序的函数
async def cohere_rerank(
    query: str, docs: list[str], model_name: str, api_key: str
) -> list[float]:
    cohere_client = CohereAsyncClient(api_key=api_key)
    response = await cohere_client.rerank(query=query, documents=docs, model=model_name)
    results = response.results
    sorted_results = sorted(results, key=lambda item: item.index)
    return [result.relevance_score for result in sorted_results]


# Function to perform LiteLLM reranking
# 执行LiteLLM重排序的函数
async def litellm_rerank(
    query: str, docs: list[str], api_url: str, model_name: str, api_key: str | None
) -> list[float]:
    headers = {} if not api_key else {"Authorization": f"Bearer {api_key}"}
    async with httpx.AsyncClient() as client:
        response = await client.post(
            api_url,
            json={
                "model": model_name,
                "query": query,
                "documents": docs,
            },
            headers=headers,
        )
        response.raise_for_status()
        result = response.json()
        return [
            item["relevance_score"]
            for item in sorted(result["results"], key=lambda x: x["index"])
        ]


@router.post("/bi-encoder-embed")
# Endpoint to process embed request
# 处理嵌入请求的端点
async def process_embed_request(
    embed_request: EmbedRequest,
) -> EmbedResponse:
    if not embed_request.texts:
        raise HTTPException(status_code=400, detail="No texts to be embedded")

    if not all(embed_request.texts):
        raise ValueError("Empty strings are not allowed for embedding.")

    try:
        if embed_request.text_type == EmbedTextType.QUERY:
            prefix = embed_request.manual_query_prefix
        elif embed_request.text_type == EmbedTextType.PASSAGE:
            prefix = embed_request.manual_passage_prefix
        else:
            prefix = None

        embeddings = await embed_text(
            texts=embed_request.texts,
            model_name=embed_request.model_name,
            deployment_name=embed_request.deployment_name,
            max_context_length=embed_request.max_context_length,
            normalize_embeddings=embed_request.normalize_embeddings,
            api_key=embed_request.api_key,
            provider_type=embed_request.provider_type,
            text_type=embed_request.text_type,
            api_url=embed_request.api_url,
            api_version=embed_request.api_version,
            prefix=prefix,
        )
        return EmbedResponse(embeddings=embeddings)
    except RateLimitError as e:
        raise HTTPException(
            status_code=429,
            detail=str(e),
        )
    except Exception as e:
        logger.exception(
            f"Error during embedding process: provider={embed_request.provider_type} model={embed_request.model_name}"
        )  # 嵌入过程发生错误：提供者和模型信息
        # 记录嵌入过程中的异常信息
        raise HTTPException(
            status_code=500, detail=f"Error during embedding process: {e}"
        )


@router.post("/cross-encoder-scores")
# Endpoint to process rerank request
# 处理重排序请求的端点
async def process_rerank_request(rerank_request: RerankRequest) -> RerankResponse:
    """Cross encoders can be purely black box from the app perspective"""
    # 从应用程序的角度来看，交叉编码器可以完全是黑盒
    if INDEXING_ONLY:
        raise RuntimeError("Indexing model server should not call intent endpoint")
        # 索引模型服务器不应调用意图端点

    if not rerank_request.documents or not rerank_request.query:
        raise HTTPException(
            status_code=400, detail="Missing documents or query for reranking"
        )
    if not all(rerank_request.documents):
        raise ValueError("Empty documents cannot be reranked.")

    try:
        if rerank_request.provider_type is None:
            sim_scores = await local_rerank(
                query=rerank_request.query,
                docs=rerank_request.documents,
                model_name=rerank_request.model_name,
            )
            return RerankResponse(scores=sim_scores)
        elif rerank_request.provider_type == RerankerProvider.LITELLM:
            if rerank_request.api_url is None:
                raise ValueError("API URL is required for LiteLLM reranking.")

            sim_scores = await litellm_rerank(
                query=rerank_request.query,
                docs=rerank_request.documents,
                api_url=rerank_request.api_url,
                model_name=rerank_request.model_name,
                api_key=rerank_request.api_key,
            )

            return RerankResponse(scores=sim_scores)

        elif rerank_request.provider_type == RerankerProvider.COHERE:
            if rerank_request.api_key is None:
                raise RuntimeError("Cohere Rerank Requires an API Key")
            sim_scores = await cohere_rerank(
                query=rerank_request.query,
                docs=rerank_request.documents,
                model_name=rerank_request.model_name,
                api_key=rerank_request.api_key,
            )
            return RerankResponse(scores=sim_scores)
        else:
            raise ValueError(f"Unsupported provider: {rerank_request.provider_type}")
    except Exception as e:
        logger.exception(f"Error during reranking process:\n{str(e)}")  # 重排序过程发生错误
        # 记录重排序过程中的异常信息
        raise HTTPException(
            status_code=500, detail="Failed to run Cross-Encoder reranking"
        )
