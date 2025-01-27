"""
文档处理路由模块
本模块提供了文档相关的API端点，包括获取文档信息和文档块信息的功能
"""

from fastapi import APIRouter
from fastapi import Depends
from fastapi import HTTPException
from fastapi import Query
from sqlalchemy.orm import Session

from onyx.auth.users import current_user
from onyx.context.search.models import IndexFilters
from onyx.context.search.preprocessing.access_filters import (
    build_access_filters_for_user,
)
from onyx.db.engine import get_session
from onyx.db.models import User
from onyx.db.search_settings import get_current_search_settings
from onyx.document_index.factory import get_default_document_index
from onyx.document_index.interfaces import VespaChunkRequest
from onyx.natural_language_processing.utils import get_tokenizer
from onyx.prompts.prompt_utils import build_doc_context_str
from onyx.server.documents.models import ChunkInfo
from onyx.server.documents.models import DocumentInfo


router = APIRouter(prefix="/document")


# 必须使用查询参数，因为FastAPI将URL类型的document_ids解释为不同的路径
@router.get("/document-size-info")
def get_document_info(
    document_id: str = Query(...),
    user: User | None = Depends(current_user),
    db_session: Session = Depends(get_session),
) -> DocumentInfo:
    """
    获取文档大小信息的API端点

    参数:
        document_id: 文档ID
        user: 当前用户对象
        db_session: 数据库会话

    返回:
        DocumentInfo: 包含文档块数量和令牌数量的信息对象

    异常:
        HTTPException: 当文档未找到时抛出404错误
    """
    search_settings = get_current_search_settings(db_session)

    document_index = get_default_document_index(
        primary_index_name=search_settings.index_name, secondary_index_name=None
    )

    user_acl_filters = build_access_filters_for_user(user, db_session)
    inference_chunks = document_index.id_based_retrieval(
        chunk_requests=[VespaChunkRequest(document_id=document_id)],
        filters=IndexFilters(access_control_list=user_acl_filters),
    )

    if not inference_chunks:
        raise HTTPException(status_code=404, detail="Document not found")

    contents = [chunk.content for chunk in inference_chunks]

    combined_contents = "\n".join(contents)

    # 获取用于LLM的实际文档上下文
    first_chunk = inference_chunks[0]
    tokenizer_encode = get_tokenizer(
        provider_type=search_settings.provider_type,
        model_name=search_settings.model_name,
    ).encode
    full_context_str = build_doc_context_str(
        semantic_identifier=first_chunk.semantic_identifier,
        source_type=first_chunk.source_type,
        content=combined_contents,
        metadata_dict=first_chunk.metadata,
        updated_at=first_chunk.updated_at,
        ind=0,
    )

    return DocumentInfo(
        num_chunks=len(inference_chunks),
        num_tokens=len(tokenizer_encode(full_context_str)),
    )


@router.get("/chunk-info")
def get_chunk_info(
    document_id: str = Query(...),
    chunk_id: int = Query(...),
    user: User | None = Depends(current_user),
    db_session: Session = Depends(get_session),
) -> ChunkInfo:
    """
    获取文档块信息的API端点

    参数:
        document_id: 文档ID
        chunk_id: 文档块ID
        user: 当前用户对象
        db_session: 数据库会话

    返回:
        ChunkInfo: 包含文档块内容和令牌数量的信息对象

    异常:
        HTTPException: 当文档块未找到时抛出404错误
    """
    search_settings = get_current_search_settings(db_session)

    document_index = get_default_document_index(
        primary_index_name=search_settings.index_name, secondary_index_name=None
    )

    user_acl_filters = build_access_filters_for_user(user, db_session)
    chunk_request = VespaChunkRequest(
        document_id=document_id,
        min_chunk_ind=chunk_id,
        max_chunk_ind=chunk_id,
    )
    inference_chunks = document_index.id_based_retrieval(
        chunk_requests=[chunk_request],
        filters=IndexFilters(access_control_list=user_acl_filters),
        batch_retrieval=True,
    )

    if not inference_chunks:
        raise HTTPException(status_code=404, detail="Chunk not found")

    chunk_content = inference_chunks[0].content

    tokenizer_encode = get_tokenizer(
        provider_type=search_settings.provider_type,
        model_name=search_settings.model_name,
    ).encode

    return ChunkInfo(
        content=chunk_content, num_tokens=len(tokenizer_encode(chunk_content))
    )
