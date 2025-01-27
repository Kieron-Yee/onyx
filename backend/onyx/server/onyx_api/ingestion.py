"""
此文件实现了文档摄取(Ingestion)相关的API接口，主要功能包括：
1. 获取连接器凭证对(connector-credential pair)相关的文档
2. 获取和更新摄取文档
3. 处理文档索引和嵌入

该模块提供了RESTful API接口，用于管理和处理文档的摄取过程。
"""

from fastapi import APIRouter
from fastapi import Depends
from fastapi import HTTPException
from sqlalchemy.orm import Session

from onyx.auth.users import api_key_dep
from onyx.configs.constants import DEFAULT_CC_PAIR_ID
from onyx.configs.constants import DocumentSource
from onyx.connectors.models import Document
from onyx.connectors.models import IndexAttemptMetadata
from onyx.db.connector_credential_pair import get_connector_credential_pair_from_id
from onyx.db.document import get_documents_by_cc_pair
from onyx.db.document import get_ingestion_documents
from onyx.db.engine import get_current_tenant_id
from onyx.db.engine import get_session
from onyx.db.models import User
from onyx.db.search_settings import get_current_search_settings
from onyx.db.search_settings import get_secondary_search_settings
from onyx.document_index.document_index_utils import get_both_index_names
from onyx.document_index.factory import get_default_document_index
from onyx.indexing.embedder import DefaultIndexingEmbedder
from onyx.indexing.indexing_pipeline import build_indexing_pipeline
from onyx.server.onyx_api.models import DocMinimalInfo
from onyx.server.onyx_api.models import IngestionDocument
from onyx.server.onyx_api.models import IngestionResult
from onyx.utils.logger import setup_logger

logger = setup_logger()

# not using /api to avoid confusion with nginx api path routing
# 不使用 /api 前缀以避免与nginx的api路由混淆
router = APIRouter(prefix="/onyx-api")


@router.get("/connector-docs/{cc_pair_id}")
def get_docs_by_connector_credential_pair(
    cc_pair_id: int,
    _: User | None = Depends(api_key_dep),
    db_session: Session = Depends(get_session),
) -> list[DocMinimalInfo]:
    """
    获取指定连接器凭证对的所有文档信息
    
    参数:
        cc_pair_id: 连接器凭证对ID
        _: 用户认证依赖
        db_session: 数据库会话
    
    返回:
        包含文档基本信息的列表
    """
    db_docs = get_documents_by_cc_pair(cc_pair_id=cc_pair_id, db_session=db_session)
    return [
        DocMinimalInfo(
            document_id=doc.id,
            semantic_id=doc.semantic_id,
            link=doc.link,
        )
        for doc in db_docs
    ]


@router.get("/ingestion")
def get_ingestion_docs(
    _: User | None = Depends(api_key_dep),
    db_session: Session = Depends(get_session),
) -> list[DocMinimalInfo]:
    """
    获取所有待摄取的文档列表
    
    参数:
        _: 用户认证依赖
        db_session: 数据库会话
    
    返回:
        包含待摄取文档基本信息的列表
    """
    db_docs = get_ingestion_documents(db_session)
    return [
        DocMinimalInfo(
            document_id=doc.id,
            semantic_id=doc.semantic_id,
            link=doc.link,
        )
        for doc in db_docs
    ]


@router.post("/ingestion")
def upsert_ingestion_doc(
    doc_info: IngestionDocument,
    _: User | None = Depends(api_key_dep),
    db_session: Session = Depends(get_session),
    tenant_id: str = Depends(get_current_tenant_id),
) -> IngestionResult:
    """
    更新或插入待摄取的文档
    
    参数:
        doc_info: 文档信息
        _: 用户认证依赖
        db_session: 数据库会话
        tenant_id: 租户ID
    
    返回:
        摄取结果，包含文档ID和是否为新文档的信息
    
    异常:
        HTTPException: 当指定的连接器凭证对不存在时抛出400错误
        RuntimeError: 当存在次要索引但未配置搜索设置时抛出
    """
    doc_info.document.from_ingestion_api = True

    document = Document.from_base(doc_info.document)

    # TODO once the frontend is updated with this enum, remove this logic
    # TODO 一旦前端更新了这个枚举，移除这个逻辑
    if document.source == DocumentSource.INGESTION_API:
        document.source = DocumentSource.FILE

    cc_pair = get_connector_credential_pair_from_id(
        cc_pair_id=doc_info.cc_pair_id or DEFAULT_CC_PAIR_ID, db_session=db_session
    )
    if cc_pair is None:
        raise HTTPException(
            status_code=400, detail="Connector-Credential Pair specified does not exist"
        )

    # Need to index for both the primary and secondary index if possible
    curr_ind_name, sec_ind_name = get_both_index_names(db_session)
    curr_doc_index = get_default_document_index(
        primary_index_name=curr_ind_name, secondary_index_name=None
    )

    search_settings = get_current_search_settings(db_session)

    index_embedding_model = DefaultIndexingEmbedder.from_db_search_settings(
        search_settings=search_settings
    )

    indexing_pipeline = build_indexing_pipeline(
        embedder=index_embedding_model,
        document_index=curr_doc_index,
        ignore_time_skip=True,
        db_session=db_session,
        tenant_id=tenant_id,
    )

    new_doc, __chunk_count = indexing_pipeline(
        document_batch=[document],
        index_attempt_metadata=IndexAttemptMetadata(
            connector_id=cc_pair.connector_id,
            credential_id=cc_pair.credential_id,
        ),
    )

    # If there's a secondary index being built, index the doc but don't use it for return here
    if sec_ind_name:
        sec_doc_index = get_default_document_index(
            primary_index_name=curr_ind_name, secondary_index_name=None
        )

        sec_search_settings = get_secondary_search_settings(db_session)

        if sec_search_settings is None:
            # Should not ever happen
            # 这种情况不应该发生
            raise RuntimeError(
                "Secondary index exists but no search settings configured"
                # "存在次要索引但没有配置搜索设置"
            )

        new_index_embedding_model = DefaultIndexingEmbedder.from_db_search_settings(
            search_settings=sec_search_settings
        )

        sec_ind_pipeline = build_indexing_pipeline(
            embedder=new_index_embedding_model,
            document_index=sec_doc_index,
            ignore_time_skip=True,
            db_session=db_session,
            tenant_id=tenant_id,
        )

        sec_ind_pipeline(
            document_batch=[document],
            index_attempt_metadata=IndexAttemptMetadata(
                connector_id=cc_pair.connector_id,
                credential_id=cc_pair.credential_id,
            ),
        )

    return IngestionResult(document_id=document.id, already_existed=not bool(new_doc))
