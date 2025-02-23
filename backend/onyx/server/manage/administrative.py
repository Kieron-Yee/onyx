"""
这个文件包含了管理员相关的API端点实现，主要用于处理文档boost值调整、
文档隐藏状态更新、生成式AI API密钥验证以及连接器删除等管理功能。
"""

from datetime import datetime
from datetime import timedelta
from datetime import timezone
from typing import cast

from fastapi import APIRouter
from fastapi import Depends
from fastapi import HTTPException
from sqlalchemy.orm import Session

from onyx.auth.users import current_admin_user
from onyx.auth.users import current_curator_or_admin_user
from onyx.background.celery.versioned_apps.primary import app as primary_app
from onyx.configs.app_configs import GENERATIVE_MODEL_ACCESS_CHECK_FREQ
from onyx.configs.constants import DocumentSource
from onyx.configs.constants import KV_GEN_AI_KEY_CHECK_TIME
from onyx.configs.constants import OnyxCeleryPriority
from onyx.configs.constants import OnyxCeleryTask
from onyx.db.connector_credential_pair import get_connector_credential_pair
from onyx.db.connector_credential_pair import (
    update_connector_credential_pair_from_id,
)
from onyx.db.engine import get_current_tenant_id
from onyx.db.engine import get_session
from onyx.db.enums import ConnectorCredentialPairStatus
from onyx.db.feedback import fetch_docs_ranked_by_boost
from onyx.db.feedback import update_document_boost
from onyx.db.feedback import update_document_hidden
from onyx.db.index_attempt import cancel_indexing_attempts_for_ccpair
from onyx.db.models import User
from onyx.document_index.document_index_utils import get_both_index_names
from onyx.document_index.factory import get_default_document_index
from onyx.file_store.file_store import get_default_file_store
from onyx.key_value_store.factory import get_kv_store
from onyx.key_value_store.interface import KvKeyNotFoundError
from onyx.llm.factory import get_default_llms
from onyx.llm.utils import test_llm
from onyx.server.documents.models import ConnectorCredentialPairIdentifier
from onyx.server.manage.models import BoostDoc
from onyx.server.manage.models import BoostUpdateRequest
from onyx.server.manage.models import HiddenUpdateRequest
from onyx.server.models import StatusResponse
from onyx.utils.logger import setup_logger

router = APIRouter(prefix="/manage")
logger = setup_logger()

"""Admin only API endpoints"""
"""仅管理员可用的API端点"""


@router.get("/admin/doc-boosts")
def get_most_boosted_docs(
    ascending: bool,
    limit: int,
    user: User | None = Depends(current_curator_or_admin_user),
    db_session: Session = Depends(get_session),
) -> list[BoostDoc]:
    """
    获取boost值排名的文档列表
    
    参数:
        ascending: 是否按升序排列
        limit: 返回结果的数量限制
        user: 当前用户对象
        db_session: 数据库会话
    
    返回:
        包含文档boost信息的列表
    """
    boost_docs = fetch_docs_ranked_by_boost(
        ascending=ascending,
        limit=limit,
        db_session=db_session,
        user=user,
    )
    return [
        BoostDoc(
            document_id=doc.id,
            semantic_id=doc.semantic_id,
            # source=doc.source,
            link=doc.link or "",
            boost=doc.boost,
            hidden=doc.hidden,
        )
        for doc in boost_docs
    ]


@router.post("/admin/doc-boosts")
def document_boost_update(
    boost_update: BoostUpdateRequest,
    user: User | None = Depends(current_curator_or_admin_user),
    db_session: Session = Depends(get_session),
) -> StatusResponse:
    """
    更新文档的boost值
    
    参数:
        boost_update: 包含文档ID和新boost值的请求对象
        user: 当前用户对象
        db_session: 数据库会话
    
    返回:
        更新操作的状态响应
    """
    update_document_boost(
        db_session=db_session,
        document_id=boost_update.document_id,
        boost=boost_update.boost,
        user=user,
    )
    return StatusResponse(success=True, message="Updated document boost")


@router.post("/admin/doc-hidden")
def document_hidden_update(
    hidden_update: HiddenUpdateRequest,
    user: User | None = Depends(current_curator_or_admin_user),
    db_session: Session = Depends(get_session),
) -> StatusResponse:
    """
    更新文档的隐藏状态
    
    参数:
        hidden_update: 包含文档ID和隐藏状态的请求对象
        user: 当前用户对象
        db_session: 数据库会话
    
    返回:
        更新操作的状态响应
    """
    curr_ind_name, sec_ind_name = get_both_index_names(db_session)
    document_index = get_default_document_index(
        primary_index_name=curr_ind_name, secondary_index_name=sec_ind_name
    )

    update_document_hidden(
        db_session=db_session,
        document_id=hidden_update.document_id,
        hidden=hidden_update.hidden,
        document_index=document_index,
        user=user,
    )
    return StatusResponse(success=True, message="Updated document boost")


@router.get("/admin/genai-api-key/validate")
def validate_existing_genai_api_key(
    _: User = Depends(current_admin_user),
) -> None:
    """
    验证现有的生成式AI API密钥
    
    参数:
        _: 当前管理员用户对象(用于权限验证)
    
    异常:
        HTTPException: 当LLM未设置或验证失败时抛出
    """
    # Only validate every so often
    # 仅在特定时间间隔进行验证
    kv_store = get_kv_store()
    curr_time = datetime.now(tz=timezone.utc)
    try:
        last_check = datetime.fromtimestamp(
            cast(float, kv_store.load(KV_GEN_AI_KEY_CHECK_TIME)), tz=timezone.utc
        )
        check_freq_sec = timedelta(seconds=GENERATIVE_MODEL_ACCESS_CHECK_FREQ)
        if curr_time - last_check < check_freq_sec:
            return
    except KvKeyNotFoundError:
        # First time checking the key, nothing unusual
        # 第一次检查密钥，没有什么异常
        pass

    try:
        llm, __ = get_default_llms(timeout=10)
    except ValueError:
        raise HTTPException(status_code=404, detail="LLM not setup")

    error = test_llm(llm)
    if error:
        raise HTTPException(status_code=400, detail=error)

    # Mark check as successful
    # 标记检查成功
    curr_time = datetime.now(tz=timezone.utc)
    kv_store.store(KV_GEN_AI_KEY_CHECK_TIME, curr_time.timestamp())


@router.post("/admin/deletion-attempt")
def create_deletion_attempt_for_connector_id(
    connector_credential_pair_identifier: ConnectorCredentialPairIdentifier,
    user: User = Depends(current_curator_or_admin_user),
    db_session: Session = Depends(get_session),
    tenant_id: str = Depends(get_current_tenant_id),
) -> None:
    """
    为指定的连接器创建删除尝试
    
    参数:
        connector_credential_pair_identifier: 连接器凭证对标识符
        user: 当前用户对象
        db_session: 数据库会话
        tenant_id: 租户ID
    
    异常:
        HTTPException: 当连接器不存在时抛出
    """
    connector_id = connector_credential_pair_identifier.connector_id
    credential_id = connector_credential_pair_identifier.credential_id

    cc_pair = get_connector_credential_pair(
        db_session=db_session,
        connector_id=connector_id,
        credential_id=credential_id,
        user=user,
        get_editable=True,
    )
    if cc_pair is None:
        error = (
            f"Connector with ID '{connector_id}' and credential ID "
            f"'{credential_id}' does not exist. Has it already been deleted?"
        )
        logger.error(error)
        raise HTTPException(
            status_code=404,
            detail=error,
        )

    # Cancel any scheduled indexing attempts
    # 取消所有已计划的索引尝试
    cancel_indexing_attempts_for_ccpair(
        cc_pair_id=cc_pair.id, db_session=db_session, include_secondary_index=True
    )

    # TODO(rkuo): 2024-10-24 - check_deletion_attempt_is_allowed shouldn't be necessary
    # any more due to background locking improvements.
    # TODO(rkuo): 2024-10-24 - 由于后台锁定机制的改进，check_deletion_attempt_is_allowed
    # 不再需要了。
    # Remove the below permanently if everything is behaving for 30 days.

    # Check if the deletion attempt should be allowed
    # deletion_attempt_disallowed_reason = check_deletion_attempt_is_allowed(
    #     connector_credential_pair=cc_pair, db_session=db_session
    # )
    # if deletion_attempt_disallowed_reason:
    #     raise HTTPException(
    #         status_code=400,
    #         detail=deletion_attempt_disallowed_reason,
    #     )

    # mark as deleting
    # 标记为正在删除
    update_connector_credential_pair_from_id(
        db_session=db_session,
        cc_pair_id=cc_pair.id,
        status=ConnectorCredentialPairStatus.DELETING,
    )

    db_session.commit()

    # run the beat task to pick up this deletion from the db immediately
    # 立即运行beat任务以从数据库中获取此删除操作
    primary_app.send_task(
        OnyxCeleryTask.CHECK_FOR_CONNECTOR_DELETION,
        priority=OnyxCeleryPriority.HIGH,
        kwargs={"tenant_id": tenant_id},
    )

    if cc_pair.connector.source == DocumentSource.FILE:
        connector = cc_pair.connector
        file_store = get_default_file_store(db_session)
        for file_name in connector.connector_specific_config.get("file_locations", []):
            file_store.delete_file(file_name)
