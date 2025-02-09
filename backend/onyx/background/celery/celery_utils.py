"""
此文件包含了Celery相关的实用工具函数，主要用于处理任务队列状态、文档删除操作、
文档ID提取以及Celery worker状态检查等功能。
"""

from datetime import datetime
from datetime import timezone
from typing import Any

from sqlalchemy.orm import Session

from onyx.configs.app_configs import MAX_PRUNING_DOCUMENT_RETRIEVAL_PER_MINUTE
from onyx.connectors.cross_connector_utils.rate_limit_wrapper import (
    rate_limit_builder,
)
from onyx.connectors.interfaces import BaseConnector
from onyx.connectors.interfaces import LoadConnector
from onyx.connectors.interfaces import PollConnector
from onyx.connectors.interfaces import SlimConnector
from onyx.connectors.models import Document
from onyx.db.connector_credential_pair import get_connector_credential_pair
from onyx.db.enums import TaskStatus
from onyx.db.models import TaskQueueState
from onyx.indexing.indexing_heartbeat import IndexingHeartbeatInterface
from onyx.redis.redis_connector import RedisConnector
from onyx.server.documents.models import DeletionAttemptSnapshot
from onyx.utils.logger import setup_logger


logger = setup_logger()


def _get_deletion_status(
    connector_id: int,
    credential_id: int,
    db_session: Session,
    tenant_id: str | None = None,
) -> TaskQueueState | None:
    """We no longer store TaskQueueState in the DB for a deletion attempt.
    This function populates TaskQueueState by just checking redis.
    
    我们不再在数据库中存储删除尝试的TaskQueueState。
    此函数通过检查redis来填充TaskQueueState。
    
    参数:
        connector_id: 连接器ID
        credential_id: 凭证ID
        db_session: 数据库会话
        tenant_id: 租户ID，可选
        
    返回:
        TaskQueueState对象或None
    """
    cc_pair = get_connector_credential_pair(
        connector_id=connector_id, credential_id=credential_id, db_session=db_session
    )
    if not cc_pair:
        return None

    redis_connector = RedisConnector(tenant_id, cc_pair.id)
    if not redis_connector.delete.fenced:
        return None

    return TaskQueueState(
        task_id="",
        task_name=redis_connector.delete.fence_key,
        status=TaskStatus.STARTED,
    )


def get_deletion_attempt_snapshot(
    connector_id: int,
    credential_id: int,
    db_session: Session,
    tenant_id: str | None = None,
) -> DeletionAttemptSnapshot | None:
    """
    获取删除尝试的快照信息
    
    参数:
        connector_id: 连接器ID
        credential_id: 凭证ID
        db_session: 数据库会话
        tenant_id: 租户ID，可选
        
    返回:
        DeletionAttemptSnapshot对象或None
    """
    deletion_task = _get_deletion_status(
        connector_id, credential_id, db_session, tenant_id
    )
    if not deletion_task:
        return None

    return DeletionAttemptSnapshot(
        connector_id=connector_id,
        credential_id=credential_id,
        status=deletion_task.status,
    )


def document_batch_to_ids(
    doc_batch: list[Document],
) -> set[str]:
    """
    将文档批次转换为ID集合
    
    参数:
        doc_batch: 文档对象列表
    返回:
        文档ID集合
    """
    return {doc.id for doc in doc_batch}


def extract_ids_from_runnable_connector(
    runnable_connector: BaseConnector,
    callback: IndexingHeartbeatInterface | None = None,
) -> set[str]:
    """If the SlimConnector hasnt been implemented for the given connector, just pull
    all docs using the load_from_state and grab out the IDs.
    
    如果给定连接器没有实现SlimConnector，则使用load_from_state获取所有文档并提取ID。

    Optionally, a callback can be passed to handle the length of each document batch.
    可选地，可以传递回调函数来处理每个文档批次的长度。
    
    参数:
        runnable_connector: 可运行的连接器实例
        callback: 索引心跳接口回调，可选
        
    返回:
        文档ID集合
    
    异常:
        RuntimeError: 当找不到有效的runnable_connector时抛出
    """
    all_connector_doc_ids: set[str] = set()

    if isinstance(runnable_connector, SlimConnector):
        for metadata_batch in runnable_connector.retrieve_all_slim_documents():
            all_connector_doc_ids.update({doc.id for doc in metadata_batch})

    doc_batch_generator = None

    if isinstance(runnable_connector, LoadConnector):
        doc_batch_generator = runnable_connector.load_from_state()
    elif isinstance(runnable_connector, PollConnector):
        start = datetime(1970, 1, 1, tzinfo=timezone.utc).timestamp()
        end = datetime.now(timezone.utc).timestamp()
        doc_batch_generator = runnable_connector.poll_source(start=start, end=end)
    else:
        raise RuntimeError("Pruning job could not find a valid runnable_connector.")

    doc_batch_processing_func = document_batch_to_ids
    if MAX_PRUNING_DOCUMENT_RETRIEVAL_PER_MINUTE:
        doc_batch_processing_func = rate_limit_builder(
            max_calls=MAX_PRUNING_DOCUMENT_RETRIEVAL_PER_MINUTE, period=60
        )(document_batch_to_ids)
    for doc_batch in doc_batch_generator:
        if callback:
            if callback.should_stop():
                raise RuntimeError(
                    "extract_ids_from_runnable_connector: Stop signal detected"
                )

        all_connector_doc_ids.update(doc_batch_processing_func(doc_batch))

        if callback:
            callback.progress("extract_ids_from_runnable_connector", len(doc_batch))

    return all_connector_doc_ids


def celery_is_listening_to_queue(worker: Any, name: str) -> bool:
    """Checks to see if we're listening to the named queue
    检查是否正在监听指定名称的队列
    
    参数:
        worker: Celery worker实例
        name: 队列名称
        
    返回:
        是否正在监听该队列
    """
    # how to get a list of queues this worker is listening to
    # https://stackoverflow.com/questions/29790523/how-to-determine-which-queues-a-celery-worker-is-consuming-at-runtime
    queue_names = list(worker.app.amqp.queues.consume_from.keys())
    for queue_name in queue_names:
        if queue_name == name:
            return True

    return False


def celery_is_worker_primary(worker: Any) -> bool:
    """There are multiple approaches that could be taken to determine if a celery worker
    is 'primary', as defined by us. But the way we do it is to check the hostname set
    for the celery worker, which can be done on the command line with '--hostname'.
    
    有多种方法可以确定一个celery worker是否为'primary'。我们的方法是检查为celery worker
    设置的主机名，这可以通过命令行的'--hostname'参数来完成。
    
    参数:
        worker: Celery worker实例
        
    返回:
        是否为主要worker
    """
    hostname = worker.hostname
    if hostname.startswith("primary"):
        return True

    return False
