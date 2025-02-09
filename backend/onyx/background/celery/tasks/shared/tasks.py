"""
此文件包含共享的 Celery 任务，主要用于处理文档清理和同步相关的操作。
主要功能包括：
- 处理文档与连接器/凭证对之间的关系清理
- 管理文档在搜索引擎中的索引状态
- 处理文档的访问权限更新
"""

from http import HTTPStatus

import httpx
from celery import shared_task
from celery import Task
from celery.exceptions import SoftTimeLimitExceeded
from tenacity import RetryError

from onyx.access.access import get_access_for_document
from onyx.background.celery.apps.app_base import task_logger
from onyx.background.celery.tasks.shared.RetryDocumentIndex import RetryDocumentIndex
from onyx.configs.constants import OnyxCeleryTask
from onyx.db.document import delete_document_by_connector_credential_pair__no_commit
from onyx.db.document import delete_documents_complete__no_commit
from onyx.db.document import get_document
from onyx.db.document import get_document_connector_count
from onyx.db.document import mark_document_as_modified
from onyx.db.document import mark_document_as_synced
from onyx.db.document_set import fetch_document_sets_for_document
from onyx.db.engine import get_session_with_tenant
from onyx.document_index.document_index_utils import get_both_index_names
from onyx.document_index.factory import get_default_document_index
from onyx.document_index.interfaces import VespaDocumentFields
from onyx.server.documents.models import ConnectorCredentialPairIdentifier

# 文档清理任务的最大重试次数
DOCUMENT_BY_CC_PAIR_CLEANUP_MAX_RETRIES = 3

# 软时间限制比 RetryDocumentIndex 的 STOP_AFTER+MAX_WAIT 多 5 秒
# Soft time limit is 5 seconds more than RetryDocumentIndex STOP_AFTER+MAX_WAIT
LIGHT_SOFT_TIME_LIMIT = 105
LIGHT_TIME_LIMIT = LIGHT_SOFT_TIME_LIMIT + 15

@shared_task(
    name=OnyxCeleryTask.DOCUMENT_BY_CC_PAIR_CLEANUP_TASK,
    soft_time_limit=LIGHT_SOFT_TIME_LIMIT,
    time_limit=LIGHT_TIME_LIMIT,
    max_retries=DOCUMENT_BY_CC_PAIR_CLEANUP_MAX_RETRIES,
    bind=True,
)
def document_by_cc_pair_cleanup_task(
    self: Task,
    document_id: str,
    connector_id: int,
    credential_id: int,
    tenant_id: str | None,
) -> bool:
    """A lightweight subtask used to clean up document to cc pair relationships.
    Created by connection deletion and connector pruning parent tasks.
    用于清理文档与连接器/凭证对关系的轻量级子任务。
    由连接删除和连接器清理父任务创建。
    """

    """
    To delete a connector / credential pair:
    (1) find all documents associated with connector / credential pair where there
    this the is only connector / credential pair that has indexed it
    (2) delete all documents from document stores
    (3) delete all entries from postgres
    (4) find all documents associated with connector / credential pair where there
    are multiple connector / credential pairs that have indexed it
    (5) update document store entries to remove access associated with the
    connector / credential pair from the access list
    (6) delete all relevant entries from postgres

    删除连接器/凭证对的步骤：
    (1) 查找与连接器/凭证对相关的所有文档，其中只有这个连接器/凭证对索引了它
    (2) 从文档存储中删除所有文档
    (3) 从postgres中删除所有条目
    (4) 查找与连接器/凭证对相关的所有文档，其中有多个连接器/凭证对索引了它
    (5) 更新文档存储条目，从访问列表中删除与连接器/凭证对相关的访问权限
    (6) 从postgres中删除所有相关条目
    """

    # 任务开始的日志记录
    task_logger.debug(f"Task start: doc={document_id}")

    try:
        with get_session_with_tenant(tenant_id) as db_session:
            # 初始化操作类型和受影响的块数
            action = "skip"
            chunks_affected = 0

            # 获取当前和次要索引名称
            curr_ind_name, sec_ind_name = get_both_index_names(db_session)
            doc_index = get_default_document_index(
                primary_index_name=curr_ind_name, secondary_index_name=sec_ind_name
            )

            retry_index = RetryDocumentIndex(doc_index)

            # 获取文档的连接器引用计数
            count = get_document_connector_count(db_session, document_id)
            if count == 1:
                # 如果这是最后一个连接器/凭证对引用，则完全删除文档
                action = "delete"

                chunks_affected = retry_index.delete_single(document_id)
                delete_documents_complete__no_commit(
                    db_session=db_session,
                    document_ids=[document_id],
                )
            elif count > 1:
                action = "update"

                # count > 1 表示文档仍然有其他的 cc_pair 引用
                doc = get_document(document_id, db_session)
                if not doc:
                    return False

                # 以下函数不包括正在被删除的 cc_pairs
                # 即它们将正确地省略当前 cc_pair 的访问权限
                doc_access = get_access_for_document(
                    document_id=document_id, db_session=db_session
                )

                doc_sets = fetch_document_sets_for_document(document_id, db_session)
                update_doc_sets: set[str] = set(doc_sets)

                fields = VespaDocumentFields(
                    document_sets=update_doc_sets,
                    access=doc_access,
                    boost=doc.boost,
                    hidden=doc.hidden,
                )

                # 更新 Vespa。如果文档不存在则没问题，否则会抛出异常
                chunks_affected = retry_index.update_single(document_id, fields=fields)

                # 文档仍然有其他 cc_pair 引用，所以只需要重新同步到 Vespa
                delete_document_by_connector_credential_pair__no_commit(
                    db_session=db_session,
                    document_id=document_id,
                    connector_credential_pair_identifier=ConnectorCredentialPairIdentifier(
                        connector_id=connector_id,
                        credential_id=credential_id,
                    ),
                )

                mark_document_as_synced(document_id, db_session)
            else:
                pass

            db_session.commit()

            task_logger.info(
                f"doc={document_id} "
                f"action={action} "
                f"refcount={count} "
                f"chunks={chunks_affected}"
            )
    except SoftTimeLimitExceeded:
        # 处理软时间限制超时异常
        task_logger.info(f"软时间限制超时异常。doc={document_id}")
        return False
    except Exception as ex:
        # 处理重试错误和其他异常
        if isinstance(ex, RetryError):
            task_logger.warning(
                f"Tenacity retry failed: num_attempts={ex.last_attempt.attempt_number}"
            )

            # only set the inner exception if it is of type Exception
            e_temp = ex.last_attempt.exception()
            if isinstance(e_temp, Exception):
                e = e_temp
        else:
            e = ex

        if isinstance(e, httpx.HTTPStatusError):
            if e.response.status_code == HTTPStatus.BAD_REQUEST:
                task_logger.exception(
                    f"Non-retryable HTTPStatusError: "
                    f"doc={document_id} "
                    f"status={e.response.status_code}"
                )
            return False

        task_logger.exception(f"Unexpected exception: doc={document_id}")

        if self.request.retries < DOCUMENT_BY_CC_PAIR_CLEANUP_MAX_RETRIES:
            # 仍在重试中。指数退避从 2^4 到 2^6，即 16, 32, 64
            countdown = 2 ** (self.request.retries + 4)
            self.retry(exc=e, countdown=countdown)
        else:
            # 这是最后一次尝试！将文档标记为脏数据，以便最终通过过期文档协调进行带外修复
            task_logger.warning(
                f"Max celery task retries reached. Marking doc as dirty for reconciliation: "
                f"doc={document_id}"
            )
            with get_session_with_tenant(tenant_id) as db_session:
                # 现在删除 cc pair 关系，让协调机制在 vespa 中清理它
                delete_document_by_connector_credential_pair__no_commit(
                    db_session=db_session,
                    document_id=document_id,
                    connector_credential_pair_identifier=ConnectorCredentialPairIdentifier(
                        connector_id=connector_id,
                        credential_id=credential_id,
                    ),
                )
                mark_document_as_modified(document_id, db_session)
        return False

    return True
