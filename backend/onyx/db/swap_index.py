"""
此文件主要用于处理搜索索引的交换操作。
包含检查索引是否需要交换、执行索引交换等相关功能。
"""

from sqlalchemy.orm import Session

from onyx.configs.constants import KV_REINDEX_KEY
from onyx.db.connector_credential_pair import get_connector_credential_pairs
from onyx.db.connector_credential_pair import resync_cc_pair
from onyx.db.enums import IndexModelStatus
from onyx.db.index_attempt import cancel_indexing_attempts_past_model
from onyx.db.index_attempt import (
    count_unique_cc_pairs_with_successful_index_attempts,
)
from onyx.db.models import SearchSettings
from onyx.db.search_settings import get_current_search_settings
from onyx.db.search_settings import get_secondary_search_settings
from onyx.db.search_settings import update_search_settings_status
from onyx.key_value_store.factory import get_kv_store
from onyx.utils.logger import setup_logger


logger = setup_logger()


def check_index_swap(db_session: Session) -> SearchSettings | None:
    """
    检查并执行索引交换操作。
    
    主要功能:
    1. 检查新索引模型是否已完成构建
    2. 如果完成则交换新旧索引
    3. 清理旧索引相关数据
    
    原注释:
    Get count of cc-pairs and count of successful index_attempts for the
    new model grouped by connector + credential, if it's the same, then assume
    new index is done building. If so, swap the indices and expire the old one.
    (获取cc-pairs数量和新模型成功索引尝试的数量(按connector和credential分组)，
    如果数量相同，则认为新索引已构建完成。如果是这样，则交换索引并使旧索引过期。)

    Returns None if search settings did not change, or the old search settings if they
    did change.
    (如果搜索设置没有改变则返回None，如果改变则返回旧的搜索设置。)
    """
    
    old_search_settings = None

    # Default CC-pair created for Ingestion API unused here
    # (为摄取API创建的默认CC-pair在此处未使用)
    all_cc_pairs = get_connector_credential_pairs(db_session)
    cc_pair_count = max(len(all_cc_pairs) - 1, 0)
    search_settings = get_secondary_search_settings(db_session)

    if not search_settings:
        return None

    unique_cc_indexings = count_unique_cc_pairs_with_successful_index_attempts(
        search_settings_id=search_settings.id, db_session=db_session
    )

    # Index Attempts are cleaned up as well when the cc-pair is deleted so the logic in this
    # function is correct. The unique_cc_indexings are specifically for the existing cc-pairs
    # (当cc-pair被删除时索引尝试也会被清理，所以这个函数的逻辑是正确的。unique_cc_indexings专门用于现有的cc-pairs)
    if unique_cc_indexings > cc_pair_count:
        logger.error("More unique indexings than cc pairs, should not occur")
        # (唯一索引数量超过cc pairs数量，这种情况不应该发生)

    if cc_pair_count == 0 or cc_pair_count == unique_cc_indexings:
        # Swap indices (交换索引)
        current_search_settings = get_current_search_settings(db_session)
        update_search_settings_status(
            search_settings=current_search_settings,
            new_status=IndexModelStatus.PAST,
            db_session=db_session,
        )

        update_search_settings_status(
            search_settings=search_settings,
            new_status=IndexModelStatus.PRESENT,
            db_session=db_session,
        )

        if cc_pair_count > 0:
            kv_store = get_kv_store()
            kv_store.store(KV_REINDEX_KEY, False)

            # Expire jobs for the now past index/embedding model
            # (使现在过期的索引/嵌入模型的任务失效)
            cancel_indexing_attempts_past_model(db_session)

            # Recount aggregates (重新计算聚合数据)
            for cc_pair in all_cc_pairs:
                resync_cc_pair(cc_pair, db_session=db_session)

            old_search_settings = current_search_settings

    return old_search_settings
