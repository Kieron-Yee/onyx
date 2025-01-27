"""
文档索引工厂模块

本模块负责创建和管理文档索引实例。主要提供了获取默认文档索引和当前主要文档索引的功能。
目前支持Vespa作为索引引擎。
"""

from sqlalchemy.orm import Session

from onyx.db.search_settings import get_current_search_settings
from onyx.document_index.interfaces import DocumentIndex
from onyx.document_index.vespa.index import VespaIndex
from shared_configs.configs import MULTI_TENANT


def get_default_document_index(
    primary_index_name: str,
    secondary_index_name: str | None,
) -> DocumentIndex:
    """
    获取默认的文档索引实例。
    
    Primary index is the index that is used for querying/updating etc.
    Secondary index is for when both the currently used index and the upcoming
    index both need to be updated, updates are applied to both indices
    主索引用于查询和更新操作。当需要同时更新当前使用的索引和即将使用的索引时，
    次级索引用于确保更新同时应用到两个索引。

    Args:
        primary_index_name (str): 主索引名称
        secondary_index_name (str | None): 次级索引名称，可以为空

    Returns:
        DocumentIndex: 文档索引实例
    
    说明：
        目前仅支持Vespa索引引擎
    """
    # Currently only supporting Vespa
    # 目前仅支持Vespa
    return VespaIndex(
        index_name=primary_index_name,
        secondary_index_name=secondary_index_name,
        multitenant=MULTI_TENANT,
    )


def get_current_primary_default_document_index(db_session: Session) -> DocumentIndex:
    """
    获取当前主要的默认文档索引实例。
    
    TODO: Use redis to cache this or something
    待办：考虑使用Redis进行缓存

    Args:
        db_session (Session): 数据库会话实例

    Returns:
        DocumentIndex: 当前使用的主文档索引实例
    """
    search_settings = get_current_search_settings(db_session)
    return get_default_document_index(
        primary_index_name=search_settings.index_name,
        secondary_index_name=None,
    )
