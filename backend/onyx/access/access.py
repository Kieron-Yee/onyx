"""
本模块主要用于处理文档访问权限相关的功能，包括获取文档访问权限信息、
用户ACL权限等核心功能。主要包含了对单个文档和多个文档访问权限的处理逻辑。
"""

from sqlalchemy.orm import Session

from onyx.access.models import DocumentAccess
from onyx.access.utils import prefix_user_email
from onyx.configs.constants import PUBLIC_DOC_PAT
from onyx.db.document import get_access_info_for_document
from onyx.db.document import get_access_info_for_documents
from onyx.db.models import User
from onyx.utils.variable_functionality import fetch_versioned_implementation


def _get_access_for_document(
    document_id: str,
    db_session: Session,
) -> DocumentAccess:
    """
    获取单个文档的访问权限信息
    
    Args:
        document_id: 文档ID
        db_session: 数据库会话
    
    Returns:
        返回包含文档访问权限的DocumentAccess对象
    """
    info = get_access_info_for_document(
        db_session=db_session,
        document_id=document_id,
    )

    return DocumentAccess.build(
        user_emails=info[1] if info and info[1] else [],
        user_groups=[],
        external_user_emails=[],
        external_user_group_ids=[],
        is_public=info[2] if info else False,
    )


def get_access_for_document(
    document_id: str,
    db_session: Session,
) -> DocumentAccess:
    """
    获取单个文档访问权限的版本化实现封装函数
    
    Args:
        document_id: 文档ID
        db_session: 数据库会话
    
    Returns:
        返回文档访问权限对象
    """
    versioned_get_access_for_document_fn = fetch_versioned_implementation(
        "onyx.access.access", "_get_access_for_document"
    )
    return versioned_get_access_for_document_fn(document_id, db_session)  # type: ignore


def get_null_document_access() -> DocumentAccess:
    """
    创建一个空的文档访问权限对象，用于表示最受限的访问权限
    
    Returns:
        返回一个没有任何权限的DocumentAccess对象
    """
    return DocumentAccess(
        user_emails=set(),
        user_groups=set(),
        is_public=False,
        external_user_emails=set(),
        external_user_group_ids=set(),
    )


def _get_access_for_documents(
    document_ids: list[str],
    db_session: Session,
) -> dict[str, DocumentAccess]:
    """
    批量获取多个文档的访问权限信息
    
    Args:
        document_ids: 文档ID列表
        db_session: 数据库会话
    
    Returns:
        返回文档ID到访问权限对象的映射字典
    
    Note:
        有时文档可能还未被索引作业索引，在这种情况下，文档不存在，因此我们使用最低权限。 
        具体来说，EE版本会检查MIT版本权限并创建一个超集。这确保即使文档尚未被索引，此流程也不会失败。
        # Sometimes the document has not be indexed by the indexing job yet, in those cases
        # the document does not exist and so we use least permissive. Specifically the EE version
        # checks the MIT version permissions and creates a superset. This ensures that this flow
        # does not fail even if the Document has not yet been indexed.
    """
    document_access_info = get_access_info_for_documents(
        db_session=db_session,
        document_ids=document_ids,
    )
    doc_access = {
        document_id: DocumentAccess(
            user_emails=set([email for email in user_emails if email]),
            # MIT version will wipe all groups and external groups on update
            user_groups=set(),
            is_public=is_public,
            external_user_emails=set(),
            external_user_group_ids=set(),
        )
        for document_id, user_emails, is_public in document_access_info
    }

    # Sometimes the document has not be indexed by the indexing job yet, in those cases
    # the document does not exist and so we use least permissive. Specifically the EE version
    # checks the MIT version permissions and creates a superset. This ensures that this flow
    # does not fail even if the Document has not yet been indexed.
    for doc_id in document_ids:
        if doc_id not in doc_access:
            doc_access[doc_id] = get_null_document_access()
    return doc_access


def get_access_for_documents(
    document_ids: list[str],
    db_session: Session,
) -> dict[str, DocumentAccess]:
    """
    获取多个文档访问权限的版本化实现封装函数
    
    获取给定文档的所有访问信息。
    Fetches all access information for the given documents.
    
    Args:
        document_ids: 文档ID列表
        db_session: 数据库会话
    
    Returns:
        返回文档ID到访问权限对象的映射字典
    """
    versioned_get_access_for_documents_fn = fetch_versioned_implementation(
        "onyx.access.access", "_get_access_for_documents"
    )
    return versioned_get_access_for_documents_fn(
        document_ids, db_session
    )  # type: ignore


def _get_acl_for_user(user: User | None, db_session: Session) -> set[str]:
    """
    获取用户的ACL权限条目列表
    
    这是用来在下游过滤掉用户无权访问的文档。如果文档的ACL中至少有一个条目
    匹配返回集合中的一个条目，则用户应该可以访问该文档。
    
    Returns a list of ACL entries that the user has access to. This is meant to be
    used downstream to filter out documents that the user does not have access to. The
    user should have access to a document if at least one entry in the document's ACL
    matches one entry in the returned set.
    
    Args:
        user: 用户对象
        db_session: 数据库会话
    
    Returns:
        返回用户可访问的ACL条目集合
    """
    if user:
        return {prefix_user_email(user.email), PUBLIC_DOC_PAT}
    return {PUBLIC_DOC_PAT}


def get_acl_for_user(user: User | None, db_session: Session | None = None) -> set[str]:
    """
    获取用户ACL权限的版本化实现封装函数
    
    Args:
        user: 用户对象
        db_session: 数据库会话
    
    Returns:
        返回用户的ACL权限集合
    """
    versioned_acl_for_user_fn = fetch_versioned_implementation(
        "onyx.access.access", "_get_acl_for_user"
    )
    return versioned_acl_for_user_fn(user, db_session)  # type: ignore
