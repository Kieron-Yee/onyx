"""
此文件主要用于处理系统中的反馈相关功能，包括文档检索反馈和聊天消息反馈。
主要实现了文档boost值的更新、文档可见性的控制、用户反馈的创建等功能。
"""
from datetime import datetime
from datetime import timezone
from uuid import UUID

from fastapi import HTTPException
from sqlalchemy import and_
from sqlalchemy import asc
from sqlalchemy import delete
from sqlalchemy import desc
from sqlalchemy import exists
from sqlalchemy import Select
from sqlalchemy import select
from sqlalchemy.orm import aliased
from sqlalchemy.orm import Session

from onyx.configs.constants import MessageType
from onyx.configs.constants import SearchFeedbackType
from onyx.db.chat import get_chat_message
from onyx.db.enums import AccessType
from onyx.db.models import ChatMessageFeedback
from onyx.db.models import ConnectorCredentialPair
from onyx.db.models import Document as DbDocument
from onyx.db.models import DocumentByConnectorCredentialPair
from onyx.db.models import DocumentRetrievalFeedback
from onyx.db.models import User
from onyx.db.models import User__UserGroup
from onyx.db.models import UserGroup__ConnectorCredentialPair
from onyx.db.models import UserRole
from onyx.document_index.interfaces import DocumentIndex
from onyx.utils.logger import setup_logger

logger = setup_logger()

"""
_fetch_db_doc_by_id 函数
功能：通过文档ID从数据库中获取文档对象
参数：
    - doc_id: 文档ID
    - db_session: 数据库会话
返回：数据库文档对象
异常：如果文档ID无效则抛出ValueError
"""
def _fetch_db_doc_by_id(doc_id: str, db_session: Session) -> DbDocument:
    stmt = select(DbDocument).where(DbDocument.id == doc_id)
    result = db_session.execute(stmt)
    doc = result.scalar_one_or_none()

    if not doc:
        raise ValueError("Invalid Document ID Provided")

    return doc

"""
_add_user_filters 函数
功能：为查询语句添加用户权限过滤条件
参数：
    - stmt: 原始查询语句
    - user: 用户对象
    - get_editable: 是否只获取可编辑的内容
返回：添加了用户过滤条件的查询语句
"""
def _add_user_filters(
    stmt: Select, user: User | None, get_editable: bool = True
) -> Select:
    # If user is None, assume the user is an admin or auth is disabled
    # 如果用户为空，假定用户是管理员或认证被禁用
    if user is None or user.role == UserRole.ADMIN:
        return stmt

    DocByCC = aliased(DocumentByConnectorCredentialPair)
    CCPair = aliased(ConnectorCredentialPair)
    UG__CCpair = aliased(UserGroup__ConnectorCredentialPair)
    User__UG = aliased(User__UserGroup)

    """
    Here we select documents by relation:
    User -> User__UserGroup -> UserGroup__ConnectorCredentialPair ->
    ConnectorCredentialPair -> DocumentByConnectorCredentialPair -> Document
    通过以下关系选择文档：
    用户 -> 用户_用户组 -> 用户组_连接器凭证对 ->
    连接器凭证对 -> 文档连接器凭证对 -> 文档
    """
    stmt = (
        stmt.outerjoin(DocByCC, DocByCC.id == DbDocument.id)
        .outerjoin(
            CCPair,
            and_(
                CCPair.connector_id == DocByCC.connector_id,
                CCPair.credential_id == DocByCC.credential_id,
            ),
        )
        .outerjoin(UG__CCpair, UG__CCpair.cc_pair_id == CCPair.id)
        .outerjoin(User__UG, User__UG.user_group_id == UG__CCpair.user_group_id)
    )

    """
    Filter Documents by:
    - if the user is in the user_group that owns the object
    - if the user is not a global_curator, they must also have a curator relationship
    to the user_group
    - if editing is being done, we also filter out objects that are owned by groups
    that the user isn't a curator for
    - if we are not editing, we show all objects in the groups the user is a curator
    for (as well as public objects as well)

    按以下条件过滤文档：
    - 用户是否在拥有该对象的用户组中
    - 如果用户不是全局策展人，他们必须与用户组有策展人关系
    - 如果是编辑操作，我们还会过滤掉用户不是策展人的组所拥有的对象
    - 如果不是编辑操作，我们显示用户是策展人的组中的所有对象（以及公共对象）
    """
    where_clause = User__UG.user_id == user.id
    if user.role == UserRole.CURATOR and get_editable:
        where_clause &= User__UG.is_curator == True  # noqa: E712
    if get_editable:
        user_groups = select(User__UG.user_group_id).where(User__UG.user_id == user.id)
        where_clause &= (
            ~exists()
            .where(UG__CCpair.cc_pair_id == CCPair.id)
            .where(~UG__CCpair.user_group_id.in_(user_groups))
            .correlate(CCPair)
        )
    else:
        where_clause |= CCPair.access_type == AccessType.PUBLIC

    return stmt.where(where_clause)

"""
fetch_docs_ranked_by_boost 函数
功能：获取按boost值排序的文档列表
参数：
    - db_session: 数据库会话
    - user: 用户对象
    - ascending: 是否升序排序
    - limit: 限制返回的文档数量
返回：排序后的文档列表
"""
def fetch_docs_ranked_by_boost(
    db_session: Session,
    user: User | None = None,
    ascending: bool = False,
    limit: int = 100,
) -> list[DbDocument]:
    order_func = asc if ascending else desc
    stmt = select(DbDocument)

    stmt = _add_user_filters(stmt=stmt, user=user, get_editable=False)

    stmt = stmt.order_by(
        order_func(DbDocument.boost), order_func(DbDocument.semantic_id)
    )
    stmt = stmt.limit(limit)
    result = db_session.execute(stmt)
    doc_list = result.scalars().all()

    return list(doc_list)

"""
update_document_boost 函数
功能：更新文档的boost值
参数：
    - db_session: 数据库会话
    - document_id: 文档ID
    - boost: 新的boost值
    - user: 用户对象
异常：如果用户无权编辑文档则抛出HTTPException
"""
def update_document_boost(
    db_session: Session,
    document_id: str,
    boost: int,
    user: User | None = None,
) -> None:
    stmt = select(DbDocument).where(DbDocument.id == document_id)
    stmt = _add_user_filters(stmt, user, get_editable=True)
    result: DbDocument | None = db_session.execute(stmt).scalar_one_or_none()
    if result is None:
        raise HTTPException(
            status_code=400, detail="Document is not editable by this user"
        )

    result.boost = boost

    # updating last_modified triggers sync
    # TODO: Should this submit to the queue directly so that the UI can update?
    result.last_modified = datetime.now(timezone.utc)
    db_session.commit()

"""
update_document_hidden 函数
功能：更新文档的隐藏状态
参数：
    - db_session: 数据库会话
    - document_id: 文档ID
    - hidden: 是否隐藏
    - document_index: 文档索引对象
    - user: 用户对象
异常：如果用户无权编辑文档则抛出HTTPException
"""
def update_document_hidden(
    db_session: Session,
    document_id: str,
    hidden: bool,
    document_index: DocumentIndex,
    user: User | None = None,
) -> None:
    stmt = select(DbDocument).where(DbDocument.id == document_id)
    stmt = _add_user_filters(stmt, user, get_editable=True)
    result = db_session.execute(stmt).scalar_one_or_none()
    if result is None:
        raise HTTPException(
            status_code=400, detail="Document is not editable by this user"
        )

    result.hidden = hidden

    # updating last_modified triggers sync
    # TODO: Should this submit to the queue directly so that the UI can update?
    result.last_modified = datetime.now(timezone.utc)
    db_session.commit()

"""
create_doc_retrieval_feedback 函数
功能：创建文档检索反馈并更新文档的boost值
参数：
    - message_id: 消息ID
    - document_id: 文档ID
    - document_rank: 文档排名
    - document_index: 文档索引对象
    - db_session: 数据库会话
    - clicked: 是否被点击
    - feedback: 反馈类型
"""
def create_doc_retrieval_feedback(
    message_id: int,
    document_id: str,
    document_rank: int,
    document_index: DocumentIndex,
    db_session: Session,
    clicked: bool = False,
    feedback: SearchFeedbackType | None = None,
) -> None:
    """Creates a new Document feedback row and updates the boost value in Postgres and Vespa"""
    db_doc = _fetch_db_doc_by_id(document_id, db_session)

    retrieval_feedback = DocumentRetrievalFeedback(
        chat_message_id=message_id,
        document_id=document_id,
        document_rank=document_rank,
        clicked=clicked,
        feedback=feedback,
    )

    if feedback is not None:
        if feedback == SearchFeedbackType.ENDORSE:
            db_doc.boost += 1
        elif feedback == SearchFeedbackType.REJECT:
            db_doc.boost -= 1
        elif feedback == SearchFeedbackType.HIDE:
            db_doc.hidden = True
        elif feedback == SearchFeedbackType.UNHIDE:
            db_doc.hidden = False
        else:
            raise ValueError("Unhandled document feedback type")

    if feedback in [
        SearchFeedbackType.ENDORSE,
        SearchFeedbackType.REJECT,
        SearchFeedbackType.HIDE,
    ]:
        # updating last_modified triggers sync
        # TODO: Should this submit to the queue directly so that the UI can update?
        db_doc.last_modified = datetime.now(timezone.utc)

    db_session.add(retrieval_feedback)
    db_session.commit()

"""
delete_document_feedback_for_documents__no_commit 函数
功能：删除指定文档的所有反馈记录（不提交事务）
参数：
    - document_ids: 文档ID列表
    - db_session: 数据库会话
注意：此函数不会提交事务，用于更大的事务块中
"""
def delete_document_feedback_for_documents__no_commit(
    document_ids: list[str], db_session: Session
) -> None:
    """NOTE: does not commit transaction so that this can be used as part of a
    larger transaction block."""
    stmt = delete(DocumentRetrievalFeedback).where(
        DocumentRetrievalFeedback.document_id.in_(document_ids)
    )
    db_session.execute(stmt)

"""
create_chat_message_feedback 函数
功能：创建聊天消息的反馈
参数：
    - is_positive: 是否是正面反馈
    - feedback_text: 反馈文本
    - chat_message_id: 聊天消息ID
    - user_id: 用户ID
    - db_session: 数据库会话
    - required_followup: 是否需要人工跟进
    - predefined_feedback: 预定义的反馈
异常：如果没有提供任何反馈内容或消息类型不正确则抛出ValueError
"""
def create_chat_message_feedback(
    is_positive: bool | None,
    feedback_text: str | None,
    chat_message_id: int,
    user_id: UUID | None,
    db_session: Session,
    # Slack user requested help from human
    required_followup: bool | None = None,
    predefined_feedback: str | None = None,  # Added predefined_feedback parameter
) -> None:
    if (
        is_positive is None
        and feedback_text is None
        and required_followup is None
        and predefined_feedback is None
    ):
        raise ValueError("No feedback provided")

    chat_message = get_chat_message(
        chat_message_id=chat_message_id, user_id=user_id, db_session=db_session
    )

    if chat_message.message_type != MessageType.ASSISTANT:
        raise ValueError("Can only provide feedback on LLM Outputs")

    message_feedback = ChatMessageFeedback(
        chat_message_id=chat_message_id,
        is_positive=is_positive,
        feedback_text=feedback_text,
        required_followup=required_followup,
        predefined_feedback=predefined_feedback,
    )

    db_session.add(message_feedback)
    db_session.commit()
