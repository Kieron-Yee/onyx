"""
这个文件主要负责处理文档标签(Tag)相关的数据库操作，包括标签的创建、查询和删除等功能。
提供了一系列函数来管理文档和标签之间的关联关系，以及处理孤立标签的清理工作。
"""

from sqlalchemy import and_
from sqlalchemy import delete
from sqlalchemy import func
from sqlalchemy import or_
from sqlalchemy import select
from sqlalchemy.orm import Session

from onyx.configs.constants import DocumentSource
from onyx.db.models import Document
from onyx.db.models import Document__Tag
from onyx.db.models import Tag
from onyx.utils.logger import setup_logger

logger = setup_logger()


def check_tag_validity(tag_key: str, tag_value: str) -> bool:
    """检查标签的有效性
    
    如果标签过长，则不应使用（这会在Postgres中导致错误，因为唯一约束只能应用于小于2704字节的条目）。
    此外，过长的标签实际上也不太可用/有用。
    
    If a tag is too long, it should not be used (it will cause an error in Postgres
    as the unique constraint can only apply to entries that are less than 2704 bytes).
    Additionally, extremely long tags are not really usable / useful.
    """
    if len(tag_key) + len(tag_value) > 255:
        logger.error(
            f"Tag with key '{tag_key}' and value '{tag_value}' is too long, cannot be used"
        )
        return False

    return True


def create_or_add_document_tag(
    tag_key: str,
    tag_value: str,
    source: DocumentSource,
    document_id: str,
    db_session: Session,
) -> Tag | None:
    """创建新标签或将现有标签添加到文档中
    
    该函数会检查标签是否存在，如果不存在则创建新标签。
    然后将标签与指定的文档关联起来。如果标签已经与文档关联，则不会重复添加。
    """
    if not check_tag_validity(tag_key, tag_value):
        return None

    document = db_session.get(Document, document_id)
    if not document:
        raise ValueError("Invalid Document, cannot attach Tags")

    tag_stmt = select(Tag).where(
        Tag.tag_key == tag_key,
        Tag.tag_value == tag_value,
        Tag.source == source,
    )
    tag = db_session.execute(tag_stmt).scalar_one_or_none()

    if not tag:
        tag = Tag(tag_key=tag_key, tag_value=tag_value, source=source)
        db_session.add(tag)

    if tag not in document.tags:
        document.tags.append(tag)

    db_session.commit()
    return tag


def create_or_add_document_tag_list(
    tag_key: str,
    tag_values: list[str],
    source: DocumentSource,
    document_id: str,
    db_session: Session,
) -> list[Tag]:
    """批量创建或添加多个标签到文档中
    
    这个函数可以一次性处理多个标签值，它会：
    1. 过滤出有效的标签值
    2. 检查已存在的标签
    3. 创建不存在的标签
    4. 将所有标签关联到指定文档
    """
    valid_tag_values = [
        tag_value for tag_value in tag_values if check_tag_validity(tag_key, tag_value)
    ]
    if not valid_tag_values:
        return []

    document = db_session.get(Document, document_id)
    if not document:
        raise ValueError("Invalid Document, cannot attach Tags")

    existing_tags_stmt = select(Tag).where(
        Tag.tag_key == tag_key,
        Tag.tag_value.in_(valid_tag_values),
        Tag.source == source,
    )
    existing_tags = list(db_session.execute(existing_tags_stmt).scalars().all())
    existing_tag_values = {tag.tag_value for tag in existing_tags}

    new_tags = []
    for tag_value in valid_tag_values:
        if tag_value not in existing_tag_values:
            new_tag = Tag(tag_key=tag_key, tag_value=tag_value, source=source)
            db_session.add(new_tag)
            new_tags.append(new_tag)
            existing_tag_values.add(tag_value)

    if new_tags:
        logger.debug(
            f"Created new tags: {', '.join([f'{tag.tag_key}:{tag.tag_value}' for tag in new_tags])}"
        )

    all_tags = existing_tags + new_tags

    for tag in all_tags:
        if tag not in document.tags:
            document.tags.append(tag)

    db_session.commit()
    return all_tags


def find_tags(
    tag_key_prefix: str | None,
    tag_value_prefix: str | None,
    sources: list[DocumentSource] | None,
    limit: int | None,
    db_session: Session,
    require_both_to_match: bool = False,
) -> list[Tag]:
    """查找符合条件的标签
    
    支持通过标签键前缀、值前缀和来源来搜索标签。
    可以设置是否要求键和值前缀都匹配，以及限制返回结果的数量。
    
    # if set, both tag_key_prefix and tag_value_prefix must be a match
    # 如果设置为True，则标签键前缀和值前缀都必须匹配
    """
    query = select(Tag)

    if tag_key_prefix or tag_value_prefix:
        conditions = []
        if tag_key_prefix:
            conditions.append(Tag.tag_key.ilike(f"{tag_key_prefix}%"))
        if tag_value_prefix:
            conditions.append(Tag.tag_value.ilike(f"{tag_value_prefix}%"))

        final_prefix_condition = (
            and_(*conditions) if require_both_to_match else or_(*conditions)
        )
        query = query.where(final_prefix_condition)

    if sources:
        query = query.where(Tag.source.in_(sources))

    if limit:
        query = query.limit(limit)

    result = db_session.execute(query)

    tags = result.scalars().all()
    return list(tags)


def delete_document_tags_for_documents__no_commit(
    document_ids: list[str], db_session: Session
) -> None:
    """删除指定文档的所有标签关联，并清理孤立标签
    
    这个函数会：
    1. 删除指定文档ID列表中所有文档的标签关联
    2. 查找并删除不再与任何文档关联的孤立标签
    注意：此函数不会自动提交事务，需要调用方手动提交
    """
    stmt = delete(Document__Tag).where(Document__Tag.document_id.in_(document_ids))
    db_session.execute(stmt)

    orphan_tags_query = (
        select(Tag.id)
        .outerjoin(Document__Tag, Tag.id == Document__Tag.tag_id)
        .group_by(Tag.id)
        .having(func.count(Document__Tag.document_id) == 0)
    )

    orphan_tags = db_session.execute(orphan_tags_query).scalars().all()

    if orphan_tags:
        delete_orphan_tags_stmt = delete(Tag).where(Tag.id.in_(orphan_tags))
        db_session.execute(delete_orphan_tags_stmt)
