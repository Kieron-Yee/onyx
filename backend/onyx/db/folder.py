"""
此文件用于处理聊天文件夹相关的数据库操作，包括创建、修改、删除文件夹，
以及管理文件夹中的聊天会话等功能。
"""

from uuid import UUID

from sqlalchemy.orm import Session

from onyx.db.chat import delete_chat_session
from onyx.db.models import ChatFolder
from onyx.db.models import ChatSession
from onyx.utils.logger import setup_logger

logger = setup_logger()


def get_user_folders(
    user_id: UUID | None,
    db_session: Session,
) -> list[ChatFolder]:
    """
    获取指定用户的所有文件夹。
    
    Args:
        user_id: 用户ID
        db_session: 数据库会话
    Returns:
        该用户的所有聊天文件夹列表
    """
    return db_session.query(ChatFolder).filter(ChatFolder.user_id == user_id).all()


def update_folder_display_priority(
    user_id: UUID | None,
    display_priority_map: dict[int, int],
    db_session: Session,
) -> None:
    """
    更新文件夹的显示优先级。
    
    Args:
        user_id: 用户ID
        display_priority_map: 文件夹ID到显示优先级的映射字典
        db_session: 数据库会话
    """
    folders = get_user_folders(user_id=user_id, db_session=db_session)
    folder_ids = {folder.id for folder in folders}
    if folder_ids != set(display_priority_map.keys()):
        raise ValueError("Invalid Folder IDs provided")

    for folder in folders:
        folder.display_priority = display_priority_map[folder.id]

    db_session.commit()


def get_folder_by_id(
    user_id: UUID | None,
    folder_id: int,
    db_session: Session,
) -> ChatFolder:
    """
    根据文件夹ID获取文件夹对象。
    
    Args:
        user_id: 用户ID
        folder_id: 文件夹ID
        db_session: 数据库会话
    Returns:
        指定的聊天文件夹对象
    Raises:
        ValueError: 当文件夹不存在时
        PermissionError: 当文件夹不属于指定用户时
    """
    folder = (
        db_session.query(ChatFolder).filter(ChatFolder.id == folder_id).one_or_none()
    )
    if not folder:
        raise ValueError("Folder by specified id does not exist")

    if folder.user_id != user_id:
        raise PermissionError(f"Folder does not belong to user: {user_id}")

    return folder


def create_folder(
    user_id: UUID | None, folder_name: str | None, db_session: Session
) -> int:
    """
    创建新的聊天文件夹。
    
    Args:
        user_id: 用户ID
        folder_name: 文件夹名称
        db_session: 数据库会话
    Returns:
        新创建的文件夹ID
    """
    new_folder = ChatFolder(
        user_id=user_id,
        name=folder_name,
    )
    db_session.add(new_folder)
    db_session.commit()

    return new_folder.id


def rename_folder(
    user_id: UUID | None, folder_id: int, folder_name: str | None, db_session: Session
) -> None:
    """
    重命名指定的聊天文件夹。
    
    Args:
        user_id: 用户ID
        folder_id: 文件夹ID
        folder_name: 新的文件夹名称
        db_session: 数据库会话
    """
    folder = get_folder_by_id(
        user_id=user_id, folder_id=folder_id, db_session=db_session
    )

    folder.name = folder_name
    db_session.commit()


def add_chat_to_folder(
    user_id: UUID | None, folder_id: int, chat_session: ChatSession, db_session: Session
) -> None:
    """
    将聊天会话添加到指定文件夹中。
    
    Args:
        user_id: 用户ID
        folder_id: 文件夹ID
        chat_session: 要添加的聊天会话对象
        db_session: 数据库会话
    """
    folder = get_folder_by_id(
        user_id=user_id, folder_id=folder_id, db_session=db_session
    )

    chat_session.folder_id = folder.id

    db_session.commit()


def remove_chat_from_folder(
    user_id: UUID | None, folder_id: int, chat_session: ChatSession, db_session: Session
) -> None:
    """
    从指定文件夹中移除聊天会话。
    
    Args:
        user_id: 用户ID
        folder_id: 文件夹ID
        chat_session: 要移除的聊天会话对象
        db_session: 数据库会话
    Raises:
        ValueError: 当聊天会话不在指定文件夹中，或文件夹不属于指定用户时
    """
    folder = get_folder_by_id(
        user_id=user_id, folder_id=folder_id, db_session=db_session
    )

    if chat_session.folder_id != folder.id:
        raise ValueError("聊天会话不在指定的文件夹中。")

    if folder.user_id != user_id:
        raise ValueError(
            f"尝试从不属于该用户的文件夹中移除聊天会话，"
            f"用户ID: {user_id}"
        )

    chat_session.folder_id = None
    if chat_session in folder.chat_sessions:
        folder.chat_sessions.remove(chat_session)

    db_session.commit()


def delete_folder(
    user_id: UUID | None,
    folder_id: int,
    including_chats: bool,
    db_session: Session,
) -> None:
    """
    删除指定的聊天文件夹。
    
    Args:
        user_id: 用户ID
        folder_id: 文件夹ID
        including_chats: 是否同时删除文件夹中的聊天会话
        db_session: 数据库会话
    """
    folder = get_folder_by_id(
        user_id=user_id, folder_id=folder_id, db_session=db_session
    )

    # 假设任何文件夹中的聊天数量都不会很多 / Assuming there will not be a massive number of chats in any given folder
    if including_chats:
        for chat_session in folder.chat_sessions:
            delete_chat_session(
                user_id=user_id,
                chat_session_id=chat_session.id,
                db_session=db_session,
            )

    db_session.delete(folder)
    db_session.commit()
