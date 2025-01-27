"""
文件夹相关的API路由模块

本模块实现了文件夹功能相关的所有REST API端点，包括：
- 获取用户文件夹列表
- 创建新文件夹
- 重命名文件夹
- 删除文件夹
- 调整文件夹显示顺序
- 向文件夹中添加/删除聊天会话
"""

from fastapi import APIRouter
from fastapi import Depends
from fastapi import HTTPException
from fastapi import Path
from sqlalchemy.orm import Session

from onyx.auth.users import current_user
from onyx.db.chat import get_chat_session_by_id
from onyx.db.engine import get_session
from onyx.db.folder import add_chat_to_folder
from onyx.db.folder import create_folder
from onyx.db.folder import delete_folder
from onyx.db.folder import get_user_folders
from onyx.db.folder import remove_chat_from_folder
from onyx.db.folder import rename_folder
from onyx.db.folder import update_folder_display_priority
from onyx.db.models import User
from onyx.server.features.folder.models import DeleteFolderOptions
from onyx.server.features.folder.models import FolderChatSessionRequest
from onyx.server.features.folder.models import FolderCreationRequest
from onyx.server.features.folder.models import FolderResponse
from onyx.server.features.folder.models import FolderUpdateRequest
from onyx.server.features.folder.models import GetUserFoldersResponse
from onyx.server.models import DisplayPriorityRequest
from onyx.server.query_and_chat.models import ChatSessionDetails

router = APIRouter(prefix="/folder")


@router.get("")
def get_folders(
    user: User = Depends(current_user),
    db_session: Session = Depends(get_session),
) -> GetUserFoldersResponse:
    """
    获取用户的所有文件夹信息

    Args:
        user: 当前用户对象
        db_session: 数据库会话对象

    Returns:
        GetUserFoldersResponse: 包含用户所有文件夹信息的响应对象
    """
    folders = get_user_folders(
        user_id=user.id if user else None,
        db_session=db_session,
    )
    folders.sort()
    return GetUserFoldersResponse(
        folders=[
            FolderResponse(
                folder_id=folder.id,
                folder_name=folder.name,
                display_priority=folder.display_priority,
                chat_sessions=[
                    ChatSessionDetails(
                        id=chat_session.id,
                        name=chat_session.description,
                        persona_id=chat_session.persona_id,
                        time_created=chat_session.time_created.isoformat(),
                        shared_status=chat_session.shared_status,
                        folder_id=folder.id,
                    )
                    for chat_session in folder.chat_sessions
                    if not chat_session.deleted
                ],
            )
            for folder in folders
        ]
    )


@router.put("/reorder")
def put_folder_display_priority(
    display_priority_request: DisplayPriorityRequest,
    user: User | None = Depends(current_user),
    db_session: Session = Depends(get_session),
) -> None:
    """
    更新文件夹的显示优先级

    Args:
        display_priority_request: 包含文件夹ID和显示优先级映射的请求对象
        user: 当前用户对象
        db_session: 数据库会话对象
    """
    update_folder_display_priority(
        user_id=user.id if user else None,
        display_priority_map=display_priority_request.display_priority_map,
        db_session=db_session,
    )


@router.post("")
def create_folder_endpoint(
    request: FolderCreationRequest,
    user: User = Depends(current_user),
    db_session: Session = Depends(get_session),
) -> int:
    """
    创建新文件夹

    Args:
        request: 包含文件夹名称的创建请求对象
        user: 当前用户对象
        db_session: 数据库会话对象

    Returns:
        int: 新创建的文件夹ID
    """
    return create_folder(
        user_id=user.id if user else None,
        folder_name=request.folder_name,
        db_session=db_session,
    )


@router.patch("/{folder_id}")
def patch_folder_endpoint(
    request: FolderUpdateRequest,
    folder_id: int = Path(..., description="The ID of the folder to rename"), # 要重命名的文件夹ID
    user: User = Depends(current_user),
    db_session: Session = Depends(get_session),
) -> None:
    """
    重命名文件夹

    Args:
        request: 包含新文件夹名称的更新请求对象
        folder_id: 要重命名的文件夹ID
        user: 当前用户对象
        db_session: 数据库会话对象

    Raises:
        HTTPException: 重命名失败时抛出400错误
    """
    try:
        rename_folder(
            user_id=user.id if user else None,
            folder_id=folder_id,
            folder_name=request.folder_name,
            db_session=db_session,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/{folder_id}")
def delete_folder_endpoint(
    request: DeleteFolderOptions,
    folder_id: int = Path(..., description="The ID of the folder to delete"), # 要删除的文件夹ID
    user: User = Depends(current_user),
    db_session: Session = Depends(get_session),
) -> None:
    """
    删除文件夹

    Args:
        request: 包含删除选项的请求对象
        folder_id: 要删除的文件夹ID
        user: 当前用户对象
        db_session: 数据库会话对象

    Raises:
        HTTPException: 删除失败时抛出400错误
    """
    user_id = user.id if user else None
    try:
        delete_folder(
            user_id=user_id,
            folder_id=folder_id,
            including_chats=request.including_chats,
            db_session=db_session,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/{folder_id}/add-chat-session")
def add_chat_to_folder_endpoint(
    request: FolderChatSessionRequest,
    folder_id: int = Path(
        ..., description="The ID of the folder in which to add the chat session" # 要添加聊天会话的文件夹ID
    ),
    user: User = Depends(current_user),
    db_session: Session = Depends(get_session),
) -> None:
    """
    向文件夹中添加聊天会话

    Args:
        request: 包含聊天会话ID的请求对象
        folder_id: 目标文件夹ID
        user: 当前用户对象
        db_session: 数据库会话对象

    Raises:
        HTTPException: 添加失败时抛出400错误
    """
    user_id = user.id if user else None
    try:
        chat_session = get_chat_session_by_id(
            chat_session_id=request.chat_session_id,
            user_id=user_id,
            db_session=db_session,
        )
        add_chat_to_folder(
            user_id=user.id if user else None,
            folder_id=folder_id,
            chat_session=chat_session,
            db_session=db_session,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/{folder_id}/remove-chat-session/")
def remove_chat_from_folder_endpoint(
    request: FolderChatSessionRequest,
    folder_id: int = Path(
        ..., description="The ID of the folder from which to remove the chat session" # 要移除聊天会话的文件夹ID
    ),
    user: User = Depends(current_user),
    db_session: Session = Depends(get_session),
) -> None:
    """
    从文件夹中移除聊天会话

    Args:
        request: 包含聊天会话ID的请求对象
        folder_id: 源文件夹ID
        user: 当前用户对象
        db_session: 数据库会话对象

    Raises:
        HTTPException: 移除失败时抛出400错误
    """
    user_id = user.id if user else None
    try:
        chat_session = get_chat_session_by_id(
            chat_session_id=request.chat_session_id,
            user_id=user_id,
            db_session=db_session,
        )
        remove_chat_from_folder(
            user_id=user_id,
            folder_id=folder_id,
            chat_session=chat_session,
            db_session=db_session,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
