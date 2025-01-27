"""
此文件实现了提示词(Prompt)相关的API路由处理。
主要功能包括：
- 创建、更新、删除和获取提示词
- 提示词列表的获取
- 提示词与用户关联的处理
"""

from fastapi import APIRouter
from fastapi import Depends
from fastapi import HTTPException
from sqlalchemy.orm import Session
from starlette import status

from onyx.auth.users import current_user
from onyx.db.engine import get_session
from onyx.db.models import User
from onyx.db.persona import get_personas_by_ids
from onyx.db.persona import get_prompt_by_id
from onyx.db.persona import get_prompts
from onyx.db.persona import mark_prompt_as_deleted
from onyx.db.persona import upsert_prompt
from onyx.server.features.prompt.models import CreatePromptRequest
from onyx.server.features.prompt.models import PromptSnapshot
from onyx.utils.logger import setup_logger


# Note: As prompts are fairly innocuous/harmless, there are no protections
# to prevent users from messing with prompts of other users.
# 注：由于提示词相对无害，因此没有设置防止用户修改其他用户提示词的保护机制。

logger = setup_logger()

basic_router = APIRouter(prefix="/prompt")


def create_update_prompt(
    prompt_id: int | None,
    create_prompt_request: CreatePromptRequest,
    user: User | None,
    db_session: Session,
) -> PromptSnapshot:
    """
    创建或更新提示词的核心函数
    
    参数:
        prompt_id: 提示词ID，如果是新建则为None
        create_prompt_request: 创建提示词的请求数据
        user: 当前用户对象
        db_session: 数据库会话
    
    返回:
        PromptSnapshot: 提示词快照对象
    """
    personas = (
        list(
            get_personas_by_ids(
                persona_ids=create_prompt_request.persona_ids,
                db_session=db_session,
            )
        )
        if create_prompt_request.persona_ids
        else []
    )

    prompt = upsert_prompt(
        prompt_id=prompt_id,
        user=user,
        name=create_prompt_request.name,
        description=create_prompt_request.description,
        system_prompt=create_prompt_request.system_prompt,
        task_prompt=create_prompt_request.task_prompt,
        include_citations=create_prompt_request.include_citations,
        datetime_aware=create_prompt_request.datetime_aware,
        personas=personas,
        db_session=db_session,
    )
    return PromptSnapshot.from_model(prompt)


@basic_router.post("")
def create_prompt(
    create_prompt_request: CreatePromptRequest,
    user: User | None = Depends(current_user),
    db_session: Session = Depends(get_session),
) -> PromptSnapshot:
    """
    创建新的提示词
    
    参数:
        create_prompt_request: 创建提示词的请求数据
        user: 当前用户对象
        db_session: 数据库会话
    
    返回:
        PromptSnapshot: 新创建的提示词快照
    
    异常:
        HTTP 400: 创建失败，提供的信息无效
        HTTP 500: 服务器内部错误
    """
    try:
        return create_update_prompt(
            prompt_id=None,
            create_prompt_request=create_prompt_request,
            user=user,
            db_session=db_session,
        )
    except ValueError as ve:
        logger.exception(ve)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to create Persona, invalid info.",
        )
    except Exception as e:
        logger.exception(e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred. Please try again later.",
        )


@basic_router.patch("/{prompt_id}")
def update_prompt(
    prompt_id: int,
    update_prompt_request: CreatePromptRequest,
    user: User | None = Depends(current_user),
    db_session: Session = Depends(get_session),
) -> PromptSnapshot:
    """
    更新现有的提示词
    
    参数:
        prompt_id: 要更新的提示词ID
        update_prompt_request: 更新提示词的请求数据
        user: 当前用户对象
        db_session: 数据库会话
    
    返回:
        PromptSnapshot: 更新后的提示词快照
    
    异常:
        HTTP 400: 更新失败，提供的信息无效
        HTTP 500: 服务器内部错误
    """
    try:
        return create_update_prompt(
            prompt_id=prompt_id,
            create_prompt_request=update_prompt_request,
            user=user,
            db_session=db_session,
        )
    except ValueError as ve:
        logger.exception(ve)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to create Persona, invalid info.",
        )
    except Exception as e:
        logger.exception(e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred. Please try again later.",
        )


@basic_router.delete("/{prompt_id}")
def delete_prompt(
    prompt_id: int,
    user: User | None = Depends(current_user),
    db_session: Session = Depends(get_session),
) -> None:
    """
    删除指定的提示词
    
    参数:
        prompt_id: 要删除的提示词ID
        user: 当前用户对象
        db_session: 数据库会话
    """
    mark_prompt_as_deleted(
        prompt_id=prompt_id,
        user=user,
        db_session=db_session,
    )


@basic_router.get("")
def list_prompts(
    user: User | None = Depends(current_user),
    db_session: Session = Depends(get_session),
) -> list[PromptSnapshot]:
    """
    获取提示词列表
    
    参数:
        user: 当前用户对象
        db_session: 数据库会话
    
    返回:
        list[PromptSnapshot]: 提示词快照列表
    """
    user_id = user.id if user is not None else None
    return [
        PromptSnapshot.from_model(prompt)
        for prompt in get_prompts(user_id=user_id, db_session=db_session)
    ]


@basic_router.get("/{prompt_id}")
def get_prompt(
    prompt_id: int,
    user: User | None = Depends(current_user),
    db_session: Session = Depends(get_session),
) -> PromptSnapshot:
    """
    获取指定ID的提示词
    
    参数:
        prompt_id: 提示词ID
        user: 当前用户对象
        db_session: 数据库会话
    
    返回:
        PromptSnapshot: 提示词快照对象
    """
    return PromptSnapshot.from_model(
        get_prompt_by_id(
            prompt_id=prompt_id,
            user=user,
            db_session=db_session,
        )
    )
