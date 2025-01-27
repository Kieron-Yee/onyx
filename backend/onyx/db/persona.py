"""
这个文件主要负责系统中Persona(角色)相关的数据库操作。
包括创建、更新、获取和删除Persona,以及管理Persona的权限、提示词、工具等相关功能。
"""

from collections.abc import Sequence
from datetime import datetime
from functools import lru_cache
from uuid import UUID

from fastapi import HTTPException
from sqlalchemy import delete
from sqlalchemy import exists
from sqlalchemy import func
from sqlalchemy import not_
from sqlalchemy import or_
from sqlalchemy import Select
from sqlalchemy import select
from sqlalchemy import update
from sqlalchemy.orm import aliased
from sqlalchemy.orm import joinedload
from sqlalchemy.orm import Session

from onyx.auth.schemas import UserRole
from onyx.configs.chat_configs import BING_API_KEY
from onyx.configs.chat_configs import CONTEXT_CHUNKS_ABOVE
from onyx.configs.chat_configs import CONTEXT_CHUNKS_BELOW
from onyx.context.search.enums import RecencyBiasSetting
from onyx.db.constants import SLACK_BOT_PERSONA_PREFIX
from onyx.db.engine import get_sqlalchemy_engine
from onyx.db.models import DocumentSet
from onyx.db.models import Persona
from onyx.db.models import Persona__User
from onyx.db.models import Persona__UserGroup
from onyx.db.models import PersonaCategory
from onyx.db.models import Prompt
from onyx.db.models import StarterMessage
from onyx.db.models import Tool
from onyx.db.models import User
from onyx.db.models import User__UserGroup
from onyx.db.models import UserGroup
from onyx.server.features.persona.models import CreatePersonaRequest
from onyx.server.features.persona.models import PersonaSnapshot
from onyx.utils.logger import setup_logger
from onyx.utils.variable_functionality import fetch_versioned_implementation

logger = setup_logger()


def _add_user_filters(
    stmt: Select, user: User | None, get_editable: bool = True
) -> Select:
    """
    为查询语句添加用户过滤条件
    
    参数:
        stmt: 查询语句对象
        user: 用户对象,None表示admin或禁用认证
        get_editable: 是否只获取可编辑的内容
    """
    # 如果用户是None,假定用户是admin或auth被禁用
    if user is None or user.role == UserRole.ADMIN:
        return stmt

    Persona__UG = aliased(Persona__UserGroup)
    User__UG = aliased(User__UserGroup)
    """
    这里我们通过关系选择cc_pairs:
    用户 -> User__UserGroup -> Persona__UserGroup -> Persona
    """
    stmt = (
        stmt.outerjoin(Persona__UG)
        .outerjoin(
            User__UserGroup,
            User__UserGroup.user_group_id == Persona__UG.user_group_id,
        )
        .outerjoin(
            Persona__User,
            Persona__User.persona_id == Persona.id,
        )
    )
    """
    过滤Persona:
    - 如果用户在拥有Persona的user_group中
    - 如果用户不是global_curator,他们还必须与user_group有curator关系
    - 如果正在编辑,我们还会过滤掉用户不是curator的组拥有的Persona
    - 如果我们不在编辑,我们显示用户是curator的组中的所有Persona(以及公共Persona)
    - 如果我们不在编辑,我们返回直接连接到用户的所有Persona
    """
    where_clause = User__UserGroup.user_id == user.id
    if user.role == UserRole.CURATOR and get_editable:
        where_clause &= User__UserGroup.is_curator == True  # noqa: E712
    if get_editable:
        user_groups = select(User__UG.user_group_id).where(User__UG.user_id == user.id)
        if user.role == UserRole.CURATOR:
            user_groups = user_groups.where(User__UG.is_curator == True)  # noqa: E712
        where_clause &= (
            ~exists()
            .where(Persona__UG.persona_id == Persona.id)
            .where(~Persona__UG.user_group_id.in_(user_groups))
            .correlate(Persona)
        )
    else:
        where_clause |= Persona.is_public == True  # noqa: E712
        where_clause &= Persona.is_visible == True  # noqa: E712
        where_clause |= Persona__User.user_id == user.id
    where_clause |= Persona.user_id == user.id

    return stmt.where(where_clause)


def fetch_persona_by_id(
    db_session: Session, persona_id: int, user: User | None, get_editable: bool = True
) -> Persona:
    """
    根据ID获取一个Persona对象
    
    参数:
        db_session: 数据库会话
        persona_id: Persona的ID
        user: 当前用户,用于权限验证
        get_editable: 是否只获取可编辑的内容
    """
    stmt = select(Persona).where(Persona.id == persona_id).distinct()
    stmt = _add_user_filters(stmt=stmt, user=user, get_editable=get_editable)
    persona = db_session.scalars(stmt).one_or_none()
    if not persona:
        raise HTTPException(
            status_code=403,
            detail=f"Persona with ID {persona_id} does not exist or user is not authorized to access it",
        )
    return persona


def get_best_persona_id_for_user(
    db_session: Session, user: User | None, persona_id: int | None = None
) -> int | None:
    """
    获取对指定用户最合适的Persona ID
    
    如果提供了persona_id且用户有权访问,则返回该ID
    否则返回用户可访问的优先级最高的Persona ID
    """
    if persona_id is not None:
        stmt = select(Persona).where(Persona.id == persona_id).distinct()
        stmt = _add_user_filters(
            stmt=stmt,
            user=user,
            # 我们不想在这里按可编辑过滤,我们只想查看persona是否对用户可用
            get_editable=False,
        )
        persona = db_session.scalars(stmt).one_or_none()
        if persona:
            return persona.id

    # 如果未找到persona,或slack bot使用文档集而不是persona,
    # 我们需要找到最适合用户的persona
    # 这是用户有权访问的显示优先级最高的persona
    stmt = select(Persona).order_by(Persona.display_priority.desc()).distinct()
    stmt = _add_user_filters(stmt=stmt, user=user, get_editable=True)
    persona = db_session.scalars(stmt).one_or_none()
    return persona.id if persona else None


def _get_persona_by_name(
    persona_name: str, user: User | None, db_session: Session
) -> Persona | None:
    """
    根据名称获取Persona对象
    
    参数:
        persona_name: Persona的名称
        user: 当前用户,用于权限验证
        db_session: 数据库会话
    """
    # 管理员可以查看所有,普通用户只能获取自己的
    # 如果用户是None,假定用户是admin或auth被禁用
    stmt = select(Persona).where(Persona.name == persona_name)
    if user and user.role != UserRole.ADMIN:
        stmt = stmt.where(Persona.user_id == user.id)
    result = db_session.execute(stmt).scalar_one_or_none()
    return result


def make_persona_private(
    persona_id: int,
    user_ids: list[UUID] | None,
    group_ids: list[int] | None,
    db_session: Session,
) -> None:
    """
    将Persona设为私有
    
    参数:
        persona_id: Persona的ID
        user_ids: 允许访问的用户ID列表
        group_ids: 允许访问的用户组ID列表
        db_session: 数据库会话
    """
    if user_ids is not None:
        db_session.query(Persona__User).filter(
            Persona__User.persona_id == persona_id
        ).delete(synchronize_session="fetch")

        for user_uuid in user_ids:
            db_session.add(Persona__User(persona_id=persona_id, user_id=user_uuid))

        db_session.commit()

    # 如果有人从EE切换到MIT,可能会导致错误
    if group_ids:
        raise NotImplementedError("Onyx MIT does not支持私有Persona")


def create_update_persona(
    persona_id: int | None,
    create_persona_request: CreatePersonaRequest,
    user: User | None,
    db_session: Session,
) -> PersonaSnapshot:
    """
    创建或更新Persona
    
    参数:
        persona_id: Persona的ID,如果为None则创建新Persona
        create_persona_request: 创建Persona的请求数据
        user: 当前用户,用于权限验证
        db_session: 数据库会话
    """
    # 实际使用这些的权限稍后检查
    try:
        persona_data = {
            "persona_id": persona_id,
            "user": user,
            "db_session": db_session,
            **create_persona_request.model_dump(exclude={"users", "groups"}),
        }

        persona = upsert_persona(**persona_data)

        versioned_make_persona_private = fetch_versioned_implementation(
            "onyx.db.persona", "make_persona_private"
        )

        # 将Persona设为私有
        versioned_make_persona_private(
            persona_id=persona.id,
            user_ids=create_persona_request.users,
            group_ids=create_persona_request.groups,
            db_session=db_session,
        )

    except ValueError as e:
        logger.exception("创建Persona失败")
        raise HTTPException(status_code=400, detail=str(e))

    return PersonaSnapshot.from_model(persona)


def update_persona_shared_users(
    persona_id: int,
    user_ids: list[UUID],
    user: User | None,
    db_session: Session,
) -> None:
    """
    更新Persona的共享用户
    
    参数:
        persona_id: Persona的ID
        user_ids: 共享用户的ID列表
        user: 当前用户,用于权限验证
        db_session: 数据库会话
    """
    persona = fetch_persona_by_id(
        db_session=db_session, persona_id=persona_id, user=user, get_editable=True
    )

    if persona.is_public:
        raise HTTPException(status_code=400, detail="不能共享公共Persona")

    versioned_make_persona_private = fetch_versioned_implementation(
        "onyx.db.persona", "make_persona_private"
    )

    # 将Persona设为私有
    versioned_make_persona_private(
        persona_id=persona_id,
        user_ids=user_ids,
        group_ids=None,
        db_session=db_session,
    )


def update_persona_public_status(
    persona_id: int,
    is_public: bool,
    db_session: Session,
    user: User | None,
) -> None:
    """
    更新Persona的公共状态
    
    参数:
        persona_id: Persona的ID
        is_public: 是否设为公共
        db_session: 数据库会话
        user: 当前用户,用于权限验证
    """
    persona = fetch_persona_by_id(
        db_session=db_session, persona_id=persona_id, user=user, get_editable=True
    )
    if user and user.role != UserRole.ADMIN and persona.user_id != user.id:
        raise ValueError("你没有权限修改此Persona")

    persona.is_public = is_public
    db_session.commit()


def get_prompts(
    user_id: UUID | None,
    db_session: Session,
    include_default: bool = True,
    include_deleted: bool = False,
) -> Sequence[Prompt]:
    """
    获取提示词列表
    
    参数:
        user_id: 用户ID,如果为None则获取所有用户的提示词
        db_session: 数据库会话
        include_default: 是否包含默认提示词
        include_deleted: 是否包含已删除的提示词
    """
    stmt = select(Prompt).where(
        or_(Prompt.user_id == user_id, Prompt.user_id.is_(None))
    )

    if not include_default:
        stmt = stmt.where(Prompt.default_prompt.is_(False))
    if not include_deleted:
        stmt = stmt.where(Prompt.deleted.is_(False))

    return db_session.scalars(stmt).all()


def get_personas(
    user: User | None,
    db_session: Session,
    get_editable: bool = True,
    include_default: bool = True,
    include_slack_bot_personas: bool = False,
    include_deleted: bool = False,
    joinedload_all: bool = False,
) -> Sequence[Persona]:
    """
    获取Persona列表
    
    参数:
        user: 当前用户,用于权限验证
        db_session: 数据库会话
        get_editable: 是否只获取可编辑的内容
        include_default: 是否包含默认Persona
        include_slack_bot_personas: 是否包含Slack Bot Persona
        include_deleted: 是否包含已删除的Persona
        joinedload_all: 是否加载所有关联数据
    """
    stmt = select(Persona).distinct()
    stmt = _add_user_filters(stmt=stmt, user=user, get_editable=get_editable)
    if not include_default:
        stmt = stmt.where(Persona.builtin_persona.is_(False))
    if not include_slack_bot_personas:
        stmt = stmt.where(not_(Persona.name.startswith(SLACK_BOT_PERSONA_PREFIX)))
    if not include_deleted:
        stmt = stmt.where(Persona.deleted.is_(False))

    if joinedload_all:
        stmt = stmt.options(
            joinedload(Persona.prompts),
            joinedload(Persona.tools),
            joinedload(Persona.document_sets),
            joinedload(Persona.groups),
            joinedload(Persona.users),
        )

    return db_session.execute(stmt).unique().scalars().all()


def mark_persona_as_deleted(
    persona_id: int,
    user: User | None,
    db_session: Session,
) -> None:
    """
    将Persona标记为已删除
    
    参数:
        persona_id: Persona的ID
        user: 当前用户,用于权限验证
        db_session: 数据库会话
    """
    persona = get_persona_by_id(persona_id=persona_id, user=user, db_session=db_session)
    persona.deleted = True
    db_session.commit()


def mark_persona_as_not_deleted(
    persona_id: int,
    user: User | None,
    db_session: Session,
) -> None:
    """
    将Persona标记为未删除
    
    参数:
        persona_id: Persona的ID
        user: 当前用户,用于权限验证
        db_session: 数据库会话
    """
    persona = get_persona_by_id(
        persona_id=persona_id, user=user, db_session=db_session, include_deleted=True
    )
    if persona.deleted:
        persona.deleted = False
        db_session.commit()
    else:
        raise ValueError(f"ID为{persona_id}的Persona未被删除.")


def mark_delete_persona_by_name(
    persona_name: str, db_session: Session, is_default: bool = True
) -> None:
    """
    根据名称将Persona标记为已删除
    
    参数:
        persona_name: Persona的名称
        db_session: 数据库会话
        is_default: 是否为默认Persona
    """
    stmt = (
        update(Persona)
        .where(Persona.name == persona_name, Persona.builtin_persona == is_default)
        .values(deleted=True)
    )

    db_session.execute(stmt)
    db_session.commit()


def update_all_personas_display_priority(
    display_priority_map: dict[int, int],
    db_session: Session,
) -> None:
    """
    更新所有Persona的显示优先级
    
    参数:
        display_priority_map: Persona ID到显示优先级的映射
        db_session: 数据库会话
    """
    personas = get_personas(user=None, db_session=db_session)
    available_persona_ids = {persona.id for persona in personas}
    if available_persona_ids != set(display_priority_map.keys()):
        raise ValueError("提供的Persona ID无效")

    for persona in personas:
        persona.display_priority = display_priority_map[persona.id]
    db_session.commit()


def upsert_prompt(
    user: User | None,
    name: str,
    description: str,
    system_prompt: str,
    task_prompt: str,
    include_citations: bool,
    datetime_aware: bool,
    personas: list[Persona] | None,
    db_session: Session,
    prompt_id: int | None = None,
    default_prompt: bool = True,
    commit: bool = True,
) -> Prompt:
    """
    插入或更新提示词
    
    参数:
        user: 当前用户,用于权限验证
        name: 提示词名称
        description: 提示词描述
        system_prompt: 系统提示词
        task_prompt: 任务提示词
        include_citations: 是否包含引用
        datetime_aware: 是否包含日期时间
        personas: 关联的Persona列表
        db_session: 数据库会话
        prompt_id: 提示词ID,如果为None则创建新提示词
        default_prompt: 是否为默认提示词
        commit: 是否提交事务
    """
    if prompt_id is not None:
        prompt = db_session.query(Prompt).filter_by(id=prompt_id).first()
    else:
        prompt = get_prompt_by_name(prompt_name=name, user=user, db_session=db_session)

    if prompt:
        if not default_prompt and prompt.default_prompt:
            raise ValueError("不能使用非默认提示词更新默认提示词.")

        prompt.name = name
        prompt.description = description
        prompt.system_prompt = system_prompt
        prompt.task_prompt = task_prompt
        prompt.include_citations = include_citations
        prompt.datetime_aware = datetime_aware
        prompt.default_prompt = default_prompt

        if personas is not None:
            prompt.personas.clear()
            prompt.personas = personas

    else:
        prompt = Prompt(
            id=prompt_id,
            user_id=user.id if user else None,
            name=name,
            description=description,
            system_prompt=system_prompt,
            task_prompt=task_prompt,
            include_citations=include_citations,
            datetime_aware=datetime_aware,
            default_prompt=default_prompt,
            personas=personas or [],
        )
        db_session.add(prompt)

    if commit:
        db_session.commit()
    else:
        # 刷新会话以便Prompt有ID
        db_session.flush()

    return prompt


def upsert_persona(
    user: User | None,
    name: str,
    description: str,
    num_chunks: float,
    llm_relevance_filter: bool,
    llm_filter_extraction: bool,
    recency_bias: RecencyBiasSetting,
    llm_model_provider_override: str | None,
    llm_model_version_override: str | None,
    starter_messages: list[StarterMessage] | None,
    is_public: bool,
    db_session: Session,
    prompt_ids: list[int] | None = None,
    document_set_ids: list[int] | None = None,
    tool_ids: list[int] | None = None,
    persona_id: int | None = None,
    commit: bool = True,
    icon_color: str | None = None,
    icon_shape: int | None = None,
    uploaded_image_id: str | None = None,
    display_priority: int | None = None,
    is_visible: bool = True,
    remove_image: bool | None = None,
    search_start_date: datetime | None = None,
    builtin_persona: bool = False,
    is_default_persona: bool = False,
    category_id: int | None = None,
    chunks_above: int = CONTEXT_CHUNKS_ABOVE,
    chunks_below: int = CONTEXT_CHUNKS_BELOW,
) -> Persona:
    """
    插入或更新Persona
    
    参数:
        user: 当前用户,用于权限验证
        name: Persona名称
        description: Persona描述
        num_chunks: 块数量
        llm_relevance_filter: 是否启用LLM相关性过滤
        llm_filter_extraction: 是否启用LLM过滤提取
        recency_bias: 最近偏好设置
        llm_model_provider_override: LLM模型提供者覆盖
        llm_model_version_override: LLM模型版本覆盖
        starter_messages: 启动消息列表
        is_public: 是否为公共Persona
        db_session: 数据库会话
        prompt_ids: 提示词ID列表
        document_set_ids: 文档集ID列表
        tool_ids: 工具ID列表
        persona_id: Persona ID,如果为None则创建新Persona
        commit: 是否提交事务
        icon_color: 图标颜色
        icon_shape: 图标形状
        uploaded_image_id: 上传的图片ID
        display_priority: 显示优先级
        is_visible: 是否可见
        remove_image: 是否移除图片
        search_start_date: 搜索开始日期
        builtin_persona: 是否为内置Persona
        is_default_persona: 是否为默认Persona
        category_id: 类别ID
        chunks_above: 上方块数量
        chunks_below: 下方块数量
    """
    if persona_id is not None:
        existing_persona = db_session.query(Persona).filter_by(id=persona_id).first()
    else:
        existing_persona = _get_persona_by_name(
            persona_name=name, user=user, db_session=db_session
        )

    # 获取并附加工具
    tools = None
    if tool_ids is not None:
        tools = db_session.query(Tool).filter(Tool.id.in_(tool_ids)).all()
        if not tools and tool_ids:
            raise ValueError("工具未找到")

    # 获取并附加文档集
    document_sets = None
    if document_set_ids is not None:
        document_sets = (
            db_session.query(DocumentSet)
            .filter(DocumentSet.id.in_(document_set_ids))
            .all()
        )
        if not document_sets and document_set_ids:
            raise ValueError("文档集未找到")

    # 获取并附加提示词
    prompts = None
    if prompt_ids is not None:
        prompts = db_session.query(Prompt).filter(Prompt.id.in_(prompt_ids)).all()

    if prompts is not None and len(prompts) == 0:
        raise ValueError(
            f"无效的Persona配置,未指定有效的提示词. 指定的ID为: '{prompt_ids}'"
        )

    # 确保所有指定的工具有效
    if tools:
        validate_persona_tools(tools)

    if existing_persona:
        # 内置Persona只能通过YAML配置更新
        # 这确保了核心系统Persona不会被意外修改
        if existing_persona.builtin_persona and not builtin_persona:
            raise ValueError("不能使用非内置Persona更新内置Persona.")

        # 检查用户是否有权限编辑Persona
        # 如果用户没有权限,将引发异常
        existing_persona = fetch_persona_by_id(
            db_session=db_session,
            persona_id=existing_persona.id,
            user=user,
            get_editable=True,
        )

        # 以下更新不包括`default`, `built-in`和显示优先级
        # 显示优先级在`display-priority`端点中单独处理
        # `default`和`built-in`属性只能在创建Persona时设置
        existing_persona.name = name
        existing_persona.description = description
        existing_persona.num_chunks = num_chunks
        existing_persona.chunks_above = chunks_above
        existing_persona.chunks_below = chunks_below
        existing_persona.llm_relevance_filter = llm_relevance_filter
        existing_persona.llm_filter_extraction = llm_filter_extraction
        existing_persona.recency_bias = recency_bias
        existing_persona.llm_model_provider_override = llm_model_provider_override
        existing_persona.llm_model_version_override = llm_model_version_override
        existing_persona.starter_messages = starter_messages
        existing_persona.deleted = False  # 如果之前已删除,则取消删除
        existing_persona.is_public = is_public
        existing_persona.icon_color = icon_color
        existing_persona.icon_shape = icon_shape
        if remove_image or uploaded_image_id:
            existing_persona.uploaded_image_id = uploaded_image_id
        existing_persona.is_visible = is_visible
        existing_persona.search_start_date = search_start_date
        existing_persona.category_id = category_id
        # 不要删除任何手动添加的关联,除非提供了新的更新列表
        if document_sets is not None:
            existing_persona.document_sets.clear()
            existing_persona.document_sets = document_sets or []

        if prompts is not None:
            existing_persona.prompts.clear()
            existing_persona.prompts = prompts

        if tools is not None:
            existing_persona.tools = tools or []

        # 仅在未设置显示优先级时更新显示优先级
        if existing_persona.display_priority is None:
            existing_persona.display_priority = display_priority

        persona = existing_persona

    else:
        if not prompts:
            raise ValueError(
                "无效的Persona配置. 必须为新Persona指定至少一个提示词."
            )

        new_persona = Persona(
            id=persona_id,
            user_id=user.id if user else None,
            is_public=is_public,
            name=name,
            description=description,
            num_chunks=num_chunks,
            chunks_above=chunks_above,
            chunks_below=chunks_below,
            llm_relevance_filter=llm_relevance_filter,
            llm_filter_extraction=llm_filter_extraction,
            recency_bias=recency_bias,
            builtin_persona=builtin_persona,
            prompts=prompts,
            document_sets=document_sets or [],
            llm_model_provider_override=llm_model_provider_override,
            llm_model_version_override=llm_model_version_override,
            starter_messages=starter_messages,
            tools=tools or [],
            icon_shape=icon_shape,
            icon_color=icon_color,
            uploaded_image_id=uploaded_image_id,
            display_priority=display_priority,
            is_visible=is_visible,
            search_start_date=search_start_date,
            is_default_persona=is_default_persona,
            category_id=category_id,
        )
        db_session.add(new_persona)
        persona = new_persona
    if commit:
        db_session.commit()
    else:
        # 刷新会话以便Persona有ID
        db_session.flush()

    return persona


def mark_prompt_as_deleted(
    prompt_id: int,
    user: User | None,
    db_session: Session,
) -> None:
    """
    将提示词标记为已删除
    
    参数:
        prompt_id: 提示词ID
        user: 当前用户,用于权限验证
        db_session: 数据库会话
    """
    prompt = get_prompt_by_id(prompt_id=prompt_id, user=user, db_session=db_session)
    prompt.deleted = True
    db_session.commit()


def delete_old_default_personas(
    db_session: Session,
) -> None:
    """
    删除旧的默认Persona
    
    注意,这会暂时锁定Summarize和Paraphrase Persona
    需要稍后更优雅的修复,或者这些Persona永远不应有ID
    """
    stmt = (
        update(Persona)
        .where(Persona.builtin_persona, Persona.id > 0)
        .values(deleted=True, name=func.concat(Persona.name, "_old"))
    )

    db_session.execute(stmt)
    db_session.commit()


def update_persona_visibility(
    persona_id: int,
    is_visible: bool,
    db_session: Session,
    user: User | None = None,
) -> None:
    """
    更新Persona的可见性
    
    参数:
        persona_id: Persona的ID
        is_visible: 是否可见
        db_session: 数据库会话
        user: 当前用户,用于权限验证
    """
    persona = fetch_persona_by_id(
        db_session=db_session, persona_id=persona_id, user=user, get_editable=True
    )

    persona.is_visible = is_visible
    db_session.commit()


def validate_persona_tools(tools: list[Tool]) -> None:
    """
    验证Persona的工具
    
    参数:
        tools: 工具列表
    """
    for tool in tools:
        if tool.name == "InternetSearchTool" and not BING_API_KEY:
            raise ValueError(
                "未找到Bing API密钥,请联系您的Onyx管理员以获取添加!"
            )


def get_prompts_by_ids(prompt_ids: list[int], db_session: Session) -> list[Prompt]:
    """
    根据ID列表获取提示词
    
    参数:
        prompt_ids: 提示词ID列表
        db_session: 数据库会话
    """
    if not prompt_ids:
        return []
    prompts = db_session.scalars(
        select(Prompt).where(Prompt.id.in_(prompt_ids)).where(Prompt.deleted.is_(False))
    ).all()

    return list(prompts)


def get_prompt_by_id(
    prompt_id: int,
    user: User | None,
    db_session: Session,
    include_deleted: bool = False,
) -> Prompt:
    """
    根据ID获取提示词
    
    参数:
        prompt_id: 提示词ID
        user: 当前用户,用于权限验证
        db_session: 数据库会话
        include_deleted: 是否包含已删除的提示词
    """
    stmt = select(Prompt).where(Prompt.id == prompt_id)

    # 如果未指定用户或他们是管理员,他们应该
    # 访问所有提示词,因此不需要此where子句
    if user and user.role != UserRole.ADMIN:
        stmt = stmt.where(or_(Prompt.user_id == user.id, Prompt.user_id.is_(None)))

    if not include_deleted:
        stmt = stmt.where(Prompt.deleted.is_(False))

    result = db_session.execute(stmt)
    prompt = result.scalar_one_or_none()

    if prompt is None:
        raise ValueError(
            f"ID为{prompt_id}的提示词不存在或不属于用户"
        )

    return prompt


def _get_default_prompt(db_session: Session) -> Prompt:
    """
    获取默认提示词
    
    参数:
        db_session: 数据库会话
    """
    stmt = select(Prompt).where(Prompt.id == 0)
    result = db_session.execute(stmt)
    prompt = result.scalar_one_or_none()

    if prompt is None:
        raise RuntimeError("未找到默认提示词")

    return prompt


def get_default_prompt(db_session: Session) -> Prompt:
    """
    获取默认提示词
    
    参数:
        db_session: 数据库会话
    """
    return _get_default_prompt(db_session)


@lru_cache()
def get_default_prompt__read_only() -> Prompt:
    """
    由于lru_cache / SQLAlchemy的工作方式,这可能会导致问题
    当尝试将返回的`Prompt`对象附加到`Persona`时. 如果你
    除了读取之外的任何操作,你应该使用`get_default_prompt`
    方法.
    """
    with Session(get_sqlalchemy_engine()) as db_session:
        return _get_default_prompt(db_session)


def get_persona_by_id(
    persona_id: int,
    user: User | None,
    db_session: Session,
    include_deleted: bool = False,
    is_for_edit: bool = True,  # 默认假定为true以确保安全
) -> Persona:
    """
    根据ID获取Persona对象
    
    参数:
        persona_id: Persona的ID
        user: 当前用户,用于权限验证
        db_session: 数据库会话
        include_deleted: 是否包含已删除的Persona
        is_for_edit: 是否用于编辑
    """
    persona_stmt = (
        select(Persona)
        .distinct()
        .outerjoin(Persona.groups)
        .outerjoin(Persona.users)
        .outerjoin(UserGroup.user_group_relationships)
        .where(Persona.id == persona_id)
    )

    if not include_deleted:
        persona_stmt = persona_stmt.where(Persona.deleted.is_(False))

    if not user or user.role == UserRole.ADMIN:
        result = db_session.execute(persona_stmt)
        persona = result.scalar_one_or_none()
        if persona is None:
            raise ValueError(f"ID为{persona_id}的Persona不存在")
        return persona

    # 或检查用户是否拥有Persona
    or_conditions = Persona.user_id == user.id
    # 允许访问如果Persona用户ID为None
    or_conditions |= Persona.user_id == None  # noqa: E711
    if not is_for_edit:
        # 如果用户在与Persona相关的组中
        or_conditions |= User__UserGroup.user_id == user.id
        # 如果用户在Persona的.users中
        or_conditions |= User.id == user.id
        or_conditions |= Persona.is_public == True  # noqa: E712
    elif user.role == UserRole.GLOBAL_CURATOR:
        # 全局策展人可以编辑他们所在组的Persona
        or_conditions |= User__UserGroup.user_id == user.id
    elif user.role == UserRole.CURATOR:
        # 策展人可以编辑他们是策展人的组的Persona
        or_conditions |= (User__UserGroup.user_id == user.id) & (
            User__UserGroup.is_curator == True  # noqa: E712
        )

    persona_stmt = persona_stmt.where(or_conditions)
    result = db_session.execute(persona_stmt)
    persona = result.scalar_one_or_none()
    if persona is None:
        raise ValueError(
            f"ID为{persona_id}的Persona不存在或不属于用户"
        )
    return persona


def get_personas_by_ids(
    persona_ids: list[int], db_session: Session
) -> Sequence[Persona]:
    """
    根据ID列表获取Persona对象
    
    参数:
        persona_ids: Persona ID列表
        db_session: 数据库会话
    """
    if not persona_ids:
        return []
    personas = db_session.scalars(
        select(Persona).where(Persona.id.in_(persona_ids))
    ).all()

    return personas


def get_prompt_by_name(
    prompt_name: str, user: User | None, db_session: Session
) -> Prompt | None:
    """
    根据名称获取提示词
    
    参数:
        prompt_name: 提示词名称
        user: 当前用户,用于权限验证
        db_session: 数据库会话
    """
    stmt = select(Prompt).where(Prompt.name == prompt_name)

    # 如果未指定用户或他们是管理员,他们应该
    # 访问所有提示词,因此不需要此where子句
    if user and user.role != UserRole.ADMIN:
        stmt = stmt.where(Prompt.user_id == user.id)

    # 按ID排序以确保在存在多个提示词时结果一致
    stmt = stmt.order_by(Prompt.id).limit(1)
    result = db_session.execute(stmt).scalar_one_or_none()
    return result


def delete_persona_by_name(
    persona_name: str, db_session: Session, is_default: bool = True
) -> None:
    """
    根据名称删除Persona
    
    参数:
        persona_name: Persona的名称
        db_session: 数据库会话
        is_default: 是否为默认Persona
    """
    stmt = delete(Persona).where(
        Persona.name == persona_name, Persona.builtin_persona == is_default
    )

    db_session.execute(stmt)
    db_session.commit()


def get_assistant_categories(db_session: Session) -> list[PersonaCategory]:
    """
    获取助手类别列表
    
    参数:
        db_session: 数据库会话
    """
    return db_session.query(PersonaCategory).all()


def create_assistant_category(
    db_session: Session, name: str, description: str
) -> PersonaCategory:
    """
    创建助手类别
    
    参数:
        db_session: 数据库会话
        name: 类别名称
        description: 类别描述
    """
    category = PersonaCategory(name=name, description=description)
    db_session.add(category)
    db_session.commit()
    return category


def update_persona_category(
    category_id: int,
    category_description: str,
    category_name: str,
    db_session: Session,
) -> None:
    """
    更新助手类别
    
    参数:
        category_id: 类别ID
        category_description: 类别描述
        category_name: 类别名称
        db_session: 数据库会话
    """
    persona_category = (
        db_session.query(PersonaCategory)
        .filter(PersonaCategory.id == category_id)
        .one_or_none()
    )
    if persona_category is None:
        raise ValueError(f"ID为{category_id}的助手类别不存在")
    persona_category.description = category_description
    persona_category.name = category_name
    db_session.commit()


def delete_persona_category(category_id: int, db_session: Session) -> None:
    """
    删除助手类别
    
    参数:
        category_id: 类别ID
        db_session: 数据库会话
    """
    db_session.query(PersonaCategory).filter(PersonaCategory.id == category_id).delete()
    db_session.commit()
