"""
此文件实现了LLM(大语言模型)管理相关的API接口。
主要功能包括：
1. LLM提供商的配置管理（增删改查）
2. LLM测试接口
3. 默认LLM提供商设置
4. LLM提供商信息查询接口
"""

from collections.abc import Callable

from fastapi import APIRouter
from fastapi import Depends
from fastapi import HTTPException
from fastapi import Query
from sqlalchemy.orm import Session

from onyx.auth.users import current_admin_user
from onyx.auth.users import current_chat_accesssible_user
from onyx.db.engine import get_session
from onyx.db.llm import fetch_existing_llm_providers
from onyx.db.llm import fetch_provider
from onyx.db.llm import remove_llm_provider
from onyx.db.llm import update_default_provider
from onyx.db.llm import upsert_llm_provider
from onyx.db.models import User
from onyx.llm.factory import get_default_llms
from onyx.llm.factory import get_llm
from onyx.llm.llm_provider_options import fetch_available_well_known_llms
from onyx.llm.llm_provider_options import WellKnownLLMProviderDescriptor
from onyx.llm.utils import litellm_exception_to_error_msg
from onyx.llm.utils import test_llm
from onyx.server.manage.llm.models import FullLLMProvider
from onyx.server.manage.llm.models import LLMProviderDescriptor
from onyx.server.manage.llm.models import LLMProviderUpsertRequest
from onyx.server.manage.llm.models import TestLLMRequest
from onyx.utils.logger import setup_logger
from onyx.utils.threadpool_concurrency import run_functions_tuples_in_parallel

logger = setup_logger()

# 创建管理员路由和基础路由
admin_router = APIRouter(prefix="/admin/llm")
basic_router = APIRouter(prefix="/llm")


@admin_router.get("/built-in/options")
def fetch_llm_options(
    _: User | None = Depends(current_admin_user),
) -> list[WellKnownLLMProviderDescriptor]:
    """
    获取内置的LLM选项列表
    
    Args:
        _: 当前管理员用户，通过依赖注入获取
        
    Returns:
        list[WellKnownLLMProviderDescriptor]: 返回可用的内置LLM提供商描述符列表
    """
    return fetch_available_well_known_llms()


@admin_router.post("/test")
def test_llm_configuration(
    test_llm_request: TestLLMRequest,
    _: User | None = Depends(current_admin_user),
) -> None:
    """
    测试LLM配置是否可用
    
    Args:
        test_llm_request: LLM测试请求参数
        _: 当前管理员用户，通过依赖注入获取
        
    Raises:
        HTTPException: 如果测试失败则抛出400错误
    """
    llm = get_llm(
        provider=test_llm_request.provider,
        model=test_llm_request.default_model_name,
        api_key=test_llm_request.api_key,
        api_base=test_llm_request.api_base,
        api_version=test_llm_request.api_version,
        custom_config=test_llm_request.custom_config,
        deployment_name=test_llm_request.deployment_name,
    )

    functions_with_args: list[tuple[Callable, tuple]] = [(test_llm, (llm,))]
    if (
        test_llm_request.fast_default_model_name
        and test_llm_request.fast_default_model_name
        != test_llm_request.default_model_name
    ):
        fast_llm = get_llm(
            provider=test_llm_request.provider,
            model=test_llm_request.fast_default_model_name,
            api_key=test_llm_request.api_key,
            api_base=test_llm_request.api_base,
            api_version=test_llm_request.api_version,
            custom_config=test_llm_request.custom_config,
            deployment_name=test_llm_request.deployment_name,
        )
        functions_with_args.append((test_llm, (fast_llm,)))

    parallel_results = run_functions_tuples_in_parallel(
        functions_with_args, allow_failures=False
    )
    error = parallel_results[0] or (
        parallel_results[1] if len(parallel_results) > 1 else None
    )

    if error:
        client_error_msg = litellm_exception_to_error_msg(
            error, llm, fallback_to_error_msg=True
        )
        raise HTTPException(status_code=400, detail=client_error_msg)


@admin_router.post("/test/default")
def test_default_provider(
    _: User | None = Depends(current_admin_user),
) -> None:
    """
    测试默认LLM提供商配置是否可用
    
    Args:
        _: 当前管理员用户，通过依赖注入获取
        
    Raises:
        HTTPException: 如果没有设置默认提供商或测试失败则抛出400错误
    """
    try:
        llm, fast_llm = get_default_llms()
    except ValueError:
        logger.exception("Failed to fetch default LLM Provider")
        raise HTTPException(status_code=400, detail="No LLM Provider setup")

    functions_with_args: list[tuple[Callable, tuple]] = [
        (test_llm, (llm,)),
        (test_llm, (fast_llm,)),
    ]
    parallel_results = run_functions_tuples_in_parallel(
        functions_with_args, allow_failures=False
    )
    error = parallel_results[0] or (
        parallel_results[1] if len(parallel_results) > 1 else None
    )
    if error:
        raise HTTPException(status_code=400, detail=error)


@admin_router.get("/provider")
def list_llm_providers(
    _: User | None = Depends(current_admin_user),
    db_session: Session = Depends(get_session),
) -> list[FullLLMProvider]:
    """
    获取所有LLM提供商列表
    
    Args:
        _: 当前管理员用户，通过依赖注入获取
        db_session: 数据库会话，通过依赖注入获取
        
    Returns:
        list[FullLLMProvider]: 返回所有LLM提供商的完整信息列表
    """
    return [
        FullLLMProvider.from_model(llm_provider_model)
        for llm_provider_model in fetch_existing_llm_providers(db_session)
    ]


@admin_router.put("/provider")
def put_llm_provider(
    llm_provider: LLMProviderUpsertRequest,
    is_creation: bool = Query(
        False,
        description="True if updating an existing provider, False if creating a new one",  # True表示更新现有提供商，False表示创建新提供商
    ),
    _: User | None = Depends(current_admin_user),
    db_session: Session = Depends(get_session),
) -> FullLLMProvider:
    """
    创建或更新LLM提供商配置
    
    Args:
        llm_provider: LLM提供商配置信息
        is_creation: 是否为创建操作
        _: 当前管理员用户，通过依赖注入获取
        db_session: 数据库会话，通过依赖注入获取
        
    Returns:
        FullLLMProvider: 返回更新后的LLM提供商完整信息
        
    Raises:
        HTTPException: 如果创建时提供商已存在，或更新操作失败则抛出400错误
    """
    # validate request (e.g. if we're intending to create but the name already exists we should throw an error)
    # NOTE: may involve duplicate fetching to Postgres, but we're assuming SQLAlchemy is smart enough to cache
    # the result
    existing_provider = fetch_provider(db_session, llm_provider.name)
    if existing_provider and is_creation:
        raise HTTPException(
            status_code=400,
            detail=f"LLM Provider with name {llm_provider.name} already exists",
        )

    # Ensure default_model_name and fast_default_model_name are in display_model_names
    # This is necessary for custom models and Bedrock/Azure models
    if llm_provider.display_model_names is None:
        llm_provider.display_model_names = []

    if llm_provider.default_model_name not in llm_provider.display_model_names:
        llm_provider.display_model_names.append(llm_provider.default_model_name)

    if (
        llm_provider.fast_default_model_name
        and llm_provider.fast_default_model_name not in llm_provider.display_model_names
    ):
        llm_provider.display_model_names.append(llm_provider.fast_default_model_name)

    try:
        return upsert_llm_provider(
            llm_provider=llm_provider,
            db_session=db_session,
        )
    except ValueError as e:
        logger.exception("Failed to upsert LLM Provider")
        raise HTTPException(status_code=400, detail=str(e))


@admin_router.delete("/provider/{provider_id}")
def delete_llm_provider(
    provider_id: int,
    _: User | None = Depends(current_admin_user),
    db_session: Session = Depends(get_session),
) -> None:
    """
    删除指定的LLM提供商
    
    Args:
        provider_id: 要删除的LLM提供商ID
        _: 当前管理员用户，通过依赖注入获取
        db_session: 数据库会话，通过依赖注入获取
    """
    remove_llm_provider(db_session, provider_id)


@admin_router.post("/provider/{provider_id}/default")
def set_provider_as_default(
    provider_id: int,
    _: User | None = Depends(current_admin_user),
    db_session: Session = Depends(get_session),
) -> None:
    """
    将指定的LLM提供商设置为默认提供商
    
    Args:
        provider_id: 要设置为默认的LLM提供商ID
        _: 当前管理员用户，通过依赖注入获取
        db_session: 数据库会话，通过依赖注入获取
    """
    update_default_provider(provider_id=provider_id, db_session=db_session)


"""Endpoints for all"""  # 所有用户可访问的接口


@basic_router.get("/provider")
def list_llm_provider_basics(
    user: User | None = Depends(current_chat_accesssible_user),
    db_session: Session = Depends(get_session),
) -> list[LLMProviderDescriptor]:
    """
    获取当前用户可访问的LLM提供商基本信息列表
    
    Args:
        user: 当前用户，通过依赖注入获取
        db_session: 数据库会话，通过依赖注入获取
        
    Returns:
        list[LLMProviderDescriptor]: 返回用户可访问的LLM提供商基本信息列表
    """
    return [
        LLMProviderDescriptor.from_model(llm_provider_model)
        for llm_provider_model in fetch_existing_llm_providers(db_session, user)
    ]
