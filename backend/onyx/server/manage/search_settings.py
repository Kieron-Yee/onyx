"""
这个文件主要包含了搜索设置相关的API路由处理函数。
主要功能包括：
- 管理搜索设置（创建、更新、删除）
- 管理嵌入模型配置
- 处理非结构化API密钥
"""

from fastapi import APIRouter
from fastapi import Depends
from fastapi import HTTPException
from fastapi import status
from sqlalchemy.orm import Session

from onyx.auth.users import current_admin_user
from onyx.auth.users import current_user
from onyx.configs.app_configs import DISABLE_INDEX_UPDATE_ON_SWAP
from onyx.context.search.models import SavedSearchSettings
from onyx.context.search.models import SearchSettingsCreationRequest
from onyx.db.connector_credential_pair import get_connector_credential_pairs
from onyx.db.connector_credential_pair import resync_cc_pair
from onyx.db.engine import get_session
from onyx.db.index_attempt import expire_index_attempts
from onyx.db.models import IndexModelStatus
from onyx.db.models import User
from onyx.db.search_settings import create_search_settings
from onyx.db.search_settings import delete_search_settings
from onyx.db.search_settings import get_current_search_settings
from onyx.db.search_settings import get_embedding_provider_from_provider_type
from onyx.db.search_settings import get_secondary_search_settings
from onyx.db.search_settings import update_current_search_settings
from onyx.db.search_settings import update_search_settings_status
from onyx.document_index.factory import get_default_document_index
from onyx.file_processing.unstructured import delete_unstructured_api_key
from onyx.file_processing.unstructured import get_unstructured_api_key
from onyx.file_processing.unstructured import update_unstructured_api_key
from onyx.natural_language_processing.search_nlp_models import clean_model_name
from onyx.server.manage.embedding.models import SearchSettingsDeleteRequest
from onyx.server.manage.models import FullModelVersionResponse
from onyx.server.models import IdReturn
from onyx.utils.logger import setup_logger
from shared_configs.configs import ALT_INDEX_SUFFIX

router = APIRouter(prefix="/search-settings")
logger = setup_logger()


@router.post("/set-new-search-settings")
def set_new_search_settings(
    search_settings_new: SearchSettingsCreationRequest,
    _: User | None = Depends(current_admin_user),
    db_session: Session = Depends(get_session),
) -> IdReturn:
    """Creates a new EmbeddingModel row and cancels the previous secondary indexing if any
    创建新的嵌入模型记录并取消之前的二级索引（如果存在）
    如果使用了与当前或二级索引相同的模型名称，则会报错

    参数:
        search_settings_new: 新的搜索设置请求对象
        _: 当前管理员用户（用于权限验证）
        db_session: 数据库会话

    返回:
        IdReturn: 新创建的搜索设置ID
    """
    if search_settings_new.index_name:
        logger.warning("Index name was specified by request, this is not suggested")
        # 索引名称由请求指定，这不建议这样做

    # Validate cloud provider exists or create new LiteLLM provider
    # 验证云服务提供商是否存在或创建新的LiteLLM提供商
    if search_settings_new.provider_type is not None:
        cloud_provider = get_embedding_provider_from_provider_type(
            db_session, provider_type=search_settings_new.provider_type
        )

        if cloud_provider is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"No embedding provider exists for cloud embedding type {search_settings_new.provider_type}",
            )

    search_settings = get_current_search_settings(db_session)

    if search_settings_new.index_name is None:
        # We define index name here
        # 在这里定义索引名称
        index_name = f"danswer_chunk_{clean_model_name(search_settings_new.model_name)}"
        if (
            search_settings_new.model_name == search_settings.model_name
            and not search_settings.index_name.endswith(ALT_INDEX_SUFFIX)
        ):
            index_name += ALT_INDEX_SUFFIX
        search_values = search_settings_new.dict()
        search_values["index_name"] = index_name
        new_search_settings_request = SavedSearchSettings(**search_values)
    else:
        new_search_settings_request = SavedSearchSettings(**search_settings_new.dict())

    secondary_search_settings = get_secondary_search_settings(db_session)

    if secondary_search_settings:
        # Cancel any background indexing jobs
        # 取消所有后台索引任务
        expire_index_attempts(
            search_settings_id=secondary_search_settings.id, db_session=db_session
        )

        # Mark previous model as a past model directly
        # 直接将先前的模型标记为过去的模型
        update_search_settings_status(
            search_settings=secondary_search_settings,
            new_status=IndexModelStatus.PAST,
            db_session=db_session,
        )

    new_search_settings = create_search_settings(
        search_settings=new_search_settings_request, db_session=db_session
    )

    # Ensure Vespa has the new index immediately
    # 确保Vespa立即拥有新索引
    document_index = get_default_document_index(
        primary_index_name=search_settings.index_name,
        secondary_index_name=new_search_settings.index_name,
    )

    document_index.ensure_indices_exist(
        index_embedding_dim=search_settings.model_dim,
        secondary_index_embedding_dim=new_search_settings.model_dim,
    )

    # Pause index attempts for the currently in use index to preserve resources
    # 暂停当前使用中的索引的索引尝试以保留资源
    if DISABLE_INDEX_UPDATE_ON_SWAP:
        expire_index_attempts(
            search_settings_id=search_settings.id, db_session=db_session
        )
        for cc_pair in get_connector_credential_pairs(db_session):
            resync_cc_pair(cc_pair, db_session=db_session)

    db_session.commit()
    return IdReturn(id=new_search_settings.id)


@router.post("/cancel-new-embedding")
def cancel_new_embedding(
    _: User | None = Depends(current_admin_user),
    db_session: Session = Depends(get_session),
) -> None:
    """取消新的嵌入模型设置
    
    参数:
        _: 当前管理员用户（用于权限验证）
        db_session: 数据库会话
    """
    secondary_search_settings = get_secondary_search_settings(db_session)

    if secondary_search_settings:
        expire_index_attempts(
            search_settings_id=secondary_search_settings.id, db_session=db_session
        )

        update_search_settings_status(
            search_settings=secondary_search_settings,
            new_status=IndexModelStatus.PAST,
            db_session=db_session,
        )


@router.delete("/delete-search-settings")
def delete_search_settings_endpoint(
    deletion_request: SearchSettingsDeleteRequest,
    _: User | None = Depends(current_admin_user),
    db_session: Session = Depends(get_session),
) -> None:
    """删除指定的搜索设置
    
    参数:
        deletion_request: 删除请求对象，包含要删除的搜索设置ID
        _: 当前管理员用户（用于权限验证）
        db_session: 数据库会话
    """
    try:
        delete_search_settings(
            db_session=db_session,
            search_settings_id=deletion_request.search_settings_id,
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.get("/get-current-search-settings")
def get_current_search_settings_endpoint(
    _: User | None = Depends(current_user),
    db_session: Session = Depends(get_session),
) -> SavedSearchSettings:
    """获取当前的搜索设置
    
    参数:
        _: 当前用户（用于权限验证）
        db_session: 数据库会话
        
    返回:
        SavedSearchSettings: 当前搜索设置对象
    """
    current_search_settings = get_current_search_settings(db_session)
    return SavedSearchSettings.from_db_model(current_search_settings)


@router.get("/get-secondary-search-settings")
def get_secondary_search_settings_endpoint(
    _: User | None = Depends(current_user),
    db_session: Session = Depends(get_session),
) -> SavedSearchSettings | None:
    """获取二级搜索设置
    
    参数:
        _: 当前用户（用于权限验证）
        db_session: 数据库会话
        
    返回:
        SavedSearchSettings | None: 二级搜索设置对象，如果不存在则返回None
    """
    secondary_search_settings = get_secondary_search_settings(db_session)
    if not secondary_search_settings:
        return None

    return SavedSearchSettings.from_db_model(secondary_search_settings)


@router.get("/get-all-search-settings")
def get_all_search_settings(
    _: User | None = Depends(current_user),
    db_session: Session = Depends(get_session),
) -> FullModelVersionResponse:
    """获取所有搜索设置，包括当前设置和二级设置
    
    参数:
        _: 当前用户（用于权限验证）
        db_session: 数据库会话
        
    返回:
        FullModelVersionResponse: 包含当前和二级搜索设置的完整响应对象
    """
    current_search_settings = get_current_search_settings(db_session)
    secondary_search_settings = get_secondary_search_settings(db_session)
    return FullModelVersionResponse(
        current_settings=SavedSearchSettings.from_db_model(current_search_settings),
        secondary_settings=SavedSearchSettings.from_db_model(secondary_search_settings)
        if secondary_search_settings
        else None,
    )


# Updates current non-reindex search settings
# 更新当前的非重索引搜索设置
@router.post("/update-inference-settings")
def update_saved_search_settings(
    search_settings: SavedSearchSettings,
    _: User | None = Depends(current_admin_user),
    db_session: Session = Depends(get_session),
) -> None:
    """更新当前的推理设置（不重新索引）
    
    参数:
        search_settings: 要更新的搜索设置
        _: 当前管理员用户（用于权限验证）
        db_session: 数据库会话
    """
    update_current_search_settings(
        search_settings=search_settings, db_session=db_session
    )


@router.get("/unstructured-api-key-set")
def unstructured_api_key_set(
    _: User | None = Depends(current_admin_user),
) -> bool:
    """检查非结构化API密钥是否已设置
    
    参数:
        _: 当前管理员用户（用于权限验证）
        
    返回:
        bool: API密钥是否已设置
    """
    api_key = get_unstructured_api_key()
    print(api_key)
    return api_key is not None


@router.put("/upsert-unstructured-api-key")
def upsert_unstructured_api_key(
    unstructured_api_key: str,
    _: User | None = Depends(current_admin_user),
) -> None:
    """更新或插入非结构化API密钥
    
    参数:
        unstructured_api_key: 新的API密钥
        _: 当前管理员用户（用于权限验证）
    """
    update_unstructured_api_key(unstructured_api_key)


@router.delete("/delete-unstructured-api-key")
def delete_unstructured_api_key_endpoint(
    _: User | None = Depends(current_admin_user),
) -> None:
    """删除非结构化API密钥
    
    参数:
        _: 当前管理员用户（用于权限验证）
    """
    delete_unstructured_api_key()
