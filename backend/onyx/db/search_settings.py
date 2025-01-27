"""
这个文件主要负责搜索设置的数据库操作，包括创建、获取、更新和删除搜索设置等功能。
主要处理与搜索相关的配置管理，如嵌入模型设置、重排序设置等。
"""

from sqlalchemy import and_
from sqlalchemy import delete
from sqlalchemy import select
from sqlalchemy.orm import Session

from onyx.configs.model_configs import ASYM_PASSAGE_PREFIX
from onyx.configs.model_configs import ASYM_QUERY_PREFIX
from onyx.configs.model_configs import DEFAULT_DOCUMENT_ENCODER_MODEL
from onyx.configs.model_configs import DOC_EMBEDDING_DIM
from onyx.configs.model_configs import DOCUMENT_ENCODER_MODEL
from onyx.configs.model_configs import NORMALIZE_EMBEDDINGS
from onyx.configs.model_configs import OLD_DEFAULT_DOCUMENT_ENCODER_MODEL
from onyx.configs.model_configs import OLD_DEFAULT_MODEL_DOC_EMBEDDING_DIM
from onyx.configs.model_configs import OLD_DEFAULT_MODEL_NORMALIZE_EMBEDDINGS
from onyx.context.search.models import SavedSearchSettings
from onyx.db.engine import get_session_with_default_tenant
from onyx.db.llm import fetch_embedding_provider
from onyx.db.models import CloudEmbeddingProvider
from onyx.db.models import IndexAttempt
from onyx.db.models import IndexModelStatus
from onyx.db.models import SearchSettings
from onyx.indexing.models import IndexingSetting
from onyx.natural_language_processing.search_nlp_models import clean_model_name
from onyx.natural_language_processing.search_nlp_models import warm_up_cross_encoder
from onyx.server.manage.embedding.models import (
    CloudEmbeddingProvider as ServerCloudEmbeddingProvider,
)
from onyx.utils.logger import setup_logger
from shared_configs.configs import PRESERVED_SEARCH_FIELDS
from shared_configs.enums import EmbeddingProvider

logger = setup_logger()

"""
创建搜索设置
参数：
    search_settings: 保存的搜索设置
    db_session: 数据库会话
    status: 索引模型状态，默认为FUTURE
返回：
    创建的SearchSettings对象
"""
def create_search_settings(
    search_settings: SavedSearchSettings,
    db_session: Session,
    status: IndexModelStatus = IndexModelStatus.FUTURE,
) -> SearchSettings:
    embedding_model = SearchSettings(
        model_name=search_settings.model_name,
        model_dim=search_settings.model_dim,
        normalize=search_settings.normalize,
        query_prefix=search_settings.query_prefix,
        passage_prefix=search_settings.passage_prefix,
        status=status,
        index_name=search_settings.index_name,
        provider_type=search_settings.provider_type,
        multipass_indexing=search_settings.multipass_indexing,
        multilingual_expansion=search_settings.multilingual_expansion,
        disable_rerank_for_streaming=search_settings.disable_rerank_for_streaming,
        rerank_model_name=search_settings.rerank_model_name,
        rerank_provider_type=search_settings.rerank_provider_type,
        rerank_api_key=search_settings.rerank_api_key,
        num_rerank=search_settings.num_rerank,
    )

    db_session.add(embedding_model)
    db_session.commit()

    return embedding_model

"""
根据提供商类型获取嵌入提供商
参数：
    db_session: 数据库会话
    provider_type: 提供商类型
返回：
    CloudEmbeddingProvider对象或None
"""
def get_embedding_provider_from_provider_type(
    db_session: Session, provider_type: EmbeddingProvider
) -> CloudEmbeddingProvider | None:
    query = select(CloudEmbeddingProvider).where(
        CloudEmbeddingProvider.provider_type == provider_type
    )
    provider = db_session.execute(query).scalars().first()
    return provider if provider else None

"""
获取当前数据库中的嵌入提供商
参数：
    db_session: 数据库会话
返回：
    ServerCloudEmbeddingProvider对象或None
"""
def get_current_db_embedding_provider(
    db_session: Session,
) -> ServerCloudEmbeddingProvider | None:
    search_settings = get_current_search_settings(db_session=db_session)

    if search_settings.provider_type is None:
        return None

    embedding_provider = fetch_embedding_provider(
        db_session=db_session,
        provider_type=search_settings.provider_type,
    )
    if embedding_provider is None:
        raise RuntimeError("No embedding provider exists for this model.")

    current_embedding_provider = ServerCloudEmbeddingProvider.from_request(
        cloud_provider_model=embedding_provider
    )

    return current_embedding_provider

"""
删除指定的搜索设置
参数：
    db_session: 数据库会话
    search_settings_id: 要删除的搜索设置ID
"""
def delete_search_settings(db_session: Session, search_settings_id: int) -> None:
    current_settings = get_current_search_settings(db_session)

    if current_settings.id == search_settings_id:
        raise ValueError("Cannot delete currently active search settings")

    # First, delete associated index attempts
    index_attempts_query = delete(IndexAttempt).where(
        IndexAttempt.search_settings_id == search_settings_id
    )
    db_session.execute(index_attempts_query)

    # Then, delete the search settings
    search_settings_query = delete(SearchSettings).where(
        and_(
            SearchSettings.id == search_settings_id,
            SearchSettings.status != IndexModelStatus.PRESENT,
        )
    )

    db_session.execute(search_settings_query)
    db_session.commit()

"""
获取当前生效的搜索设置
参数：
    db_session: 数据库会话
返回：
    当前活动的SearchSettings对象
"""
def get_current_search_settings(db_session: Session) -> SearchSettings:
    query = (
        select(SearchSettings)
        .where(SearchSettings.status == IndexModelStatus.PRESENT)
        .order_by(SearchSettings.id.desc())
    )
    result = db_session.execute(query)
    latest_settings = result.scalars().first()

    if not latest_settings:
        raise RuntimeError("No search settings specified, DB is not in a valid state")
    return latest_settings

"""
获取次要搜索设置（处于FUTURE状态的设置）
参数：
    db_session: 数据库会话
返回：
    次要SearchSettings对象或None
"""
def get_secondary_search_settings(db_session: Session) -> SearchSettings | None:
    query = (
        select(SearchSettings)
        .where(SearchSettings.status == IndexModelStatus.FUTURE)
        .order_by(SearchSettings.id.desc())
    )
    result = db_session.execute(query)
    latest_settings = result.scalars().first()

    return latest_settings

"""
获取所有活动的搜索设置
返回的列表中第一个永远是当前搜索设置，如果有正在迁移的新设置会作为第二个条目
Returns active search settings. The first entry will always be the current search 
settings. If there are new search settings that are being migrated to, those will be 
the second entry.
参数：
    db_session: 数据库会话
返回：
    活动搜索设置列表
"""
def get_active_search_settings(db_session: Session) -> list[SearchSettings]:
    """Returns active search settings. The first entry will always be the current search
    settings. If there are new search settings that are being migrated to, those will be
    the second entry."""
    search_settings_list: list[SearchSettings] = []

    # Get the primary search settings
    primary_search_settings = get_current_search_settings(db_session)
    search_settings_list.append(primary_search_settings)

    # Check for secondary search settings
    secondary_search_settings = get_secondary_search_settings(db_session)
    if secondary_search_settings is not None:
        # If secondary settings exist, add them to the list
        search_settings_list.append(secondary_search_settings)

    return search_settings_list

"""
获取所有搜索设置
参数:
    db_session: 数据库会话
返回:
    所有搜索设置的列表
"""
def get_all_search_settings(db_session: Session) -> list[SearchSettings]:
    query = select(SearchSettings).order_by(SearchSettings.id.desc())
    result = db_session.execute(query)
    all_settings = result.scalars().all()
    return list(all_settings)

"""
获取多语言扩展设置
参数:
    db_session: 可选的数据库会话参数
返回:
    多语言扩展列表
"""
def get_multilingual_expansion(db_session: Session | None = None) -> list[str]:
    if db_session is None:
        with get_session_with_default_tenant() as db_session:
            search_settings = get_current_search_settings(db_session)
    else:
        search_settings = get_current_search_settings(db_session)
    if not search_settings:
        return []
    return search_settings.multilingual_expansion

"""
更新搜索设置的具体字段
参数:
    current_settings: 当前的搜索设置
    updated_settings: 更新的搜索设置
    preserved_fields: 需要保留的字段列表
"""
def update_search_settings(
    current_settings: SearchSettings,
    updated_settings: SavedSearchSettings,
    preserved_fields: list[str],
) -> None:
    for field, value in updated_settings.dict().items():
        if field not in preserved_fields:
            setattr(current_settings, field, value)

"""
更新当前活动的搜索设置
参数:
    db_session: 数据库会话
    search_settings: 要更新的搜索设置
    preserved_fields: 需要保留的字段列表，默认使用PRESERVED_SEARCH_FIELDS
"""
def update_current_search_settings(
    db_session: Session,
    search_settings: SavedSearchSettings,
    preserved_fields: list[str] = PRESERVED_SEARCH_FIELDS,
) -> None:
    current_settings = get_current_search_settings(db_session)
    if not current_settings:
        logger.warning("No current search settings found to update")
        return

    # Whenever we update the current search settings, we should ensure that the local reranking model is warmed up.
    if (
        search_settings.rerank_provider_type is None
        and search_settings.rerank_model_name is not None
        and current_settings.rerank_model_name != search_settings.rerank_model_name
    ):
        warm_up_cross_encoder(search_settings.rerank_model_name)

    update_search_settings(current_settings, search_settings, preserved_fields)
    db_session.commit()
    logger.info("Current search settings updated successfully")

"""
更新次要搜索设置
参数:
    db_session: 数据库会话
    search_settings: 要更新的搜索设置
    preserved_fields: 需要保留的字段列表，默认使用PRESERVED_SEARCH_FIELDS
"""
def update_secondary_search_settings(
    db_session: Session,
    search_settings: SavedSearchSettings,
    preserved_fields: list[str] = PRESERVED_SEARCH_FIELDS,
) -> None:
    secondary_settings = get_secondary_search_settings(db_session)
    if not secondary_settings:
        logger.warning("No secondary search settings found to update")
        return

    preserved_fields = PRESERVED_SEARCH_FIELDS
    update_search_settings(secondary_settings, search_settings, preserved_fields)

    db_session.commit()
    logger.info("Secondary search settings updated successfully")

"""
更新搜索设置的状态
参数:
    search_settings: 要更新的搜索设置
    new_status: 新的状态值
    db_session: 数据库会话
"""
def update_search_settings_status(
    search_settings: SearchSettings, new_status: IndexModelStatus, db_session: Session
) -> None:
    search_settings.status = new_status
    db_session.commit()

"""
检查用户是否覆盖了默认的嵌入模型设置
返回:
    布尔值，表示是否覆盖了默认设置
"""
def user_has_overridden_embedding_model() -> bool:
    return DOCUMENT_ENCODER_MODEL != DEFAULT_DOCUMENT_ENCODER_MODEL

"""
获取旧的默认搜索设置
返回:
    包含旧默认配置的SearchSettings对象
"""
def get_old_default_search_settings() -> SearchSettings:
    is_overridden = user_has_overridden_embedding_model()
    return SearchSettings(
        model_name=(
            DOCUMENT_ENCODER_MODEL
            if is_overridden
            else OLD_DEFAULT_DOCUMENT_ENCODER_MODEL
        ),
        model_dim=(
            DOC_EMBEDDING_DIM if is_overridden else OLD_DEFAULT_MODEL_DOC_EMBEDDING_DIM
        ),
        normalize=(
            NORMALIZE_EMBEDDINGS
            if is_overridden
            else OLD_DEFAULT_MODEL_NORMALIZE_EMBEDDINGS
        ),
        query_prefix=(ASYM_QUERY_PREFIX if is_overridden else ""),
        passage_prefix=(ASYM_PASSAGE_PREFIX if is_overridden else ""),
        status=IndexModelStatus.PRESENT,
        index_name="danswer_chunk",
    )

"""
获取新的默认搜索设置
参数:
    is_present: 是否为当前激活状态
返回:
    包含新默认配置的SearchSettings对象
"""
def get_new_default_search_settings(is_present: bool) -> SearchSettings:
    return SearchSettings(
        model_name=DOCUMENT_ENCODER_MODEL,
        model_dim=DOC_EMBEDDING_DIM,
        normalize=NORMALIZE_EMBEDDINGS,
        query_prefix=ASYM_QUERY_PREFIX,
        passage_prefix=ASYM_PASSAGE_PREFIX,
        status=IndexModelStatus.PRESENT if is_present else IndexModelStatus.FUTURE,
        index_name=f"danswer_chunk_{clean_model_name(DOCUMENT_ENCODER_MODEL)}",
    )

"""
获取旧的默认嵌入模型设置
返回:
    包含旧默认嵌入模型配置的IndexingSetting对象
"""
def get_old_default_embedding_model() -> IndexingSetting:
    is_overridden = user_has_overridden_embedding_model()
    return IndexingSetting(
        model_name=(
            DOCUMENT_ENCODER_MODEL
            if is_overridden
            else OLD_DEFAULT_DOCUMENT_ENCODER_MODEL
        ),
        model_dim=(
            DOC_EMBEDDING_DIM if is_overridden else OLD_DEFAULT_MODEL_DOC_EMBEDDING_DIM
        ),
        normalize=(
            NORMALIZE_EMBEDDINGS
            if is_overridden
            else OLD_DEFAULT_MODEL_NORMALIZE_EMBEDDINGS
        ),
        query_prefix=(ASYM_QUERY_PREFIX if is_overridden else ""),
        passage_prefix=(ASYM_PASSAGE_PREFIX if is_overridden else ""),
        index_name="danswer_chunk",
        multipass_indexing=False,
        api_url=None,
    )

"""
获取新的默认嵌入模型设置
返回:
    包含新默认嵌入模型配置的IndexingSetting对象
"""
def get_new_default_embedding_model() -> IndexingSetting:
    return IndexingSetting(
        model_name=DOCUMENT_ENCODER_MODEL,
        model_dim=DOC_EMBEDDING_DIM,
        normalize=NORMALIZE_EMBEDDINGS,
        query_prefix=ASYM_QUERY_PREFIX,
        passage_prefix=ASYM_PASSAGE_PREFIX,
        index_name=f"danswer_chunk_{clean_model_name(DOCUMENT_ENCODER_MODEL)}",
        multipass_indexing=False,
        api_url=None,
    )
