import time

from sqlalchemy.orm import Session

from onyx.configs.app_configs import DISABLE_INDEX_UPDATE_ON_SWAP
from onyx.configs.app_configs import MANAGED_VESPA
from onyx.configs.app_configs import VESPA_NUM_ATTEMPTS_ON_STARTUP
from onyx.configs.constants import KV_REINDEX_KEY
from onyx.configs.constants import KV_SEARCH_SETTINGS
from onyx.configs.model_configs import FAST_GEN_AI_MODEL_VERSION
from onyx.configs.model_configs import GEN_AI_API_KEY
from onyx.configs.model_configs import GEN_AI_MODEL_VERSION
from onyx.context.search.models import SavedSearchSettings
from onyx.context.search.retrieval.search_runner import download_nltk_data
from onyx.db.connector import check_connectors_exist
from onyx.db.connector import create_initial_default_connector
from onyx.db.connector_credential_pair import associate_default_cc_pair
from onyx.db.connector_credential_pair import get_connector_credential_pairs
from onyx.db.connector_credential_pair import resync_cc_pair
from onyx.db.credentials import create_initial_public_credential
from onyx.db.document import check_docs_exist
from onyx.db.index_attempt import cancel_indexing_attempts_past_model
from onyx.db.index_attempt import expire_index_attempts
from onyx.db.llm import fetch_default_provider
from onyx.db.llm import update_default_provider
from onyx.db.llm import upsert_llm_provider
from onyx.db.persona import delete_old_default_personas
from onyx.db.search_settings import get_current_search_settings
from onyx.db.search_settings import get_secondary_search_settings
from onyx.db.search_settings import update_current_search_settings
from onyx.db.search_settings import update_secondary_search_settings
from onyx.db.swap_index import check_index_swap
from onyx.document_index.factory import get_default_document_index
from onyx.document_index.interfaces import DocumentIndex
from onyx.document_index.vespa.index import VespaIndex
from onyx.indexing.models import IndexingSetting
from onyx.key_value_store.factory import get_kv_store
from onyx.key_value_store.interface import KvKeyNotFoundError
from onyx.natural_language_processing.search_nlp_models import EmbeddingModel
from onyx.natural_language_processing.search_nlp_models import warm_up_bi_encoder
from onyx.natural_language_processing.search_nlp_models import warm_up_cross_encoder
from onyx.seeding.load_docs import seed_initial_documents
from onyx.seeding.load_yamls import load_chat_yamls
from onyx.server.manage.llm.models import LLMProviderUpsertRequest
from onyx.server.settings.store import load_settings
from onyx.server.settings.store import store_settings
from onyx.tools.built_in_tools import auto_add_search_tool_to_personas
from onyx.tools.built_in_tools import load_builtin_tools
from onyx.tools.built_in_tools import refresh_built_in_tools_cache
from onyx.utils.gpu_utils import gpu_status_request
from onyx.utils.logger import setup_logger
from shared_configs.configs import ALT_INDEX_SUFFIX
from shared_configs.configs import MODEL_SERVER_HOST
from shared_configs.configs import MODEL_SERVER_PORT
from shared_configs.configs import MULTI_TENANT
from shared_configs.configs import SUPPORTED_EMBEDDING_MODELS
from shared_configs.model_server_models import SupportedEmbeddingModel


logger = setup_logger()


def setup_onyx(
    db_session: Session, tenant_id: str | None, cohere_enabled: bool = False
) -> None:
    """
    Setup Onyx for a particular tenant. In the Single Tenant case, it will set it up for the default schema
    on server startup. In the MT case, it will be called when the tenant is created.

    The Tenant Service calls the tenants/create endpoint which runs this.
    """
    # 为特定租户设置Onyx。在单租户情况下，它将在服务器启动时为默认模式设置。在多租户情况下，它将在创建租户时调用。
    # 租户服务调用tenants/create端点来运行此操作。
    check_index_swap(db_session=db_session)
    search_settings = get_current_search_settings(db_session)
    secondary_search_settings = get_secondary_search_settings(db_session)

    # Break bad state for thrashing indexes
    # 解决索引抖动的错误状态
    if secondary_search_settings and DISABLE_INDEX_UPDATE_ON_SWAP:
        expire_index_attempts(
            search_settings_id=search_settings.id, db_session=db_session
        )

        for cc_pair in get_connector_credential_pairs(db_session):
            resync_cc_pair(cc_pair, db_session=db_session)

    # Expire all old embedding models indexing attempts, technically redundant
    # 过期所有旧的嵌入模型索引尝试，技术上是多余的
    cancel_indexing_attempts_past_model(db_session)

    logger.notice(f'Using Embedding model: "{search_settings.model_name}"')
    # 使用嵌入模型
    if search_settings.query_prefix or search_settings.passage_prefix:
        logger.notice(f'Query embedding prefix: "{search_settings.query_prefix}"')
        # 查询嵌入前缀
        logger.notice(f'Passage embedding prefix: "{search_settings.passage_prefix}"')
        # 段落嵌入前缀

    if search_settings:
        if not search_settings.disable_rerank_for_streaming:
            logger.notice("Reranking is enabled.")
            # 重新排序已启用

        if search_settings.multilingual_expansion:
            logger.notice(
                f"Multilingual query expansion is enabled with {search_settings.multilingual_expansion}."
            )
            # 启用了多语言查询扩展

    if (
        search_settings.rerank_model_name
        and not search_settings.provider_type
        and not search_settings.rerank_provider_type
    ):
        warm_up_cross_encoder(search_settings.rerank_model_name)

    logger.notice("Verifying query preprocessing (NLTK) data is downloaded")
    # 验证查询预处理（NLTK）数据是否已下载
    download_nltk_data()

    # setup Postgres with default credential, llm providers, etc.
    # 使用默认凭证、llm提供程序等设置Postgres
    setup_postgres(db_session)

    translate_saved_search_settings(db_session)

    # Does the user need to trigger a reindexing to bring the document index
    # into a good state, marked in the kv store
    # 用户是否需要触发重新索引以使文档索引处于良好状态，在kv存储中标记
    if not MULTI_TENANT:
        mark_reindex_flag(db_session)

    # Ensure Vespa is setup correctly, this step is relatively near the end because Vespa
    # takes a bit of time to start up
    # 确保Vespa设置正确，这一步相对接近尾声，因为Vespa启动需要一些时间
    logger.notice("Verifying Document Index(s) is/are available.")
    document_index = get_default_document_index(
        primary_index_name=search_settings.index_name,
        secondary_index_name=secondary_search_settings.index_name
        if secondary_search_settings
        else None,
    )

    success = setup_vespa(
        document_index,
        IndexingSetting.from_db_model(search_settings),
        IndexingSetting.from_db_model(secondary_search_settings)
        if secondary_search_settings
        else None,
    )
    if not success:
        raise RuntimeError("Could not connect to Vespa within the specified timeout.")
        # 无法在指定的超时时间内连接到Vespa

    logger.notice(f"Model Server: http://{MODEL_SERVER_HOST}:{MODEL_SERVER_PORT}")
    # 模型服务器
    if search_settings.provider_type is None:
        warm_up_bi_encoder(
            embedding_model=EmbeddingModel.from_db_model(
                search_settings=search_settings,
                server_host=MODEL_SERVER_HOST,
                server_port=MODEL_SERVER_PORT,
            ),
        )

    # update multipass indexing setting based on GPU availability
    # 根据GPU可用性更新多通道索引设置
    update_default_multipass_indexing(db_session)

    seed_initial_documents(db_session, tenant_id, cohere_enabled)


def translate_saved_search_settings(db_session: Session) -> None:
    kv_store = get_kv_store()

    try:
        search_settings_dict = kv_store.load(KV_SEARCH_SETTINGS)
        if isinstance(search_settings_dict, dict):
            # Update current search settings
            # 更新当前搜索设置
            current_settings = get_current_search_settings(db_session)

            # Update non-preserved fields
            # 更新非保留字段
            if current_settings:
                current_settings_dict = SavedSearchSettings.from_db_model(
                    current_settings
                ).dict()

                new_current_settings = SavedSearchSettings(
                    **{**current_settings_dict, **search_settings_dict}
                )
                update_current_search_settings(db_session, new_current_settings)

            # Update secondary search settings
            # 更新次要搜索设置
            secondary_settings = get_secondary_search_settings(db_session)
            if secondary_settings:
                secondary_settings_dict = SavedSearchSettings.from_db_model(
                    secondary_settings
                ).dict()

                new_secondary_settings = SavedSearchSettings(
                    **{**secondary_settings_dict, **search_settings_dict}
                )
                update_secondary_search_settings(
                    db_session,
                    new_secondary_settings,
                )
            # Delete the KV store entry after successful update
            # 成功更新后删除KV存储条目
            kv_store.delete(KV_SEARCH_SETTINGS)
            logger.notice("Search settings updated and KV store entry deleted.")
            # 搜索设置已更新，KV存储条目已删除
        else:
            logger.notice("KV store search settings is empty.")
            # KV存储搜索设置为空
    except KvKeyNotFoundError:
        logger.notice("No search config found in KV store.")
        # 在KV存储中未找到搜索配置


def mark_reindex_flag(db_session: Session) -> None:
    kv_store = get_kv_store()
    try:
        value = kv_store.load(KV_REINDEX_KEY)
        logger.debug(f"Re-indexing flag has value {value}")
        # 重新索引标志的值
        return
    except KvKeyNotFoundError:
        # Only need to update the flag if it hasn't been set
        # 仅在未设置标志时才需要更新
        pass

    # If their first deployment is after the changes, it will
    # enable this when the other changes go in, need to avoid
    # this being set to False, then the user indexes things on the old version
    # 如果他们的首次部署在更改之后，它将在其他更改生效时启用此功能，需要避免将其设置为False，然后用户在旧版本上索引内容
    docs_exist = check_docs_exist(db_session)
    connectors_exist = check_connectors_exist(db_session)
    if docs_exist or connectors_exist:
        kv_store.store(KV_REINDEX_KEY, True)
    else:
        kv_store.store(KV_REINDEX_KEY, False)


def setup_vespa(
    document_index: DocumentIndex,
    index_setting: IndexingSetting,
    secondary_index_setting: IndexingSetting | None,
    num_attempts: int = VESPA_NUM_ATTEMPTS_ON_STARTUP,
) -> bool:
    # Vespa startup is a bit slow, so give it a few seconds
    # Vespa启动有点慢，所以给它几秒钟
    WAIT_SECONDS = 5
    for x in range(num_attempts):
        try:
            logger.notice(f"Setting up Vespa (attempt {x+1}/{num_attempts})...")
            # 设置Vespa（尝试次数）
            document_index.ensure_indices_exist(
                index_embedding_dim=index_setting.model_dim,
                secondary_index_embedding_dim=secondary_index_setting.model_dim
                if secondary_index_setting
                else None,
            )

            logger.notice("Vespa setup complete.")
            # Vespa设置完成
            return True
        except Exception:
            logger.notice(
                f"Vespa setup did not succeed. The Vespa service may not be ready yet. Retrying in {WAIT_SECONDS} seconds."
            )
            # Vespa设置未成功。Vespa服务可能尚未准备好。将在几秒钟后重试。
            time.sleep(WAIT_SECONDS)

    logger.error(
        f"Vespa setup did not succeed. Attempt limit reached. ({num_attempts})"
    )
    # Vespa设置未成功。已达到尝试限制。
    return False


def setup_postgres(db_session: Session) -> None:
    logger.notice("Verifying default connector/credential exist.")
    # 验证默认连接器/凭证是否存在
    create_initial_public_credential(db_session)
    create_initial_default_connector(db_session)
    associate_default_cc_pair(db_session)
    delete_old_default_personas(db_session)

    logger.notice("Loading built-in tools")
    # 加载内置工具
    load_builtin_tools(db_session)

    logger.notice("Loading default Prompts and Personas")
    # 加载默认提示和角色
    load_chat_yamls(db_session)

    refresh_built_in_tools_cache(db_session)
    auto_add_search_tool_to_personas(db_session)

    if GEN_AI_API_KEY and fetch_default_provider(db_session) is None:
        # Only for dev flows
        # 仅用于开发流程
        logger.notice("Setting up default OpenAI LLM for dev.")
        # 为开发设置默认的OpenAI LLM
        llm_model = GEN_AI_MODEL_VERSION or "gpt-4o-mini"
        fast_model = FAST_GEN_AI_MODEL_VERSION or "gpt-4o-mini"
        model_req = LLMProviderUpsertRequest(
            name="DevEnvPresetOpenAI",
            provider="openai",
            api_key=GEN_AI_API_KEY,
            api_base=None,
            api_version=None,
            custom_config=None,
            default_model_name=llm_model,
            fast_default_model_name=fast_model,
            is_public=True,
            groups=[],
            display_model_names=[llm_model, fast_model],
            model_names=[llm_model, fast_model],
        )
        new_llm_provider = upsert_llm_provider(
            llm_provider=model_req, db_session=db_session
        )
        update_default_provider(provider_id=new_llm_provider.id, db_session=db_session)


def update_default_multipass_indexing(db_session: Session) -> None:
    docs_exist = check_docs_exist(db_session)
    connectors_exist = check_connectors_exist(db_session)
    logger.debug(f"Docs exist: {docs_exist}, Connectors exist: {connectors_exist}")
    # 文档存在，连接器存在

    if not docs_exist and not connectors_exist:
        logger.info(
            "No existing docs or connectors found. Checking GPU availability for multipass indexing."
        )
        # 未找到现有文档或连接器。检查多通道索引的GPU可用性。
        gpu_available = gpu_status_request()
        logger.info(f"GPU available: {gpu_available}")
        # GPU可用

        current_settings = get_current_search_settings(db_session)

        logger.notice(f"Updating multipass indexing setting to: {gpu_available}")
        # 将多通道索引设置更新为
        updated_settings = SavedSearchSettings.from_db_model(current_settings)
        # Enable multipass indexing if GPU is available or if using a cloud provider
        # 如果GPU可用或使用云提供商，则启用多通道索引
        updated_settings.multipass_indexing = (
            gpu_available or current_settings.cloud_provider is not None
        )
        update_current_search_settings(db_session, updated_settings)

        # Update settings with GPU availability
        # 使用GPU可用性更新设置
        settings = load_settings()
        settings.gpu_enabled = gpu_available
        store_settings(settings)
        logger.notice(f"Updated settings with GPU availability: {gpu_available}")
        # 使用GPU可用性更新的设置

    else:
        logger.debug(
            "Existing docs or connectors found. Skipping multipass indexing update."
        )
        # 找到现有文档或连接器。跳过多通道索引更新。


def setup_multitenant_onyx() -> None:
    # For Managed Vespa, the schema is sent over via the Vespa Console manually.
    # 对于托管Vespa，架构通过Vespa控制台手动发送。
    if not MANAGED_VESPA:
        setup_vespa_multitenant(SUPPORTED_EMBEDDING_MODELS)


def setup_vespa_multitenant(supported_indices: list[SupportedEmbeddingModel]) -> bool:
    # This is for local testing
    # 这是用于本地测试
    WAIT_SECONDS = 5
    VESPA_ATTEMPTS = 5
    for x in range(VESPA_ATTEMPTS):
        try:
            logger.notice(f"Setting up Vespa (attempt {x+1}/{VESPA_ATTEMPTS})...")
            # 设置Vespa（尝试次数）
            VespaIndex.register_multitenant_indices(
                indices=[index.index_name for index in supported_indices]
                + [
                    f"{index.index_name}{ALT_INDEX_SUFFIX}"
                    for index in supported_indices
                ],
                embedding_dims=[index.dim for index in supported_indices]
                + [index.dim for index in supported_indices],
            )

            logger.notice("Vespa setup complete.")
            # Vespa设置完成
            return True
        except Exception:
            logger.notice(
                f"Vespa setup did not succeed. The Vespa service may not be ready yet. Retrying in {WAIT_SECONDS} seconds."
            )
            # Vespa设置未成功。Vespa服务可能尚未准备好。将在几秒钟后重试。
            time.sleep(WAIT_SECONDS)

    logger.error(
        f"Vespa setup did not succeed. Attempt limit reached. ({VESPA_ATTEMPTS})"
    )
    # Vespa设置未成功。已达到尝试限制。
    return False
