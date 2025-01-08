import sys
import traceback
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any
from typing import cast

import sentry_sdk
import uvicorn
from fastapi import APIRouter
from fastapi import FastAPI
from fastapi import HTTPException
from fastapi import Request
from fastapi import status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from httpx_oauth.clients.google import GoogleOAuth2
from sentry_sdk.integrations.fastapi import FastApiIntegration
from sentry_sdk.integrations.starlette import StarletteIntegration
from sqlalchemy.orm import Session

from onyx import __version__
from onyx.auth.schemas import UserCreate
from onyx.auth.schemas import UserRead
from onyx.auth.schemas import UserUpdate
from onyx.auth.users import auth_backend
from onyx.auth.users import create_onyx_oauth_router
from onyx.auth.users import fastapi_users
from onyx.configs.app_configs import APP_API_PREFIX
from onyx.configs.app_configs import APP_HOST
from onyx.configs.app_configs import APP_PORT
from onyx.configs.app_configs import AUTH_TYPE
from onyx.configs.app_configs import DISABLE_GENERATIVE_AI
from onyx.configs.app_configs import LOG_ENDPOINT_LATENCY
from onyx.configs.app_configs import OAUTH_CLIENT_ID
from onyx.configs.app_configs import OAUTH_CLIENT_SECRET
from onyx.configs.app_configs import POSTGRES_API_SERVER_POOL_OVERFLOW
from onyx.configs.app_configs import POSTGRES_API_SERVER_POOL_SIZE
from onyx.configs.app_configs import SYSTEM_RECURSION_LIMIT
from onyx.configs.app_configs import USER_AUTH_SECRET
from onyx.configs.app_configs import WEB_DOMAIN
from onyx.configs.constants import AuthType
from onyx.configs.constants import POSTGRES_WEB_APP_NAME
from onyx.db.engine import SqlEngine
from onyx.db.engine import warm_up_connections
from onyx.server.api_key.api import router as api_key_router
from onyx.server.auth_check import check_router_auth
from onyx.server.documents.cc_pair import router as cc_pair_router
from onyx.server.documents.connector import router as connector_router
from onyx.server.documents.credential import router as credential_router
from onyx.server.documents.document import router as document_router
from onyx.server.documents.indexing import router as indexing_router
from onyx.server.documents.standard_oauth import router as oauth_router
from onyx.server.features.document_set.api import router as document_set_router
from onyx.server.features.folder.api import router as folder_router
from onyx.server.features.notifications.api import router as notification_router
from onyx.server.features.persona.api import admin_router as admin_persona_router
from onyx.server.features.persona.api import basic_router as persona_router
from onyx.server.features.prompt.api import basic_router as prompt_router
from onyx.server.features.tool.api import admin_router as admin_tool_router
from onyx.server.features.tool.api import router as tool_router
from onyx.server.gpts.api import router as gpts_router
from onyx.server.long_term_logs.long_term_logs_api import (
    router as long_term_logs_router,
)
from onyx.server.manage.administrative import router as admin_router
from onyx.server.manage.embedding.api import admin_router as embedding_admin_router
from onyx.server.manage.embedding.api import basic_router as embedding_router
from onyx.server.manage.get_state import router as state_router
from onyx.server.manage.llm.api import admin_router as llm_admin_router
from onyx.server.manage.llm.api import basic_router as llm_router
from onyx.server.manage.search_settings import router as search_settings_router
from onyx.server.manage.slack_bot import router as slack_bot_management_router
from onyx.server.manage.users import router as user_router
from onyx.server.middleware.latency_logging import add_latency_logging_middleware
from onyx.server.middleware.rate_limiting import close_limiter
from onyx.server.middleware.rate_limiting import get_auth_rate_limiters
from onyx.server.middleware.rate_limiting import setup_limiter
from onyx.server.onyx_api.ingestion import router as onyx_api_router
from onyx.server.openai_assistants_api.full_openai_assistants_api import (
    get_full_openai_assistants_api_router,
)
from onyx.server.query_and_chat.chat_backend import router as chat_router
from onyx.server.query_and_chat.query_backend import (
    admin_router as admin_query_router,
)
from onyx.server.query_and_chat.query_backend import basic_router as query_router
from onyx.server.settings.api import admin_router as settings_admin_router
from onyx.server.settings.api import basic_router as settings_router
from onyx.server.token_rate_limits.api import (
    router as token_rate_limit_settings_router,
)
from onyx.server.utils import BasicAuthenticationError
from onyx.setup import setup_multitenant_onyx
from onyx.setup import setup_onyx
from onyx.utils.logger import setup_logger
from onyx.utils.telemetry import get_or_generate_uuid
from onyx.utils.telemetry import optional_telemetry
from onyx.utils.telemetry import RecordType
from onyx.utils.variable_functionality import fetch_versioned_implementation
from onyx.utils.variable_functionality import global_version
from onyx.utils.variable_functionality import set_is_ee_based_on_env_variable
from shared_configs.configs import CORS_ALLOWED_ORIGIN
from shared_configs.configs import MULTI_TENANT
from shared_configs.configs import SENTRY_DSN

logger = setup_logger()


def validation_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    if not isinstance(exc, RequestValidationError):
        logger.error(
            f"Unexpected exception type in validation_exception_handler - {type(exc)}"
        )
        raise exc

    exc_str = f"{exc}".replace("\n", " ").replace("   ", " ")
    logger.exception(f"{request}: {exc_str}")
    content = {"status_code": 422, "message": exc_str, "data": None}
    return JSONResponse(content=content, status_code=422)


def value_error_handler(_: Request, exc: Exception) -> JSONResponse:
    if not isinstance(exc, ValueError):
        logger.error(f"Unexpected exception type in value_error_handler - {type(exc)}")
        raise exc

    try:
        raise (exc)
    except Exception:
        # log stacktrace
        logger.exception("ValueError")
    return JSONResponse(
        status_code=400,
        content={"message": str(exc)},
    )


def include_router_with_global_prefix_prepended(
    application: FastAPI, router: APIRouter, **kwargs: Any
) -> None:
    """Adds the global prefix to all routes in the router."""
    # 为路由中的所有路径添加全局前缀
    processed_global_prefix = f"/{APP_API_PREFIX.strip('/')}" if APP_API_PREFIX else ""

    passed_in_prefix = cast(str | None, kwargs.get("prefix"))
    if passed_in_prefix:
        final_prefix = f"{processed_global_prefix}/{passed_in_prefix.strip('/')}"
    else:
        final_prefix = f"{processed_global_prefix}"
    final_kwargs: dict[str, Any] = {
        **kwargs,
        "prefix": final_prefix,
    }

    application.include_router(router, **final_kwargs)


def include_auth_router_with_prefix(
    application: FastAPI, router: APIRouter, prefix: str, tags: list[str] | None = None
) -> None:
    """Wrapper function to include an 'auth' router with prefix + rate-limiting dependencies."""
    # 包装函数，用于包含带有前缀和速率限制依赖项的“auth”路由
    final_tags = tags or ["auth"]
    include_router_with_global_prefix_prepended(
        application,
        router,
        prefix=prefix,
        tags=final_tags,
        dependencies=get_auth_rate_limiters(),
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    # Set recursion limit
    # 设置递归限制
    if SYSTEM_RECURSION_LIMIT is not None:
        sys.setrecursionlimit(SYSTEM_RECURSION_LIMIT)
        logger.notice(f"System recursion limit set to {SYSTEM_RECURSION_LIMIT}")

    SqlEngine.set_app_name(POSTGRES_WEB_APP_NAME)
    SqlEngine.init_engine(
        pool_size=POSTGRES_API_SERVER_POOL_SIZE,
        max_overflow=POSTGRES_API_SERVER_POOL_OVERFLOW,
    )
    engine = SqlEngine.get_engine()

    verify_auth = fetch_versioned_implementation(
        "onyx.auth.users", "verify_auth_setting"
    )

    # Will throw exception if an issue is found
    # 如果发现问题，将抛出异常
    verify_auth()

    if OAUTH_CLIENT_ID and OAUTH_CLIENT_SECRET:
        logger.notice("Both OAuth Client ID and Secret are configured.")
        # OAuth客户端ID和密钥均已配置

    if DISABLE_GENERATIVE_AI:
        logger.notice("Generative AI Q&A disabled")
        # 生成式AI问答已禁用

    # fill up Postgres connection pools
    # 填充Postgres连接池
    await warm_up_connections()

    if not MULTI_TENANT:
        # We cache this at the beginning so there is no delay in the first telemetry
        # 我们在开始时缓存这个，以便第一次遥测不会有延迟
        get_or_generate_uuid()

        # If we are multi-tenant, we need to only set up initial public tables
        # 如果我们是多租户，我��只需要设置初始的公共表
        with Session(engine) as db_session:
            setup_onyx(db_session, None)
    else:
        setup_multitenant_onyx()

    optional_telemetry(record_type=RecordType.VERSION, data={"version": __version__})

    # Set up rate limiter
    # 设置速率限制器
    await setup_limiter()

    yield

    # Close rate limiter
    # 关闭速率限制器
    await close_limiter()


def log_http_error(_: Request, exc: Exception) -> JSONResponse:
    status_code = getattr(exc, "status_code", 500)

    if isinstance(exc, BasicAuthenticationError):
        # For BasicAuthenticationError, just log a brief message without stack trace (almost always spam)
        # 对于BasicAuthenticationError，只记录简短消息而不记录堆栈跟踪（几乎总是垃圾邮件）
        logger.warning(f"Authentication failed: {str(exc)}")

    elif status_code >= 400:
        error_msg = f"{str(exc)}\n"
        error_msg += "".join(traceback.format_tb(exc.__traceback__))
        logger.error(error_msg)

    detail = exc.detail if isinstance(exc, HTTPException) else str(exc)
    return JSONResponse(
        status_code=status_code,
        content={"detail": detail},
    )


def get_application() -> FastAPI:
    application = FastAPI(title="Onyx Backend", version=__version__, lifespan=lifespan)
    if SENTRY_DSN:
        sentry_sdk.init(
            dsn=SENTRY_DSN,
            integrations=[StarletteIntegration(), FastApiIntegration()],
            traces_sample_rate=0.1,
        )
        logger.info("Sentry initialized")
        # Sentry已初始化
    else:
        logger.debug("Sentry DSN not provided, skipping Sentry initialization")
        # 未提供Sentry DSN，跳过Sentry初始化

    application.add_exception_handler(status.HTTP_400_BAD_REQUEST, log_http_error)
    application.add_exception_handler(status.HTTP_401_UNAUTHORIZED, log_http_error)
    application.add_exception_handler(status.HTTP_403_FORBIDDEN, log_http_error)
    application.add_exception_handler(status.HTTP_404_NOT_FOUND, log_http_error)
    application.add_exception_handler(
        status.HTTP_500_INTERNAL_SERVER_ERROR, log_http_error
    )

    include_router_with_global_prefix_prepended(application, chat_router)
    include_router_with_global_prefix_prepended(application, query_router)
    include_router_with_global_prefix_prepended(application, document_router)
    include_router_with_global_prefix_prepended(application, user_router)
    include_router_with_global_prefix_prepended(application, admin_query_router)
    include_router_with_global_prefix_prepended(application, admin_router)
    include_router_with_global_prefix_prepended(application, connector_router)
    include_router_with_global_prefix_prepended(application, user_router)
    include_router_with_global_prefix_prepended(application, credential_router)
    include_router_with_global_prefix_prepended(application, cc_pair_router)
    include_router_with_global_prefix_prepended(application, folder_router)
    include_router_with_global_prefix_prepended(application, document_set_router)
    include_router_with_global_prefix_prepended(application, search_settings_router)
    include_router_with_global_prefix_prepended(
        application, slack_bot_management_router
    )
    include_router_with_global_prefix_prepended(application, persona_router)
    include_router_with_global_prefix_prepended(application, admin_persona_router)
    include_router_with_global_prefix_prepended(application, notification_router)
    include_router_with_global_prefix_prepended(application, prompt_router)
    include_router_with_global_prefix_prepended(application, tool_router)
    include_router_with_global_prefix_prepended(application, admin_tool_router)
    include_router_with_global_prefix_prepended(application, state_router)
    include_router_with_global_prefix_prepended(application, onyx_api_router)
    include_router_with_global_prefix_prepended(application, gpts_router)
    include_router_with_global_prefix_prepended(application, settings_router)
    include_router_with_global_prefix_prepended(application, settings_admin_router)
    include_router_with_global_prefix_prepended(application, llm_admin_router)
    include_router_with_global_prefix_prepended(application, llm_router)
    include_router_with_global_prefix_prepended(application, embedding_admin_router)
    include_router_with_global_prefix_prepended(application, embedding_router)
    include_router_with_global_prefix_prepended(
        application, token_rate_limit_settings_router
    )
    include_router_with_global_prefix_prepended(application, indexing_router)
    include_router_with_global_prefix_prepended(
        application, get_full_openai_assistants_api_router()
    )
    include_router_with_global_prefix_prepended(application, long_term_logs_router)
    include_router_with_global_prefix_prepended(application, api_key_router)
    include_router_with_global_prefix_prepended(application, oauth_router)

    if AUTH_TYPE == AuthType.DISABLED:
        # Server logs this during auth setup verification step
        # 服务器在身份验证设置验证步骤中记录此信息
        pass

    if AUTH_TYPE == AuthType.BASIC or AUTH_TYPE == AuthType.CLOUD:
        include_auth_router_with_prefix(
            application,
            fastapi_users.get_auth_router(auth_backend),
            prefix="/auth",
        )

        include_auth_router_with_prefix(
            application,
            fastapi_users.get_register_router(UserRead, UserCreate),
            prefix="/auth",
        )

        include_auth_router_with_prefix(
            application,
            fastapi_users.get_reset_password_router(),
            prefix="/auth",
        )
        include_auth_router_with_prefix(
            application,
            fastapi_users.get_verify_router(UserRead),
            prefix="/auth",
        )
        include_auth_router_with_prefix(
            application,
            fastapi_users.get_users_router(UserRead, UserUpdate),
            prefix="/users",
        )

    if AUTH_TYPE == AuthType.GOOGLE_OAUTH:
        oauth_client = GoogleOAuth2(OAUTH_CLIENT_ID, OAUTH_CLIENT_SECRET)
        include_auth_router_with_prefix(
            application,
            create_onyx_oauth_router(
                oauth_client,
                auth_backend,
                USER_AUTH_SECRET,
                associate_by_email=True,
                is_verified_by_default=True,
                # Points the user back to the login page
                # 将用户指向登录页面
                redirect_url=f"{WEB_DOMAIN}/auth/oauth/callback",
            ),
            prefix="/auth/oauth",
        )

        # Need basic auth router for `logout` endpoint
        # 需要基本身份验证路由器用于`logout`端点
        include_auth_router_with_prefix(
            application,
            fastapi_users.get_logout_router(auth_backend),
            prefix="/auth",
        )

    application.add_exception_handler(
        RequestValidationError, validation_exception_handler
    )

    application.add_exception_handler(ValueError, value_error_handler)

    application.add_middleware(
        CORSMiddleware,
        allow_origins=CORS_ALLOWED_ORIGIN,  # Configurable via environment variable
        # 可通过环境变量配置
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    if LOG_ENDPOINT_LATENCY:
        add_latency_logging_middleware(application, logger)

    # Ensure all routes have auth enabled or are explicitly marked as public
    # 确保所有路由都启用了身份验证或明确标记为公共
    check_router_auth(application)

    return application


# NOTE: needs to be outside of the `if __name__ == "__main__"` block so that the
# app is exportable
# 注意：需要在`if __name__ == "__main__"`块之外，以便应用程序可以导出
set_is_ee_based_on_env_variable()
app = fetch_versioned_implementation(module="onyx.main", attribute="get_application")


if __name__ == "__main__":
    logger.notice(
        f"Starting Onyx Backend version {__version__} on http://{APP_HOST}:{str(APP_PORT)}/"
    )

    if global_version.is_ee_version():
        logger.notice("Running Enterprise Edition")
        # 运行企业版

    uvicorn.run(app, host=APP_HOST, port=APP_PORT)
