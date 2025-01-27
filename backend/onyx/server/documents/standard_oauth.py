import json
import uuid
from typing import Annotated
from typing import cast

from fastapi import APIRouter
from fastapi import Depends
from fastapi import HTTPException
from fastapi import Query
from fastapi import Request
from pydantic import BaseModel
from pydantic import ValidationError
from sqlalchemy.orm import Session

from onyx.auth.users import current_user
from onyx.configs.app_configs import WEB_DOMAIN
from onyx.configs.constants import DocumentSource
from onyx.connectors.interfaces import OAuthConnector
from onyx.db.credentials import create_credential
from onyx.db.engine import get_current_tenant_id
from onyx.db.engine import get_session
from onyx.db.models import User
from onyx.redis.redis_pool import get_redis_client
from onyx.server.documents.models import CredentialBase
from onyx.utils.logger import setup_logger
from onyx.utils.subclasses import find_all_subclasses_in_dir

logger = setup_logger()

router = APIRouter(prefix="/connector/oauth")

_OAUTH_STATE_KEY_FMT = "oauth_state:{state}"
_OAUTH_STATE_EXPIRATION_SECONDS = 10 * 60  # 10 minutes
_DESIRED_RETURN_URL_KEY = "desired_return_url"
_ADDITIONAL_KWARGS_KEY = "additional_kwargs"

# Cache for OAuth connectors, populated at module load time
_OAUTH_CONNECTORS: dict[DocumentSource, type[OAuthConnector]] = {}

"""
This module implements the standard OAuth2.0 flow for document source authentication.
此模块实现了文档源认证的标准OAuth2.0流程。

主要功能：
1. OAuth授权流程的初始化
2. OAuth回调处理
3. OAuth凭证管理
4. OAuth连接器的动态发现
"""

def _discover_oauth_connectors() -> dict[DocumentSource, type[OAuthConnector]]:
    """Walk through the connectors package to find all OAuthConnector implementations
    遍历connectors包以查找所有OAuthConnector的实现
    
    返回值：
        dict[DocumentSource, type[OAuthConnector]]: 包含所有OAuth连接器的字典，
        键为文档源，值为对应的连接器类
    """
    global _OAUTH_CONNECTORS
    if _OAUTH_CONNECTORS:  # Return cached connectors if already discovered
        return _OAUTH_CONNECTORS

    oauth_connectors = find_all_subclasses_in_dir(
        cast(type[OAuthConnector], OAuthConnector), "onyx.connectors"
    )

    _OAUTH_CONNECTORS = {cls.oauth_id(): cls for cls in oauth_connectors}
    return _OAUTH_CONNECTORS


# Discover OAuth connectors at module load time
_discover_oauth_connectors()


def _get_additional_kwargs(
    request: Request, connector_cls: type[OAuthConnector], args_to_ignore: list[str]
) -> dict[str, str]:
    """
    从请求中获取额外的关键字参数
    
    参数：
        request: FastAPI请求对象
        connector_cls: OAuth连接器类
        args_to_ignore: 需要忽略的参数列表
        
    返回值：
        dict[str, str]: 验证后的额外参数字典
        
    异常：
        HTTPException: 当参数验证失败时抛出
    """
    additional_kwargs_dict = {
        k: v for k, v in request.query_params.items() if k not in args_to_ignore
    }
    try:
        # validate
        connector_cls.AdditionalOauthKwargs(**additional_kwargs_dict)
    except ValidationError:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid additional kwargs. Got {additional_kwargs_dict}, expected "
                f"{connector_cls.AdditionalOauthKwargs.model_json_schema()}"
            ),
        )

    return additional_kwargs_dict


class AuthorizeResponse(BaseModel):
    """
    OAuth授权响应模型
    
    属性：
        redirect_url: str 重定向URL
    """
    redirect_url: str


@router.get("/authorize/{source}")
def oauth_authorize(
    request: Request,
    source: DocumentSource,
    desired_return_url: Annotated[str | None, Query()] = None,
    _: User = Depends(current_user),
    tenant_id: str | None = Depends(get_current_tenant_id),
) -> AuthorizeResponse:
    """Initiates the OAuth flow by redirecting to the provider's auth page
    通过重定向到提供商的认证页面来初始化OAuth流程
    
    参数：
        request: FastAPI请求对象
        source: 文档源
        desired_return_url: 期望的返回URL
        _: 当前用户（用于认证）
        tenant_id: 租户ID
        
    返回值：
        AuthorizeResponse: 包含重定向URL的响应对象
    """
    oauth_connectors = _discover_oauth_connectors()

    if source not in oauth_connectors:
        raise HTTPException(status_code=400, detail=f"Unknown OAuth source: {source}")

    connector_cls = oauth_connectors[source]
    base_url = WEB_DOMAIN

    # get additional kwargs from request
    # e.g. anything except for desired_return_url
    additional_kwargs = _get_additional_kwargs(
        request, connector_cls, ["desired_return_url"]
    )

    # store state in redis
    if not desired_return_url:
        desired_return_url = f"{base_url}/admin/connectors/{source}?step=0"
    redis_client = get_redis_client(tenant_id=tenant_id)
    state = str(uuid.uuid4())
    redis_client.set(
        _OAUTH_STATE_KEY_FMT.format(state=state),
        json.dumps(
            {
                _DESIRED_RETURN_URL_KEY: desired_return_url,
                _ADDITIONAL_KWARGS_KEY: additional_kwargs,
            }
        ),
        ex=_OAUTH_STATE_EXPIRATION_SECONDS,
    )

    return AuthorizeResponse(
        redirect_url=connector_cls.oauth_authorization_url(
            base_url, state, additional_kwargs
        )
    )


class CallbackResponse(BaseModel):
    """
    OAuth回调响应模型
    
    属性：
        redirect_url: str 重定向URL
    """
    redirect_url: str


@router.get("/callback/{source}")
def oauth_callback(
    source: DocumentSource,
    code: Annotated[str, Query()],
    state: Annotated[str, Query()],
    db_session: Session = Depends(get_session),
    user: User = Depends(current_user),
    tenant_id: str | None = Depends(get_current_tenant_id),
) -> CallbackResponse:
    """Handles the OAuth callback and exchanges the code for tokens
    处理OAuth回调并用代码交换令牌
    
    参数：
        source: 文档源
        code: OAuth授权码
        state: OAuth状态值
        db_session: 数据库会话
        user: 当前用户
        tenant_id: 租户ID
        
    返回值：
        CallbackResponse: 包含重定向URL的响应对象
    """
    oauth_connectors = _discover_oauth_connectors()

    if source not in oauth_connectors:
        raise HTTPException(status_code=400, detail=f"Unknown OAuth source: {source}")

    connector_cls = oauth_connectors[source]

    # get state from redis
    redis_client = get_redis_client(tenant_id=tenant_id)
    oauth_state_bytes = cast(
        bytes, redis_client.get(_OAUTH_STATE_KEY_FMT.format(state=state))
    )
    if not oauth_state_bytes:
        raise HTTPException(status_code=400, detail="Invalid OAuth state")
    oauth_state = json.loads(oauth_state_bytes.decode("utf-8"))

    desired_return_url = cast(str, oauth_state[_DESIRED_RETURN_URL_KEY])
    additional_kwargs = cast(dict[str, str], oauth_state[_ADDITIONAL_KWARGS_KEY])

    base_url = WEB_DOMAIN
    token_info = connector_cls.oauth_code_to_token(base_url, code, additional_kwargs)

    # Create a new credential with the token info
    credential_data = CredentialBase(
        credential_json=token_info,
        admin_public=True,  # Or based on some logic/parameter
        source=source,
        name=f"{source.title()} OAuth Credential",
    )

    credential = create_credential(
        credential_data=credential_data,
        user=user,
        db_session=db_session,
    )

    return CallbackResponse(
        redirect_url=(
            f"{desired_return_url}?credentialId={credential.id}"
            if "?" not in desired_return_url
            else f"{desired_return_url}&credentialId={credential.id}"
        )
    )


class OAuthAdditionalKwargDescription(BaseModel):
    """
    OAuth额外参数描述模型
    
    属性：
        name: 参数名
        display_name: 显示名称
        description: 参数描述
    """
    name: str
    display_name: str
    description: str


class OAuthDetails(BaseModel):
    """
    OAuth详情模型
    
    属性：
        oauth_enabled: 是否启用OAuth
        additional_kwargs: 额外参数列表
    """
    oauth_enabled: bool
    additional_kwargs: list[OAuthAdditionalKwargDescription]


@router.get("/details/{source}")
def oauth_details(
    source: DocumentSource,
    _: User = Depends(current_user),
) -> OAuthDetails:
    """
    获取OAuth源的详细信息
    
    参数：
        source: 文档源
        _: 当前用户（用于认证）
        
    返回值：
        OAuthDetails: OAuth详情对象
    """
    oauth_connectors = _discover_oauth_connectors()

    if source not in oauth_connectors:
        return OAuthDetails(
            oauth_enabled=False,
            additional_kwargs=[],
        )

    connector_cls = oauth_connectors[source]

    additional_kwarg_descriptions = []
    for key, value in connector_cls.AdditionalOauthKwargs.model_json_schema()[
        "properties"
    ].items():
        additional_kwarg_descriptions.append(
            OAuthAdditionalKwargDescription(
                name=key,
                display_name=value.get("title", key),
                description=value.get("description", ""),
            )
        )

    return OAuthDetails(
        oauth_enabled=True,
        additional_kwargs=additional_kwarg_descriptions,
    )
