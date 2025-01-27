"""
这个文件是认证模块的核心文件,主要实现了以下功能:
1. 用户管理(创建、验证、更新等)
2. 认证后端和策略(包括基于Redis的会话管理)
3. OAuth认证流程
4. API密钥认证
5. 权限控制和用户角色管理
"""

import json
import secrets
import uuid
from collections.abc import AsyncGenerator
from datetime import datetime
from datetime import timezone
from typing import cast
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import jwt
from email_validator import EmailNotValidError
from email_validator import EmailUndeliverableError
from email_validator import validate_email
from fastapi import APIRouter
from fastapi import Depends
from fastapi import HTTPException
from fastapi import Query
from fastapi import Request
from fastapi import Response
from fastapi import status
from fastapi.responses import RedirectResponse
from fastapi.security import OAuth2PasswordRequestForm
from fastapi_users import BaseUserManager
from fastapi_users import exceptions
from fastapi_users import FastAPIUsers
from fastapi_users import models
from fastapi_users import schemas
from fastapi_users import UUIDIDMixin
from fastapi_users.authentication import AuthenticationBackend
from fastapi_users.authentication import CookieTransport
from fastapi_users.authentication import RedisStrategy
from fastapi_users.authentication import Strategy
from fastapi_users.exceptions import UserAlreadyExists
from fastapi_users.jwt import decode_jwt
from fastapi_users.jwt import generate_jwt
from fastapi_users.jwt import SecretType
from fastapi_users.manager import UserManagerDependency
from fastapi_users.openapi import OpenAPIResponseType
from fastapi_users.router.common import ErrorCode
from fastapi_users.router.common import ErrorModel
from fastapi_users_db_sqlalchemy import SQLAlchemyUserDatabase
from httpx_oauth.integrations.fastapi import OAuth2AuthorizeCallback
from httpx_oauth.oauth2 import BaseOAuth2
from httpx_oauth.oauth2 import OAuth2Token
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from onyx.auth.api_key import get_hashed_api_key_from_request
from onyx.auth.email_utils import send_forgot_password_email
from onyx.auth.email_utils import send_user_verification_email
from onyx.auth.invited_users import get_invited_users
from onyx.auth.schemas import UserCreate
from onyx.auth.schemas import UserRole
from onyx.auth.schemas import UserUpdate
from onyx.configs.app_configs import AUTH_TYPE
from onyx.configs.app_configs import DISABLE_AUTH
from onyx.configs.app_configs import EMAIL_CONFIGURED
from onyx.configs.app_configs import REDIS_AUTH_EXPIRE_TIME_SECONDS
from onyx.configs.app_configs import REDIS_AUTH_KEY_PREFIX
from onyx.configs.app_configs import REQUIRE_EMAIL_VERIFICATION
from onyx.configs.app_configs import SESSION_EXPIRE_TIME_SECONDS
from onyx.configs.app_configs import TRACK_EXTERNAL_IDP_EXPIRY
from onyx.configs.app_configs import USER_AUTH_SECRET
from onyx.configs.app_configs import VALID_EMAIL_DOMAINS
from onyx.configs.app_configs import WEB_DOMAIN
from onyx.configs.constants import AuthType
from onyx.configs.constants import DANSWER_API_KEY_DUMMY_EMAIL_DOMAIN
from onyx.configs.constants import DANSWER_API_KEY_PREFIX
from onyx.configs.constants import MilestoneRecordType
from onyx.configs.constants import OnyxRedisLocks
from onyx.configs.constants import PASSWORD_SPECIAL_CHARS
from onyx.configs.constants import UNNAMED_KEY_PLACEHOLDER
from onyx.db.api_key import fetch_user_for_api_key
from onyx.db.auth import get_default_admin_user_emails
from onyx.db.auth import get_user_count
from onyx.db.auth import get_user_db
from onyx.db.auth import SQLAlchemyUserAdminDB
from onyx.db.engine import get_async_session
from onyx.db.engine import get_async_session_with_tenant
from onyx.db.engine import get_session_with_tenant
from onyx.db.models import OAuthAccount
from onyx.db.models import User
from onyx.db.users import get_user_by_email
from onyx.redis.redis_pool import get_async_redis_connection
from onyx.redis.redis_pool import get_redis_client
from onyx.utils.logger import setup_logger
from onyx.utils.telemetry import create_milestone_and_report
from onyx.utils.telemetry import optional_telemetry
from onyx.utils.telemetry import RecordType
from onyx.utils.variable_functionality import fetch_ee_implementation_or_noop
from onyx.utils.variable_functionality import fetch_versioned_implementation
from shared_configs.configs import async_return_default_schema
from shared_configs.configs import MULTI_TENANT
from shared_configs.contextvars import CURRENT_TENANT_ID_CONTEXTVAR

logger = setup_logger()


class BasicAuthenticationError(HTTPException):
    """
    基础认证错误类，继承自HTTPException
    
    用途:
    - 封装认证相关的错误,返回403状态码
    
    参数:
    - detail: 错误详情字符串
    """
    def __init__(self, detail: str):
        super().__init__(status_code=status.HTTP_403_FORBIDDEN, detail=detail)


def is_user_admin(user: User | None) -> bool:
    """
    检查用户是否为管理员
    
    用途:
    - 判断用户是否具有管理员权限
    
    参数:
    - user: 用户对象或None
    
    返回:
    - bool: 用户是否为管理员
    """
    if AUTH_TYPE == AuthType.DISABLED:
        return True
    if user and user.role == UserRole.ADMIN:
        return True
    return False


def verify_auth_setting() -> None:
    """
    验证认证设置是否有效
    
    用途:
    - 检查AUTH_TYPE配置是否为有效值
    - 记录当前使用的认证类型
    """
    if AUTH_TYPE not in [AuthType.DISABLED, AuthType.BASIC, AuthType.GOOGLE_OAUTH]:
        raise ValueError(
            "User must choose a valid user authentication method: " # 用户必须选择一个有效的认证方法
            "disabled, basic, or google_oauth"
        )
    logger.notice(f"Using Auth Type: {AUTH_TYPE.value}")  # 使用认证类型


def get_display_email(email: str | None, space_less: bool = False) -> str:
    """
    获取显示用的邮箱地址
    
    用途:
    - 格式化API密钥相关的邮箱显示
    - 处理未命名API密钥的显示名称
    
    参数:
    - email: 邮箱地址或None
    - space_less: 是否移除空格
    
    返回:
    - str: 格式化后的显示名称
    """
    if email and email.endswith(DANSWER_API_KEY_DUMMY_EMAIL_DOMAIN):
        name = email.split("@")[0]
        if name == DANSWER_API_KEY_PREFIX + UNNAMED_KEY_PLACEHOLDER:
            return "Unnamed API Key"  # 未命名的API密钥

        if space_less:
            return name

        return name.replace("API_KEY__", "API Key: ")

    return email or ""


def user_needs_to_be_verified() -> bool:
    """
    检查用户是否需要验证
    
    用途:
    - 判断当前认证类型下是否需要邮箱验证
    - 对于外部IDP认证的用户,认为已经通过验证
    
    返回:
    - bool: 是否需要验证
    """
    if AUTH_TYPE == AuthType.BASIC或AUTH_TYPE == AuthType.CLOUD:
        return REQUIRE_EMAIL_VERIFICATION

    # 对于其他认证类型,如果用户已通过认证,则认为用户已通过外部身份提供商验证
    return False


def anonymous_user_enabled() -> bool:
    """
    检查是否启用匿名用户访问
    
    用途:
    - 检查系统是否允许匿名用户访问
    - 在多租户模式下始终返回False
    
    返回:
    - bool: 是否启用匿名访问
    """
    if MULTI_TENANT:
        return False

    redis_client = get_redis_client(tenant_id=None)
    value = redis_client.get(OnyxRedisLocks.ANONYMOUS_USER_ENABLED)

    if value is None:
        return False

    assert isinstance(value, bytes)
    return int(value.decode("utf-8")) == 1


def verify_email_is_invited(email: str) -> None:
    """
    验证邮箱是否在邀请列表中
    
    用途:
    - 检查用户邮箱是否在白名单中
    - 验证邮箱格式是否有效
    
    参数:
    - email: 需要验证的邮箱地址
    
    抛出:
    - PermissionError: 当邮箱不在白名单中或格式无效时
    """
    whitelist = get_invited_users()
    if not whitelist:
        return

    if not email:
        raise PermissionError("Email must be specified") # 邮箱必须指定

    try:
        email_info = validate_email(email)
    except EmailUndeliverableError:
        raise PermissionError("Email is not valid") # 邮箱无效

    # 现在正在将规范化的邮箱插入数据库
    # 一段时间后我们可以移除读取时的规范化操作
    for email_whitelist in whitelist:
        try:
            email_info_whitelist = validate_email(email_whitelist)
        except EmailNotValidError:
            continue

        # 奇怪的是,规范化不包括小写邮箱地址的用户名部分...我们希望允许这种情况
        if email_info.normalized.lower() == email_info_whitelist.normalized.lower():
            return

    raise PermissionError("User not on allowed user whitelist") # 用户不在允许的白名单中


def verify_email_in_whitelist(email: str, tenant_id: str | None = None) -> None:
    """
    验证邮箱是否在白名单中(支持多租户)
    
    用途:
    - 在多租户环境下验证邮箱权限
    - 如果用户不存在则检查邀请列表
    
    参数:
    - email: 需要验证的邮箱
    - tenant_id: 租户ID,可选
    """
    with get_session_with_tenant(tenant_id) as db_session:
        if not get_user_by_email(email, db_session):
            verify_email_is_invited(email)


def verify_email_domain(email: str) -> None:
    """
    验证邮箱域名是否合法
    
    用途:
    - 检查邮箱域名是否在允许的域名列表中
    
    参数:
    - email: 需要验证的邮箱
    
    抛出:
    - HTTPException: 当邮箱格式无效或域名不在允许列表中时
    """
    if VALID_EMAIL_DOMAINS:
        if email.count("@") != 1:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email is not valid", # 邮箱格式无效
            )
        domain = email.split("@")[-1]
        if domain not in VALID_EMAIL_DOMAINS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,  
                detail="Email domain is not valid", # 邮箱域名无效
            )


class UserManager(UUIDIDMixin, BaseUserManager[User, uuid.UUID]):
    """
    用户管理器类，处理用户的创建、认证等核心功能
    
    用途:
    - 处理用户注册、验证和更新
    - 管理OAuth认证流程
    - 处理密码重置和邮箱验证
    
    属性:
    - reset_password_token_secret: 重置密码token的密钥
    - verification_token_secret: 验证邮箱token的密钥
    - user_db: 用户数据库实例
    """
    reset_password_token_secret = USER_AUTH_SECRET
    verification_token_secret = USER_AUTH_SECRET

    user_db: SQLAlchemyUserDatabase[User, uuid.UUID]

    async def create(
        self,
        user_create: schemas.UC | UserCreate,
        safe: bool = False,
        request: Optional[Request] = None,
    ) -> User:
        """
        创建新用户
        
        用途:
        - 验证用户密码
        - 处理多租户场景下的用户创建
        - 设置用户角色
        
        参数:
        - user_create: 用户创建模型
        - safe: 是否安全模式
        - request: 请求对象
        
        返回:
        - User: 创建的用户对象
        """
        # We verify the password here to make sure it's valid before we proceed
        await self.validate_password(
            user_create.password, cast(schemas.UC, user_create)
        )

        user_count: int | None = None
        referral_source = (
            request.cookies.get("referral_source", None)
            if request is not None
            else None
        )

        tenant_id = await fetch_ee_implementation_or_noop(
            "onyx.server.tenants.provisioning",
            "get_or_provision_tenant",
            async_return_default_schema,
        )(
            email=user_create.email,
            referral_source=referral_source,
            request=request,
        )

        async with get_async_session_with_tenant(tenant_id) as db_session:
            token = CURRENT_TENANT_ID_CONTEXTVAR.set(tenant_id)

            verify_email_is_invited(user_create.email)
            verify_email_domain(user_create.email)
            if MULTI_TENANT:
                tenant_user_db = SQLAlchemyUserAdminDB[User, uuid.UUID](
                    db_session, User, OAuthAccount
                )
                self.user_db = tenant_user_db
                self.database = tenant_user_db

            if hasattr(user_create, "role"):
                user_count = await get_user_count()
                if (
                    user_count == 0
                    or user_create.email in get_default_admin_user_emails()
                ):
                    user_create.role = UserRole.ADMIN
                else:
                    user_create.role = UserRole.BASIC

            try:
                user = await super().create(user_create, safe=safe, request=request)  # type: ignore
            except exceptions.UserAlreadyExists:
                user = await self.get_by_email(user_create.email)
                # 处理用户在网页外使用产品并现在通过网页创建账号的情况
                if not user.role.is_web_login() and user_create.role.is_web_login():
                    user_update = UserUpdate(
                        password=user_create.password,
                        is_verified=user_create.is_verified,
                    )
                    user = await self.update(user_update, user)
                else:
                    raise exceptions.UserAlreadyExists()

            finally:
                CURRENT_TENANT_ID_CONTEXTVAR.reset(token)

        return user

    async def validate_password(self, password: str, _: schemas.UC | models.UP) -> None:
        """
        验证密码是否符合安全要求
        
        用途:
        - 检查密码长度
        - 验证密码复杂度要求(大小写字母、数字、特殊字符)
        
        参数:
        - password: 待验证的密码
        - _: 用户创建或更新模型
        
        抛出:
        - InvalidPasswordException: 当密码不符合要求时
        """
        # Validate password according to basic security guidelines
        if len(password) < 12:
            raise exceptions.InvalidPasswordException(
                reason="Password must be at least 12 characters long."
            )
        if len(password) > 64:
            raise exceptions.InvalidPasswordException(
                reason="Password must not exceed 64 characters."
            )
        if not any(char.isupper() for char in password):
            raise exceptions.InvalidPasswordException(
                reason="Password must contain at least one uppercase letter."
            )
        if not any(char.islower() for char in password):
            raise exceptions.InvalidPasswordException(
                reason="Password must contain at least one lowercase letter."
            )
        if not any(char.isdigit() for char in password):
            raise exceptions.InvalidPasswordException(
                reason="Password must contain at least one number."
            )
        if not any(char in PASSWORD_SPECIAL_CHARS for char in password):
            raise exceptions.InvalidPasswordException(
                reason="Password must contain at least one special character from the following set: "
                f"{PASSWORD_SPECIAL_CHARS}."
            )

        return

    async def oauth_callback(
        self,
        oauth_name: str,
        access_token: str,
        account_id: str,
        account_email: str,
        expires_at: Optional[int] = None,
        refresh_token: Optional[str] = None,
        request: Optional[Request] = None,
        *,
        associate_by_email: bool = False,
        is_verified_by_default: bool = False,
    ) -> User:
        """
        处理OAuth回调
        
        用途:
        - 处理OAuth认证回调
        - 创建或更新用户账号
        - 关联OAuth账号信息
        
        参数:
        - oauth_name: OAuth提供商名称
        - access_token: 访问令牌
        - account_id: 账号ID
        - account_email: 账号邮箱
        - expires_at: 令牌过期时间
        - refresh_token: 刷新令牌
        - request: 请求对象
        - associate_by_email: 是否通过邮箱关联账号
        - is_verified_by_default: 新用户是否默认验证
        
        返回:
        - User: 认证用户对象
        
        抛出:
        - HTTPException: 当用户未找到时
        """
        referral_source = (
            getattr(request.state, "referral_source", None) if request else None
        )

        tenant_id = await fetch_ee_implementation_or_noop(
            "onyx.server.tenants.provisioning",
            "get_or_provision_tenant",
            async_return_default_schema,
        )(
            email=account_email,
            referral_source=referral_source,
            request=request,
        )

        if not tenant_id:
            raise HTTPException(status_code=401, detail="User not found")

        # 继续处理租户上下文
        token = None
        async with get_async_session_with_tenant(tenant_id) as db_session:
            token = CURRENT_TENANT_ID_CONTEXTVAR.set(tenant_id)

            verify_email_in_whitelist(account_email, tenant_id)
            verify_email_domain(account_email)

            if MULTI_TENANT:
                tenant_user_db = SQLAlchemyUserAdminDB[User, uuid.UUID](
                    db_session, User, OAuthAccount
                )
                self.user_db = tenant_user_db
                self.database = tenant_user_db

            oauth_account_dict = {
                "oauth_name": oauth_name,
                "access_token": access_token,
                "account_id": account_id,
                "account_email": account_email,
                "expires_at": expires_at,
                "refresh_token": refresh_token,
            }

            try:
                # Attempt to get user by OAuth account
                user = await self.get_by_oauth_account(oauth_name, account_id)

            except exceptions.UserNotExists:
                try:
                    # Attempt to get user by email
                    user = await self.get_by_email(account_email)
                    if not associate_by_email:
                        raise exceptions.UserAlreadyExists()

                    user = await self.user_db.add_oauth_account(
                        user, oauth_account_dict
                    )

                    # If user not found by OAuth account or email, create a new user
                except exceptions.UserNotExists:
                    password = self.password_helper.generate()
                    user_dict = {
                        "email": account_email,
                        "hashed_password": self.password_helper.hash(password),
                        "is_verified": is_verified_by_default,
                    }

                    user = await self.user_db.create(user_dict)

                    # Explicitly set the Postgres schema for this session to ensure
                    # OAuth account creation happens in the correct tenant schema

                    # Add OAuth account
                    await self.user_db.add_oauth_account(user, oauth_account_dict)
                    await self.on_after_register(user, request)

            else:
                for existing_oauth_account in user.oauth_accounts:
                    if (
                        existing_oauth_account.account_id == account_id
                        and existing_oauth_account.oauth_name == oauth_name
                    ):
                        user = await self.user_db.update_oauth_account(
                            user,
                            # 注意: OAuthAccount确实实现了OAuthAccountProtocol
                            # 但类型检查器无法识别这一点 :(
                            existing_oauth_account,  # type: ignore
                            oauth_account_dict,
                        )

            # 大多数身份提供商的令牌过期时间很短,我们不想强制用户频繁重新认证,所以默认禁用此功能 
            if expires_at and TRACK_EXTERNAL_IDP_EXPIRY:
                oidc_expiry = datetime.fromtimestamp(expires_at, tz=timezone.utc)
                await self.user_db.update(
                    user, update_dict={"oidc_expiry": oidc_expiry}
                )

            # 处理用户在网页外使用产品并现在通过网页创建账号的情况
            if not user.role.is_web_login():
                await self.user_db.update(
                    user,
                    {
                        "is_verified": is_verified_by_default,
                        "role": UserRole.BASIC,
                    },
                )
                user.is_verified = is_verified_by_default

            # 这是组织从启用外部IDP过期追踪切换到禁用时所需的
            # 否则,oidc过期时间将始终是旧的,用户将永远无法登录
            if (
                user.oidc_expiry is not None  # type: ignore
                and not TRACK_EXTERNAL_IDP_EXPIRY
            ):
                await self.user_db.update(user, {"oidc_expiry": None})
                user.oidc_expiry = None  # type: ignore

            if token:
                CURRENT_TENANT_ID_CONTEXTVAR.reset(token)

            return user

    async def on_after_register(
        self, user: User, request: Optional[Request] = None
    ) -> None:
        """
        用户注册后的回调函数
        
        用途:
        - 记录用户注册里程碑
        - 发送遥测数据
        - 多租户环境下的用户注册处理
        
        参数:
        - user: 新注册的用户
        - request: 可选的请求对象
        """
        tenant_id = await fetch_ee_implementation_or_noop(
            "onyx.server.tenants.provisioning",
            "get_or_provision_tenant",
            async_return_default_schema,
        )(
            email=user.email,
            request=request,
        )

        token = CURRENT_TENANT_ID_CONTEXTVAR.set(tenant_id)
        try:
            user_count = await get_user_count()

            with get_session_with_tenant(tenant_id=tenant_id) as db_session:
                if user_count == 1:
                    create_milestone_and_report(
                        user=user,
                        distinct_id=user.email,
                        event_type=MilestoneRecordType.USER_SIGNED_UP,
                        properties=None,
                        db_session=db_session,
                    )
                else:
                    create_milestone_and_report(
                        user=user,
                        distinct_id=user.email,
                        event_type=MilestoneRecordType.MULTIPLE_USERS,
                        properties=None,
                        db_session=db_session,
                    )
        finally:
            CURRENT_TENANT_ID_CONTEXTVAR.reset(token)

        logger.notice(f"User {user.id} has registered.")
        optional_telemetry(
            record_type=RecordType.SIGN_UP,
            data={"action": "create"},
            user_id=str(user.id),
        )

    async def on_after_forgot_password(
        self, user: User, token: str, request: Optional[Request] = None
    ) -> None:
        """
        忘记密码请求后的回调函数
        
        用途:
        - 验证邮件配置是否启用
        - 发送重置密码邮件
        
        参数:
        - user: 请求重置密码的用户
        - token: 重置密码令牌
        - request: 可选的请求对象
        
        抛出:
        - HTTPException: 当邮件功能未配置时
        """
        if not EMAIL_CONFIGURED:
            logger.error(
                "Email is not configured. Please configure email in the admin panel"
            )
            raise HTTPException(
                status.HTTP_500_INTERNAL_SERVER_ERROR,
                "Your admin has not enbaled this feature.",
            )
        send_forgot_password_email(user.email, token)

    async def on_after_request_verify(
        self, user: User, token: str, request: Optional[Request] = None
    ) -> None:
        """
        请求验证邮箱后的回调函数
        
        用途:
        - 验证邮箱域名
        - 记录验证请求
        - 发送验证邮件
        
        参数:
        - user: 请求验证的用户
        - token: 验证令牌
        - request: 可选的请求对象
        """
        verify_email_domain(user.email)

        logger.notice(
            f"Verification requested for user {user.id}. Verification token: {token}"
        )

        send_user_verification_email(user.email, token)

    async def authenticate(
        self, credentials: OAuth2PasswordRequestForm
    ) -> Optional[User]:
        """
        验证用户凭据
        
        用途:
        - 支持多租户环境下的用户认证
        - 验证用户密码
        - 处理密码哈希更新
        
        参数:
        - credentials: OAuth2密码表单凭据
        
        返回:
        - User | None: 认证成功返回用户对象，失败返回None
        
        抛出:
        - BasicAuthenticationError: 当用户不支持网页登录时
        """
        email = credentials.username

        # Get tenant_id from mapping table
        tenant_id = await fetch_ee_implementation_or_noop(
            "onyx.server.tenants.provisioning",
            "get_or_provision_tenant",
            async_return_default_schema,
        )(
            email=email,
        )
        if not tenant_id:
            # User not found in mapping
            self.password_helper.hash(credentials.password)
            return None

        # Create a tenant-specific session
        async with get_async_session_with_tenant(tenant_id) as tenant_session:
            tenant_user_db: SQLAlchemyUserDatabase = SQLAlchemyUserDatabase(
                tenant_session, User
            )
            self.user_db = tenant_user_db

            # Proceed with authentication
            try:
                user = await self.get_by_email(email)

            except exceptions.UserNotExists:
                self.password_helper.hash(credentials.password)
                return None

            if not user.role.is_web_login():
                raise BasicAuthenticationError(
                    detail="NO_WEB_LOGIN_AND_HAS_NO_PASSWORD",
                )

            verified, updated_password_hash = self.password_helper.verify_and_update(
                credentials.password, user.hashed_password
            )
            if not verified:
                return None

            if updated_password_hash is not None:
                await self.user_db.update(
                    user, {"hashed_password": updated_password_hash}
                )

            return user


async def get_user_manager(
    user_db: SQLAlchemyUserDatabase = Depends(get_user_db),
) -> AsyncGenerator[UserManager, None]:
    """
    用户管理器工厂函数
    
    用途:
    - 创建UserManager实例的依赖工厂
    - 管理用户数据库连接
    
    参数:
    - user_db: SQLAlchemy用户数据库实例
    
    返回:
    - AsyncGenerator[UserManager, None]: 生成UserManager实例的异步生成器
    """
    yield UserManager(user_db)


cookie_transport = CookieTransport(
    cookie_max_age=SESSION_EXPIRE_TIME_SECONDS,
    cookie_secure=WEB_DOMAIN.startswith("https"),
)


def get_redis_strategy() -> RedisStrategy:
    return TenantAwareRedisStrategy()


class TenantAwareRedisStrategy(RedisStrategy[User, uuid.UUID]):
    """
    支持多租户的Redis令牌存储策略
    
    用途:
    - 在Redis中管理用户认证令牌
    - 支持多租户隔离
    
    属性:
    - lifetime_seconds: 令牌有效期
    - key_prefix: Redis键前缀
    """

    def __init__(
        self,
        lifetime_seconds: Optional[int] = REDIS_AUTH_EXPIRE_TIME_SECONDS,
        key_prefix: str = REDIS_AUTH_KEY_PREFIX,
    ):
        self.lifetime_seconds = lifetime_seconds
        self.key_prefix = key_prefix

    async def write_token(self, user: User) -> str:
        """
        写入认证令牌
        
        用途:
        - 为用户生成新的认证令牌
        - 将令牌数据存储到Redis
        
        参数:
        - user: 用户对象
        
        返回:
        - str: 生成的令牌
        """
        redis = await get_async_redis_connection()

        tenant_id = await fetch_ee_implementation_or_noop(
            "onyx.server.tenants.provisioning",
            "get_or_provision_tenant",
            async_return_default_schema,
        )(email=user.email)

        token_data = {
            "sub": str(user.id),
            "tenant_id": tenant_id,
        }
        token = secrets.token_urlsafe()
        await redis.set(
            f"{self.key_prefix}{token}",
            json.dumps(token_data),
            ex=self.lifetime_seconds,
        )
        return token

    async def read_token(
        self, token: Optional[str], user_manager: BaseUserManager[User, uuid.UUID]
    ) -> Optional[User]:
        """
        读取并验证令牌
        
        用途:
        - 从Redis获取令牌数据
        - 验证令牌有效性并返回对应用户
        
        参数:
        - token: 认证令牌
        - user_manager: 用户管理器实例
        
        返回:
        - User | None: 令牌对应的用户或None
        """
        redis = await get_async_redis_connection()
        token_data_str = await redis.get(f"{self.key_prefix}{token}")
        if not token_data_str:
            return None

        try:
            token_data = json.loads(token_data_str)
            user_id = token_data["sub"]
            parsed_id = user_manager.parse_id(user_id)
            return await user_manager.get(parsed_id)
        except (exceptions.UserNotExists, exceptions.InvalidID, KeyError):
            return None

    async def destroy_token(self, token: str, user: User) -> None:
        """从异步redis中正确删除令牌"""
        redis = await get_async_redis_connection()
        await redis.delete(f"{self.key_prefix}{token}")


auth_backend = AuthenticationBackend(
    name="redis", transport=cookie_transport, get_strategy=get_redis_strategy
)


class FastAPIUserWithLogoutRouter(FastAPIUsers[models.UP, models.ID]):
    """
    扩展的FastAPIUsers路由器，增加登出功能
    
    用途:
    - 为OAuth/OIDC流程提供登出功能
    - 不需要包含登录路由
    """

    def get_logout_router(
        self,
        backend: AuthenticationBackend,
        requires_verification: bool = REQUIRE_EMAIL_VERIFICATION,
    ) -> APIRouter:
        """
        获取登出路由
        
        用途:
        - 创建处理用户登出的路由
        - 配置OpenAPI文档
        
        参数:
        - backend: 认证后端
        - requires_verification: 是否要求邮箱验证
        
        返回:
        - APIRouter: 登出功能的路由器
        """
        router = APIRouter()

        get_current_user_token = self.authenticator.current_user_token(
            active=True, verified=requires_verification
        )

        logout_responses: OpenAPIResponseType = {
            **{
                status.HTTP_401_UNAUTHORIZED: {
                    "description": "Missing token or inactive user."
                }
            },
            **backend.transport.get_openapi_logout_responses_success(),
        }

        @router.post(
            "/logout", name=f"auth:{backend.name}.logout", responses=logout_responses
        )
        async def logout(
            user_token: Tuple[models.UP, str] = Depends(get_current_user_token),
            strategy: Strategy[models.UP, models.ID] = Depends(backend.get_strategy),
        ) -> Response:
            user, token = user_token
            return await backend.logout(strategy, user, token)

        return router


fastapi_users = FastAPIUserWithLogoutRouter[User, uuid.UUID](
    get_user_manager, [auth_backend]
)


# 这里不使用verified=REQUIRE_EMAIL_VERIFICATION,因为我们在double_check_user中自行处理
# 这是必需的,因为我们希望/me端点即使在用户未验证时也返回用户信息,以便前端知道用户存在
optional_fastapi_current_user = fastapi_users.current_user(active=True, optional=True)


async def optional_user_(
    request: Request,
    user: User | None,
    async_db_session: AsyncSession,
) -> User | None:
    """
    可选用户辅助函数
    
    用途: 
    - 为企业版功能预留的基础实现
    - 在开源版本中直接返回用户
    
    参数:
    - request: 请求对象 
    - user: 可选的用户对象
    - async_db_session: 异步数据库会话
    
    返回:
    - User | None: 输入的用户对象
    """
    return user


async def optional_user(
    request: Request,
    async_db_session: AsyncSession = Depends(get_async_session),
    user: User | None = Depends(optional_fastapi_current_user),
) -> User | None:
    versioned_fetch_user = fetch_versioned_implementation(
        "onyx.auth.users", "optional_user_"
    )
    user = await versioned_fetch_user(request, user, async_db_session)

    # check if an API key is present
    if user is None:
        hashed_api_key = get_hashed_api_key_from_request(request)
        if hashed_api_key:
            user = await fetch_user_for_api_key(hashed_api_key, async_db_session)

    return user


async def double_check_user(
    user: User | None,
    optional: bool = DISABLE_AUTH,
    include_expired: bool = False,
    allow_anonymous_access: bool = False,
) -> User | None:
    """
    双重检查用户状态
    
    用途:
    - 验证用户认证状态
    - 检查用户是否需要验证
    - 检查令牌是否过期
    
    参数:
    - user: 待验证的用户
    - optional: 是否可选认证
    - include_expired: 是否包含过期令牌
    - allow_anonymous_access: 是否允许匿名访问
    
    返回:
    - User | None: 验证通过的用户或None
    
    抛出:
    - BasicAuthenticationError: 当验证失败时
    """
    if optional:
        return user

    if user is not None:
        # 如果用户尝试认证,则验证用户,如果失败不要默认允许匿名访问
        if user_needs_to_be_verified() and not user.is_verified:
            raise BasicAuthenticationError(
                detail="Access denied. User is not verified.",
            )

        if (
            user.oidc_expiry
            and user.oidc_expiry < datetime.now(timezone.utc)
            and not include_expired
        ):
            raise BasicAuthenticationError(
                detail="Access denied. User's OIDC token has expired.",
            )

        return user

    if allow_anonymous_access:
        return None

    raise BasicAuthenticationError(
        detail="Access denied. User is not authenticated.",
    )


async def current_user_with_expired_token(
    user: User | None = Depends(optional_user),
) -> User | None:
    """
    允许过期令牌的用户依赖
    
    用途:
    - 获取当前用户，包括令牌已过期的情况
    
    参数:
    - user: 可选的用户对象
    
    返回:
    - User | None: 当前用户或None
    """
    return await double_check_user(user, include_expired=True)


async def current_limited_user(
    user: User | None = Depends(optional_user),
) -> User | None:
    """
    基础限制用户依赖
    
    用途:
    - 获取当前用户，执行基本验证
    
    参数:
    - user: 可选的用户对象
    
    返回:
    - User | None: 验证后的用户或None
    """
    return await double_check_user(user)


async def current_chat_accesssible_user(
    user: User | None = Depends(optional_user),
) -> User | None:
    """
    聊天功能可访问用户依赖
    
    用途:
    - 验证用户是否可以访问聊天功能
    - 支持匿名访问模式
    
    参数:
    - user: 可选的用户对象
    
    返回:
    - User | None: 验证通过的用户或None
    """
    return await double_check_user(
        user, allow_anonymous_access=anonymous_user_enabled()
    )


async def current_user(
    user: User | None = Depends(optional_user),
) -> User | None:
    """
    标准用户依赖
    
    用途:
    - 验证用户身份
    - 检查用户权限级别
    
    参数:
    - user: 可选的用户对象
    
    返回:
    - User | None: 验证通过的用户或None
    
    抛出:
    - BasicAuthenticationError: 当用户权限不足时
    """
    user = await double_check_user(user)
    if not user:
        return None

    if user.role == UserRole.LIMITED:
        raise BasicAuthenticationError(
            detail="Access denied. User role is LIMITED. BASIC or higher permissions are required.",
        )
    return user


async def current_curator_or_admin_user(
    user: User | None = Depends(current_user),
) -> User | None:
    """
    内容管理员或系统管理员用户依赖
    
    用途:
    - 验证用户是否具有内容管理或系统管理权限
    
    参数:
    - user: 当前用户对象
    
    返回:
    - User | None: 验证通过的管理员用户或None
    
    抛出:
    - BasicAuthenticationError: 当用户不是管理员或内容管理员时
    """
    if DISABLE_AUTH:
        return None

    if not user or not hasattr(user, "role"):
        raise BasicAuthenticationError(
            detail="Access denied. User is not authenticated or lacks role information.",
        )

    allowed_roles = {UserRole.GLOBAL_CURATOR, UserRole.CURATOR, UserRole.ADMIN}
    if user.role not in allowed_roles:
        raise BasicAuthenticationError(
            detail="Access denied. User is not a curator or admin.",
        )

    return user


async def current_admin_user(user: User | None = Depends(current_user)) -> User | None:
    """
    管理员用户依赖
    
    用途:
    - 验证用户是否具有管理员权限
    
    参数:
    - user: 当前用户对象
    
    返回:
    - User | None: 管理员用户或None
    
    抛出:
    - BasicAuthenticationError: 当用户不是管理员时
    """
    if DISABLE_AUTH:
        return None

    if not user or not hasattr(user, "role") or user.role != UserRole.ADMIN:
        raise BasicAuthenticationError(
            detail="Access denied. User must be an admin to perform this action.", # 拒绝访问，用户必须是管理员才能执行此操作
        )

    return user


def get_default_admin_user_emails_() -> list[str]:
    # Onyx MIT版本无默认种子数据可用
    return []


STATE_TOKEN_AUDIENCE = "fastapi-users:oauth-state"


class OAuth2AuthorizeResponse(BaseModel):
    """
    OAuth2授权响应模型
    
    用途:
    - 封装OAuth授权URL的响应
    
    属性:
    - authorization_url: 授权重定向URL
    """
    authorization_url: str


def generate_state_token(
    data: Dict[str, str], secret: SecretType, lifetime_seconds: int = 3600
) -> str:
    """
    生成OAuth状态令牌
    
    用途:
    - 生成防止CSRF攻击的状态令牌
    - 在OAuth回调时验证请求合法性
    
    参数:
    - data: 需要编码的数据字典 
    - secret: 密钥
    - lifetime_seconds: 令牌有效期(秒)
    
    返回:
    - str: JWT格式的状态令牌
    """
    data["aud"] = STATE_TOKEN_AUDIENCE
    return generate_jwt(data, secret, lifetime_seconds)


# refer to https://github.com/fastapi-users/fastapi-users/blob/42ddc241b965475390e2bce887b084152ae1a2cd/fastapi_users/fastapi_users.py#L91
def create_onyx_oauth_router(
    oauth_client: BaseOAuth2,
    backend: AuthenticationBackend,
    state_secret: SecretType,
    redirect_url: Optional[str] = None,
    associate_by_email: bool = False,
    is_verified_by_default: bool = False,
) -> APIRouter:
    return get_oauth_router(
        oauth_client,
        backend,
        get_user_manager,
        state_secret,
        redirect_url,
        associate_by_email,
        is_verified_by_default,
    )


def get_oauth_router(
    oauth_client: BaseOAuth2,
    backend: AuthenticationBackend,
    get_user_manager: UserManagerDependency[models.UP, models.ID],
    state_secret: SecretType,
    redirect_url: Optional[str] = None,
    associate_by_email: bool = False,
    is_verified_by_default: bool = False,
) -> APIRouter:
    """
    创建OAuth路由器
    
    用途:
    - 创建处理OAuth认证流程的路由
    - 包含授权和回调接口
    
    参数:
    - oauth_client: OAuth客户端实例
    - backend: 认证后端
    - get_user_manager: 用户管理器工厂函数
    - state_secret: 状态令牌密钥
    - redirect_url: 可选的重定向URL
    - associate_by_email: 是否通过邮箱关联已有用户
    - is_verified_by_default: 新用户是否默认验证通过
    
    返回:
    - APIRouter: 包含OAuth路由的路由器
    """
    router = APIRouter()
    callback_route_name = f"oauth:{oauth_client.name}.{backend.name}.callback"

    if redirect_url is not None:
        oauth2_authorize_callback = OAuth2AuthorizeCallback(
            oauth_client,
            redirect_url=redirect_url,
        )
    else:
        oauth2_authorize_callback = OAuth2AuthorizeCallback(
            oauth_client,
            route_name=callback_route_name,
        )

    @router.get(
        "/authorize",
        name=f"oauth:{oauth_client.name}.{backend.name}.authorize",
        response_model=OAuth2AuthorizeResponse,
    )
    async def authorize(
        request: Request,
        scopes: List[str] = Query(None),
    ) -> OAuth2AuthorizeResponse:
        """
        OAuth授权端点
        
        用途:
        - 生成OAuth授权URL
        - 记录来源和重定向信息
        
        参数:
        - request: 请求对象
        - scopes: OAuth授权范围列表
        
        返回:
        - OAuth2AuthorizeResponse: 包含授权URL的响应
        """
        referral_source = request.cookies.get("referral_source", None)

        if redirect_url is not None:
            authorize_redirect_url = redirect_url
        else:
            authorize_redirect_url = str(request.url_for(callback_route_name))

        next_url = request.query_params.get("next", "/")

        state_data: Dict[str, str] = {
            "next_url": next_url,
            "referral_source": referral_source or "default_referral",
        }
        state = generate_state_token(state_data, state_secret)
        authorization_url = await oauth_client.get_authorization_url(
            authorize_redirect_url,
            state,
            scopes,
        )

        return OAuth2AuthorizeResponse(authorization_url=authorization_url)

    @router.get(
        "/callback",
        name=callback_route_name,
        description="响应根据使用的认证后端而变化",
        responses={
            status.HTTP_400_BAD_REQUEST: {
                "model": ErrorModel,
                "content": {
                    "application/json": {
                        "examples": {
                            "INVALID_STATE_TOKEN": {
                                "summary": "Invalid state token.",
                                "value": None,
                            },
                            ErrorCode.LOGIN_BAD_CREDENTIALS: {
                                "summary": "User is inactive.",
                                "value": {"detail": ErrorCode.LOGIN_BAD_CREDENTIALS},
                            },
                        }
                    }
                },
            },
        },
    )
    async def callback(
        request: Request,
        access_token_state: Tuple[OAuth2Token, str] = Depends(
            oauth2_authorize_callback
        ),
        user_manager: BaseUserManager[models.UP, models.ID] = Depends(get_user_manager),
        strategy: Strategy[models.UP, models.ID] = Depends(backend.get_strategy),
    ) -> RedirectResponse:
        """
        OAuth回调端点
        
        用途:
        - 处理OAuth提供商的回调请求
        - 验证状态令牌
        - 创建或更新用户
        - 完成用户登录
        
        参数:
        - request: 请求对象
        - access_token_state: OAuth令牌和状态信息
        - user_manager: 用户管理器实例
        - strategy: 认证策略实例
        
        返回:
        - RedirectResponse: 重定向响应
        
        抛出:
        - HTTPException: 当验证失败或用户已存在时
        """
        token, state = access_token_state
        account_id, account_email = await oauth_client.get_id_email(
            token["access_token"]
        )

        if account_email is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=ErrorCode.OAUTH_NOT_AVAILABLE_EMAIL,
            )

        try:
            state_data = decode_jwt(state, state_secret, [STATE_TOKEN_AUDIENCE])
        except jwt.DecodeError:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST)

        next_url = state_data.get("next_url", "/")
        referral_source = state_data.get("referral_source", None)

        request.state.referral_source = referral_source

        # Proceed to authenticate or create the user
        try:
            user = await user_manager.oauth_callback(
                oauth_client.name,
                token["access_token"],
                account_id,
                account_email,
                token.get("expires_at"),
                token.get("refresh_token"),
                request,
                associate_by_email=associate_by_email,
                is_verified_by_default=is_verified_by_default,
            )
        except UserAlreadyExists:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=ErrorCode.OAUTH_USER_ALREADY_EXISTS,
            )

        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=ErrorCode.LOGIN_BAD_CREDENTIALS,
            )

        # Login user
        response = await backend.login(strategy, user)
        await user_manager.on_after_login(user, request, response)

        # 从response复制headers和其他属性到redirect_response
        redirect_response = RedirectResponse(next_url, status_code=302)

        # Copy headers and other attributes from 'response' to 'redirect_response'
        for header_name, header_value in response.headers.items():
            redirect_response.headers[header_name] = header_value

        if hasattr(response, "body"):
            redirect_response.body = response.body
        if hasattr(response, "status_code"):
            redirect_response.status_code = response.status_code
        if hasattr(response, "media_type"):
            redirect_response.media_type = response.media_type
        return redirect_response

    return router


async def api_key_dep(
    request: Request, async_db_session: AsyncSession = Depends(get_async_session)
) -> User | None:
    """
    API密钥认证依赖
    
    用途:
    - 验证请求中的API密钥
    - 获取对应的用户
    
    参数:
    - request: 请求对象
    - async_db_session: 异步数据库会话
    
    返回:
    - User | None: API密钥对应的用户或None
    
    抛出:
    - HTTPException: 当API密钥缺失或无效时
    """
    if AUTH_TYPE == AuthType.DISABLED:
        return None

    hashed_api_key = get_hashed_api_key_from_request(request)
    if not hashed_api_key:
        raise HTTPException(status_code=401, detail="Missing API key")

    if hashed_api_key:
        user = await fetch_user_for_api_key(hashed_api_key, async_db_session)

    if user is None:
        raise HTTPException(status_code=401, detail="Invalid API key")

    return user
