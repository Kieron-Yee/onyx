# 导入所需的库和依赖
import threading
import uuid
from enum import Enum
from typing import cast

import requests
from sqlalchemy.orm import Session

from onyx.configs.app_configs import DISABLE_TELEMETRY
from onyx.configs.app_configs import ENTERPRISE_EDITION_ENABLED
from onyx.configs.constants import KV_CUSTOMER_UUID_KEY
from onyx.configs.constants import KV_INSTANCE_DOMAIN_KEY
from onyx.configs.constants import MilestoneRecordType
from onyx.db.engine import get_sqlalchemy_engine
from onyx.db.milestone import create_milestone_if_not_exists
from onyx.db.models import User
from onyx.key_value_store.factory import get_kv_store
from onyx.key_value_store.interface import KvKeyNotFoundError
from onyx.utils.variable_functionality import (
    fetch_versioned_implementation_with_fallback,
)
from onyx.utils.variable_functionality import noop_fallback
from shared_configs.configs import MULTI_TENANT

_DANSWER_TELEMETRY_ENDPOINT = "https://telemetry.onyx.app/anonymous_telemetry"
_CACHED_UUID: str | None = None
_CACHED_INSTANCE_DOMAIN: str | None = None


class RecordType(str, Enum):
    """遥测记录类型枚举
    VERSION: 版本信息
    SIGN_UP: 注册信息 
    USAGE: 使用情况
    LATENCY: 延迟数据
    FAILURE: 错误信息
    """
    VERSION = "version"
    SIGN_UP = "sign_up"
    USAGE = "usage"
    LATENCY = "latency"
    FAILURE = "failure"


def get_or_generate_uuid() -> str:
    """获取或生成客户 UUID
    如果已缓存则返回缓存的 UUID
    否则从 KV 存储中获取，如果不存在则生成新的 UUID 并存储
    """
    global _CACHED_UUID

    if _CACHED_UUID is not None:
        return _CACHED_UUID

    kv_store = get_kv_store()

    try:
        _CACHED_UUID = cast(str, kv_store.load(KV_CUSTOMER_UUID_KEY))
    except KvKeyNotFoundError:
        _CACHED_UUID = str(uuid.uuid4())
        kv_store.store(KV_CUSTOMER_UUID_KEY, _CACHED_UUID, encrypt=True)

    return _CACHED_UUID


def _get_or_generate_instance_domain() -> str | None:
    """获取或生成实例域名
    首先检查缓存
    然后从 KV 存储获取
    如果不存在则获取第一个用户的邮箱域名作为实例域名
    """
    global _CACHED_INSTANCE_DOMAIN

    if _CACHED_INSTANCE_DOMAIN is not None:
        return _CACHED_INSTANCE_DOMAIN

    kv_store = get_kv_store()

    try:
        _CACHED_INSTANCE_DOMAIN = cast(str, kv_store.load(KV_INSTANCE_DOMAIN_KEY))
    except KvKeyNotFoundError:
        with Session(get_sqlalchemy_engine()) as db_session:
            first_user = db_session.query(User).first()
            if first_user:
                _CACHED_INSTANCE_DOMAIN = first_user.email.split("@")[-1]
                kv_store.store(
                    KV_INSTANCE_DOMAIN_KEY, _CACHED_INSTANCE_DOMAIN, encrypt=True
                )

    return _CACHED_INSTANCE_DOMAIN


def optional_telemetry(
    record_type: RecordType, 
    data: dict, 
    user_id: str | None = None
) -> None:
    """可选的遥测数据收集函数
    
    Args:
        record_type: 遥测记录类型
        data: 需要发送的数据
        user_id: 用户ID，可选
        
    收集并发送遥测数据到服务器，在单独的线程中执行以减少性能影响
    如果 DISABLE_TELEMETRY 为 True 则不执行任何操作
    """
    if DISABLE_TELEMETRY:
        return

    try:

        def telemetry_logic() -> None:
            try:
                customer_uuid = get_or_generate_uuid()
                payload = {
                    "data": data,
                    "record": record_type,
                    # If None then it's a flow that doesn't include a user
                    # For cases where the User itself is None, a string is provided instead
                    "user_id": user_id,
                    "customer_uuid": customer_uuid,
                }
                if ENTERPRISE_EDITION_ENABLED:
                    payload["instance_domain"] = _get_or_generate_instance_domain()
                requests.post(
                    _DANSWER_TELEMETRY_ENDPOINT,
                    headers={"Content-Type": "application/json"},
                    json=payload,
                )
            except Exception:
                # This way it silences all thread level logging as well
                pass

        # Run in separate thread to have minimal overhead in main flows
        thread = threading.Thread(target=telemetry_logic, daemon=True)
        thread.start()
    except Exception:
        # Should never interfere with normal functions of Onyx
        pass


def mt_cloud_telemetry(
    distinct_id: str,
    event: MilestoneRecordType,
    properties: dict | None = None,
) -> None:
    """多租户云版本的遥测实现
    
    Args:
        distinct_id: 唯一标识符
        event: 里程碑事件类型
        properties: 事件属性，可选
        
    仅在多租户模式下执行
    用于 Onyx MT Cloud 版本的特定遥测功能
    """
    if not MULTI_TENANT:
        return

    # MIT version should not need to include any Posthog code
    # This is only for Onyx MT Cloud, this code should also never be hit, no reason for any orgs to
    # be running the Multi Tenant version of Onyx.
    fetch_versioned_implementation_with_fallback(
        module="onyx.utils.telemetry",
        attribute="event_telemetry",
        fallback=noop_fallback,
    )(distinct_id, event, properties)


def create_milestone_and_report(
    user: User | None,
    distinct_id: str, 
    event_type: MilestoneRecordType,
    properties: dict | None,
    db_session: Session,
) -> None:
    """创建里程碑记录并报告
    
    Args:
        user: 用户对象，可选
        distinct_id: 唯一标识符
        event_type: 里程碑事件类型
        properties: 事件属性，可选
        db_session: 数据库会话
        
    创建新的里程碑记录，并在创建成功时发送遥测数据
    """
    _, is_new = create_milestone_if_not_exists(user, event_type, db_session)
    if is_new:
        mt_cloud_telemetry(
            distinct_id=distinct_id,
            event=event_type,
            properties=properties,
        )
