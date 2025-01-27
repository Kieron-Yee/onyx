"""
此文件实现了一个混合存储的键值存储系统。
主要功能：
1. 提供基于Redis和PostgreSQL的混合键值存储实现
2. 支持数据加密存储
3. 实现了多租户隔离
4. 采用Redis作为缓存，PostgreSQL作为持久化存储
5. 支持在Redis故障时优雅降级到PostgreSQL
"""

import json
from collections.abc import Iterator
from contextlib import contextmanager
from typing import cast

from fastapi import HTTPException
from redis.client import Redis
from sqlalchemy import text
from sqlalchemy.orm import Session

from onyx.db.engine import get_sqlalchemy_engine
from onyx.db.engine import is_valid_schema_name
from onyx.db.models import KVStore
from onyx.key_value_store.interface import KeyValueStore
from onyx.key_value_store.interface import KvKeyNotFoundError
from onyx.redis.redis_pool import get_redis_client
from onyx.utils.logger import setup_logger
from onyx.utils.special_types import JSON_ro
from shared_configs.configs import MULTI_TENANT
from shared_configs.configs import POSTGRES_DEFAULT_SCHEMA
from shared_configs.contextvars import CURRENT_TENANT_ID_CONTEXTVAR

logger = setup_logger()


REDIS_KEY_PREFIX = "onyx_kv_store:"
KV_REDIS_KEY_EXPIRATION = 60 * 60 * 24  # 1 Day


class PgRedisKVStore(KeyValueStore):
    """
    PostgreSQL和Redis混合键值存储实现类。
    
    提供了一个使用Redis作为缓存层，PostgreSQL作为持久化存储的键值存储实现。
    当Redis操作失败时，会自动降级到PostgreSQL存储。
    支持多租户隔离和数据加密功能。
    """

    def __init__(
        self, redis_client: Redis | None = None, tenant_id: str | None = None
    ) -> None:
        """
        初始化键值存储实例。

        参数:
            redis_client: Redis客户端实例，如果为None则使用默认客户端
            tenant_id: 租户ID，用于多租户隔离，如果为None则从上下文变量获取
        """
        # 如果没有提供redis_client，则使用上下文变量中的租户ID获取默认客户端
        if redis_client is not None:
            self.redis_client = redis_client
        else:
            tenant_id = tenant_id or CURRENT_TENANT_ID_CONTEXTVAR.get()
            self.redis_client = get_redis_client(tenant_id=tenant_id)

    @contextmanager
    def get_session(self) -> Iterator[Session]:
        """
        获取数据库会话的上下文管理器。
        
        根据多租户配置设置正确的schema搜索路径。
        如果启用了多租户功能，会进行租户验证和schema设置。

        返回:
            SQLAlchemy会话对象

        异常:
            HTTPException: 当租户验证失败或租户ID无效时抛出
        """
        engine = get_sqlalchemy_engine()
        with Session(engine, expire_on_commit=False) as session:
            if MULTI_TENANT:
                tenant_id = CURRENT_TENANT_ID_CONTEXTVAR.get()
                if tenant_id == POSTGRES_DEFAULT_SCHEMA:
                    raise HTTPException(
                        status_code=401, detail="User must authenticate"
                    )
                if not is_valid_schema_name(tenant_id):
                    raise HTTPException(status_code=400, detail="Invalid tenant ID")
                # Set the search_path to the tenant's schema
                session.execute(text(f'SET search_path = "{tenant_id}"'))
            yield session

    def store(self, key: str, val: JSON_ro, encrypt: bool = False) -> None:
        """
        存储键值对。

        首先尝试存储到Redis，如果失败则优雅降级。
        然后将数据持久化到PostgreSQL，支持加密存储。

        参数:
            key: 键名
            val: 要存储的JSON兼容值
            encrypt: 是否加密存储，True则在PostgreSQL中加密存储

        说明：Redis中始终不加密存储，加密选项只影响PostgreSQL存储
        """
        # Not encrypted in Redis, but encrypted in Postgres
        try:
            self.redis_client.set(
                REDIS_KEY_PREFIX + key, json.dumps(val), ex=KV_REDIS_KEY_EXPIRATION
            )
        except Exception as e:
            # Fallback gracefully to Postgres if Redis fails
            logger.error(f"Failed to set value in Redis for key '{key}': {str(e)}")

        encrypted_val = val if encrypt else None
        plain_val = val if not encrypt else None
        with self.get_session() as session:
            obj = session.query(KVStore).filter_by(key=key).first()
            if obj:
                obj.value = plain_val
                obj.encrypted_value = encrypted_val
            else:
                obj = KVStore(
                    key=key, value=plain_val, encrypted_value=encrypted_val
                )  # type: ignore
                session.query(KVStore).filter_by(key=key).delete()  # just in case
                session.add(obj)
            session.commit()

    def load(self, key: str) -> JSON_ro:
        """
        加载指定键的值。

        首先尝试从Redis加载，如果失败或未命中则从PostgreSQL加载。
        成功从PostgreSQL加载后会尝试更新Redis缓存。

        参数:
            key: 要加载的键名

        返回:
            JSON兼容的值

        异常:
            KvKeyNotFoundError: 当键不存在时抛出
        """
        try:
            redis_value = self.redis_client.get(REDIS_KEY_PREFIX + key)
            if redis_value:
                assert isinstance(redis_value, bytes)
                return json.loads(redis_value.decode("utf-8"))
        except Exception as e:
            logger.error(f"Failed to get value from Redis for key '{key}': {str(e)}")

        with self.get_session() as session:
            obj = session.query(KVStore).filter_by(key=key).first()
            if not obj:
                raise KvKeyNotFoundError

            if obj.value is not None:
                value = obj.value
            elif obj.encrypted_value is not None:
                value = obj.encrypted_value
            else:
                value = None

            try:
                self.redis_client.set(REDIS_KEY_PREFIX + key, json.dumps(value))
            except Exception as e:
                logger.error(f"Failed to set value in Redis for key '{key}': {str(e)}")

            return cast(JSON_ro, value)

    def delete(self, key: str) -> None:
        """
        删除指定的键值对。

        同时从Redis和PostgreSQL中删除数据。
        Redis删除失败会记录错误但继续处理PostgreSQL的删除。

        参数:
            key: 要删除的键名

        异常:
            KvKeyNotFoundError: 当PostgreSQL中键不存在时抛出
        """
        try:
            self.redis_client.delete(REDIS_KEY_PREFIX + key)
        except Exception as e:
            logger.error(f"Failed to delete value from Redis for key '{key}': {str(e)}")

        with self.get_session() as session:
            result = session.query(KVStore).filter_by(key=key).delete()  # type: ignore
            if result == 0:
                raise KvKeyNotFoundError
            session.commit()
