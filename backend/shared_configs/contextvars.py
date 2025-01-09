import contextvars

from shared_configs.configs import POSTGRES_DEFAULT_SCHEMA

# Context variable for the current tenant id
# 当前租户ID的上下文变量
CURRENT_TENANT_ID_CONTEXTVAR = contextvars.ContextVar(
    "current_tenant_id", default=POSTGRES_DEFAULT_SCHEMA
)
