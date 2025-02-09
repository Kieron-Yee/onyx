"""
此文件包含 Celery 轻量级工作进程的配置参数。
主要用于设置轻量级任务的执行环境，包括代理连接、Redis设置、任务处理等相关配置。
"""

import onyx.background.celery.configs.base as shared_config
from onyx.configs.app_configs import CELERY_WORKER_LIGHT_CONCURRENCY
from onyx.configs.app_configs import CELERY_WORKER_LIGHT_PREFETCH_MULTIPLIER

# 代理服务器URL配置
broker_url = shared_config.broker_url

# 启动时是否重试代理连接
broker_connection_retry_on_startup = shared_config.broker_connection_retry_on_startup

# 代理连接池限制
broker_pool_limit = shared_config.broker_pool_limit

# 代理传输选项
broker_transport_options = shared_config.broker_transport_options

# Redis保持连接配置
redis_socket_keepalive = shared_config.redis_socket_keepalive

# Redis超时重试配置
redis_retry_on_timeout = shared_config.redis_retry_on_timeout

# Redis后端健康检查间隔
redis_backend_health_check_interval = shared_config.redis_backend_health_check_interval

# 结果存储后端配置
result_backend = shared_config.result_backend

# 结果过期时间（默认86400秒）
result_expires = shared_config.result_expires

# 任务默认优先级
task_default_priority = shared_config.task_default_priority

# 是否延迟确认任务
task_acks_late = shared_config.task_acks_late

# 工作进程并发数
worker_concurrency = CELERY_WORKER_LIGHT_CONCURRENCY

# 工作进程池类型
worker_pool = "threads"

# 工作进程预取乘数
worker_prefetch_multiplier = CELERY_WORKER_LIGHT_PREFETCH_MULTIPLIER
