"""
Celery 重任务配置文件
本文件包含适用于处理计算密集型或重量级任务的 Celery 配置。
主要特点是使用线程池和较低的并发数，适合处理需要较长时间的任务。
"""

import onyx.background.celery.configs.base as shared_config

# Broker（消息代理）相关配置
broker_url = shared_config.broker_url  # 消息代理的连接URL
broker_connection_retry_on_startup = shared_config.broker_connection_retry_on_startup  # 启动时是否重试连接
broker_pool_limit = shared_config.broker_pool_limit  # 代理连接池限制
broker_transport_options = shared_config.broker_transport_options  # 代理传输选项

# Redis 连接相关配置
redis_socket_keepalive = shared_config.redis_socket_keepalive  # Redis socket保持活动状态
redis_retry_on_timeout = shared_config.redis_retry_on_timeout  # Redis超时重试设置
redis_backend_health_check_interval = shared_config.redis_backend_health_check_interval  # Redis后端健康检查间隔

# 结果后端配置
result_backend = shared_config.result_backend  # 结果存储后端
result_expires = shared_config.result_expires  # 结果过期时间，默认86400秒（24小时）

# 任务相关配置
task_default_priority = shared_config.task_default_priority  # 任务默认优先级
task_acks_late = shared_config.task_acks_late  # 任务延迟确认设置

# 工作进程配置
worker_concurrency = 4  # 工作进程并发数，设置为4以处理重任务
worker_pool = "threads"  # 使用线程池作为工作池类型
worker_prefetch_multiplier = 1  # 预取任务数乘数，设为1以避免过度预取
