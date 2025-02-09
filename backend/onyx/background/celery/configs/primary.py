"""
此文件为Celery主要配置文件
用于设置Celery的broker、backend、worker等核心配置项
继承自base配置文件中的共享配置
"""

import onyx.background.celery.configs.base as shared_config

# Broker配置 / Broker Configuration
# 消息代理服务器的连接URL
broker_url = shared_config.broker_url
# 启动时如果连接失败是否重试
broker_connection_retry_on_startup = shared_config.broker_connection_retry_on_startup
# broker连接池的最大连接数
broker_pool_limit = shared_config.broker_pool_limit
# broker传输选项设置
broker_transport_options = shared_config.broker_transport_options

# Redis配置 / Redis Configuration
# 是否保持Redis socket连接
redis_socket_keepalive = shared_config.redis_socket_keepalive
# Redis超时时是否重试
redis_retry_on_timeout = shared_config.redis_retry_on_timeout
# Redis后端健康检查间隔时间
redis_backend_health_check_interval = shared_config.redis_backend_health_check_interval

# 结果后端配置 / Result Backend Configuration
# 结果存储后端的连接URL
result_backend = shared_config.result_backend
# 结果的过期时间（默认86400秒）
result_expires = shared_config.result_expires

# 任务配置 / Task Configuration
# 任务的默认优先级
task_default_priority = shared_config.task_default_priority
# 任务完成后再确认（late acknowledgment）
task_acks_late = shared_config.task_acks_late

# Worker配置 / Worker Configuration
# worker的并发数
worker_concurrency = 4
# worker使用的线程池类型
worker_pool = "threads"
# worker预取任务的倍数
worker_prefetch_multiplier = 1
