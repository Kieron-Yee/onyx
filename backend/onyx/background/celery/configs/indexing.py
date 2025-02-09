"""
Celery indexing worker configuration file
Celery 索引工作进程配置文件

This module contains configuration settings for the Celery worker dedicated to indexing tasks.
本模块包含用于索引任务的Celery工作进程的配置设置。
"""

import onyx.background.celery.configs.base as shared_config
from onyx.configs.app_configs import CELERY_WORKER_INDEXING_CONCURRENCY

broker_url = shared_config.broker_url
broker_connection_retry_on_startup = shared_config.broker_connection_retry_on_startup
broker_pool_limit = shared_config.broker_pool_limit
broker_transport_options = shared_config.broker_transport_options

redis_socket_keepalive = shared_config.redis_socket_keepalive
redis_retry_on_timeout = shared_config.redis_retry_on_timeout
redis_backend_health_check_interval = shared_config.redis_backend_health_check_interval

result_backend = shared_config.result_backend
result_expires = shared_config.result_expires  # 86400 seconds is the default

task_default_priority = shared_config.task_default_priority
task_acks_late = shared_config.task_acks_late

# Indexing worker specific ... this lets us track the transition to STARTED in redis
# We don't currently rely on this but it has the potential to be useful and
# indexing tasks are not high volume
# 索引工作进程特定设置...这允许我们在redis中追踪任务状态变为STARTED的转换
# 我们目前并不依赖这个功能，但它可能有用，而且索引任务的数量并不大

# we don't turn this on yet because celery occasionally runs tasks more than once
# which means a duplicate run might change the task state unexpectedly
# task_track_started = True
# 我们暂时不启用此功能，因为celery偶尔会多次运行任务
# 这意味着重复运行可能会意外改变任务状态
# task_track_started = True

# 工作进程并发数设置
worker_concurrency = CELERY_WORKER_INDEXING_CONCURRENCY
# 使用线程池作为工作进程执行方式
worker_pool = "threads"
# 每个工作进程的任务预取数量
worker_prefetch_multiplier = 1
