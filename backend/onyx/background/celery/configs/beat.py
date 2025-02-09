"""
Celery Beat Configuration File
Celery Beat 配置文件

This file contains configuration settings for Celery Beat scheduler.
此文件包含 Celery Beat 调度器的配置设置。

For detailed configuration options, see:
详细配置选项请参见：
https://docs.celeryq.dev/en/stable/userguide/configuration.html
"""

import onyx.background.celery.configs.base as shared_config

# Broker settings 消息代理设置
broker_url = shared_config.broker_url  # 消息代理的URL
broker_connection_retry_on_startup = shared_config.broker_connection_retry_on_startup  # 启动时是否重试连接
broker_pool_limit = shared_config.broker_pool_limit  # 代理连接池限制
broker_transport_options = shared_config.broker_transport_options  # 传输选项配置

# Redis connection settings Redis连接设置
redis_socket_keepalive = shared_config.redis_socket_keepalive  # Redis保持连接配置
redis_retry_on_timeout = shared_config.redis_retry_on_timeout  # Redis超时重试配置
redis_backend_health_check_interval = shared_config.redis_backend_health_check_interval  # Redis后端健康检查间隔

# Result backend settings 结果后端设置
result_backend = shared_config.result_backend  # 结果存储后端配置
result_expires = shared_config.result_expires  # 结果过期时间配置（默认86400秒）
