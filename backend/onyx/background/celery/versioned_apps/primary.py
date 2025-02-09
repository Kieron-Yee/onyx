"""Factory stub for running celery worker / celery beat.
用于运行 celery worker / celery beat 的工厂存根。
"""

# 导入必要的模块
from celery import Celery

from onyx.utils.variable_functionality import fetch_versioned_implementation
from onyx.utils.variable_functionality import set_is_ee_based_on_env_variable

# 根据环境变量设置是否为 ee 版本
set_is_ee_based_on_env_variable()

# 获取版本化的 celery 应用实例
# app 变量用于存储 Celery 应用实例，通过 fetch_versioned_implementation 函数获取特定版本的实现
app: Celery = fetch_versioned_implementation(
    "onyx.background.celery.apps.primary", "celery_app"
)
