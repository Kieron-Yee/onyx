"""Factory stub for running celery worker / celery beat.
用于运行celery worker和celery beat的工厂存根。
"""

# 导入Celery类，用于创建Celery应用实例
from celery import Celery

# 从beat模块导入celery应用实例
from onyx.background.celery.apps.beat import celery_app
# 导入环境变量设置工具函数
from onyx.utils.variable_functionality import set_is_ee_based_on_env_variable

# 根据环境变量设置EE标志
set_is_ee_based_on_env_variable()
# 声明app变量为Celery类型，并赋值为celery_app实例
app: Celery = celery_app
