"""Factory stub for running celery worker / celery beat.
This code is different from the primary/beat stubs because there is no EE version to
fetch. Port over the code in those files if we add an EE version of this worker.

用于运行celery worker / celery beat的工厂存根。
这段代码与primary/beat存根不同，因为没有EE版本需要获取。
如果我们添加了这个worker的EE版本，需要将那些文件中的代码移植过来。
"""

from celery import Celery
from onyx.utils.variable_functionality import set_is_ee_based_on_env_variable

# 根据环境变量设置是否为EE版本
set_is_ee_based_on_env_variable()


def get_app() -> Celery:
    """
    获取Celery应用实例
    
    返回:
        Celery: 返回配置好的Celery应用实例
    """
    from onyx.background.celery.apps.indexing import celery_app
    return celery_app


# 初始化Celery应用实例
app = get_app()
