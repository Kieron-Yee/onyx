"""Factory stub for running celery worker / celery beat.
This code is different from the primary/beat stubs because there is no EE version to
fetch. Port over the code in those files if we add an EE version of this worker.

用于运行 celery worker / celery beat 的工厂存根。
这段代码与 primary/beat 存根不同，因为这里没有 EE 版本需要获取。
如果我们添加了这个 worker 的 EE 版本，需要将那些文件中的代码移植过来。
"""

from celery import Celery
from onyx.utils.variable_functionality import set_is_ee_based_on_env_variable

# 根据环境变量设置是否为 EE 版本
set_is_ee_based_on_env_variable()


def get_app() -> Celery:
    """
    获取 Celery 应用实例
    
    该函数从 onyx.background.celery.apps.light 模块导入并返回 celery_app
    
    Returns:
        Celery: 返回配置好的 Celery 应用实例
    """
    from onyx.background.celery.apps.light import celery_app
    return celery_app


# 获取并初始化 Celery 应用实例
app = get_app()
