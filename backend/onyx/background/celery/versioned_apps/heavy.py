"""Factory stub for running celery worker / celery beat.
This code is different from the primary/beat stubs because there is no EE version to
fetch. Port over the code in those files if we add an EE version of this worker.

[中文]
Celery worker / celery beat 运行的工厂存根。
这段代码与 primary/beat 存根不同，因为没有需要获取的 EE 版本。
如果我们添加了此 worker 的 EE 版本，则需要将这些文件中的代码移植过来。
"""

# 导入所需的模块
from celery import Celery
from onyx.utils.variable_functionality import set_is_ee_based_on_env_variable

# 根据环境变量设置 EE 状态
set_is_ee_based_on_env_variable()


def get_app() -> Celery:
    """
    获取 Celery 应用实例
    
    Returns:
        Celery: 返回初始化好的 Celery 应用实例
    """
    from onyx.background.celery.apps.heavy import celery_app
    return celery_app


# 初始化 Celery 应用
app = get_app()
