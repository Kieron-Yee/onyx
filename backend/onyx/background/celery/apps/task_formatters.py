"""
此文件包含用于Celery任务日志格式化的自定义格式化器。
主要提供了两个格式化器类：普通文本格式化器和带颜色的格式化器，
用于在Celery任务执行时生成格式化的日志输出。
"""

import logging
from celery import current_task
from onyx.utils.logger import ColoredFormatter
from onyx.utils.logger import PlainFormatter


class CeleryTaskPlainFormatter(PlainFormatter):
    """
    Celery任务普通文本格式化器
    继承自PlainFormatter，用于生成不带颜色的纯文本格式的Celery任务日志。
    为日志记录添加任务ID和任务名称信息。
    """
    
    def format(self, record: logging.LogRecord) -> str:
        """
        格式化日志记录
        
        Args:
            record (logging.LogRecord): 需要格式化的日志记录对象
            
        Returns:
            str: 格式化后的日志字符串
            
        说明:
            在日志消息前添加任务名称和任务ID信息，
            格式为：[任务名称(任务ID)] 原始消息
        """
        task = current_task
        if task and task.request:
            record.__dict__.update(task_id=task.request.id, task_name=task.name)
            record.msg = f"[{task.name}({task.request.id})] {record.msg}"

        return super().format(record)


class CeleryTaskColoredFormatter(ColoredFormatter):
    """
    Celery任务彩色格式化器
    继承自ColoredFormatter，用于生成带颜色的Celery任务日志。
    为日志记录添加任务ID和任务名称信息，并保持颜色格式化功能。
    """
    
    def format(self, record: logging.LogRecord) -> str:
        """
        格式化日志记录
        
        Args:
            record (logging.LogRecord): 需要格式化的日志记录对象
            
        Returns:
            str: 格式化后的带颜色的日志字符串
            
        说明:
            在日志消息前添加任务名称和任务ID信息，
            格式为：[任务名称(任务ID)] 原始消息，
            同时保持父类的颜色格式化功能
        """
        task = current_task
        if task and task.request:
            record.__dict__.update(task_id=task.request.id, task_name=task.name)
            record.msg = f"[{task.name}({task.request.id})] {record.msg}"

        return super().format(record)
