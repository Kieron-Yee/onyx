"""Overrides sent over the wire / stored in the DB
通过网络传输/存储在数据库中的覆盖配置

NOTE: these models are used in many places, so have to be
kepy in a separate file to avoid circular imports.
注意：这些模型在多处被使用，因此必须保存在单独的文件中以避免循环导入。
"""

"""
文件说明：
该文件定义了LLM（大语言模型）相关的配置覆盖模型。
主要包含两个数据模型类：LLMOverride和PromptOverride，
用于处理模型配置和提示信息的覆盖设置。
"""

from pydantic import BaseModel


class LLMOverride(BaseModel):
    """
    LLM模型配置的覆盖设置类
    
    属性说明：
    model_provider: 模型提供商
    model_version: 模型版本
    temperature: 温度参数，用于控制模型输出的随机性
    """
    model_provider: str | None = None  # 模型提供商名称
    model_version: str | None = None   # 模型版本标识
    temperature: float | None = None   # 模型温度参数

    # This disables the "model_" protected namespace for pydantic
    # 这里禁用了pydantic的"model_"保护命名空间
    model_config = {"protected_namespaces": ()}


class PromptOverride(BaseModel):
    """
    提示信息的覆盖设置类
    
    属性说明：
    system_prompt: 系统提示信息
    task_prompt: 任务提示信息
    """
    system_prompt: str | None = None   # 系统级提示信息
    task_prompt: str | None = None     # 任务级提示信息
