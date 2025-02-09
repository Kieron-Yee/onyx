"""
LLM模型更新任务模块

该模块负责检查和更新LLM（语言模型）的可用模型列表。
主要功能包括：
1. 从API获取最新的模型列表
2. 更新数据库中默认提供商的模型配置
3. 处理模型列表的响应数据
"""

from typing import Any
import requests
from celery import shared_task
from celery import Task

from onyx.background.celery.apps.app_base import task_logger
from onyx.configs.app_configs import JOB_TIMEOUT
from onyx.configs.app_configs import LLM_MODEL_UPDATE_API_URL
from onyx.configs.constants import OnyxCeleryTask
from onyx.db.engine import get_session_with_tenant
from onyx.db.models import LLMProvider


def _process_model_list_response(model_list_json: Any) -> list[str]:
    """
    处理模型列表API响应数据

    Args:
        model_list_json (Any): API返回的JSON响应数据

    Returns:
        list[str]: 处理后的模型名称列表

    Raises:
        ValueError: 当响应数据格式不正确时抛出
    """
    # Handle case where response is wrapped in a "data" field
    # 处理响应被包装在"data"字段中的情况
    if isinstance(model_list_json, dict) and "data" in model_list_json:
        model_list_json = model_list_json["data"]

    if not isinstance(model_list_json, list):
        raise ValueError(
            f"Invalid response from API - expected list, got {type(model_list_json)}"
            f"无效的API响应 - 期望列表类型，实际获得 {type(model_list_json)}"
        )

    # Handle both string list and object list cases
    # 同时处理字符串列表和对象列表的情况
    model_names: list[str] = []
    for item in model_list_json:
        if isinstance(item, str):
            model_names.append(item)
        elif isinstance(item, dict) and "model_name" in item:
            model_names.append(item["model_name"])
        else:
            raise ValueError(
                f"Invalid item in model list - expected string or dict with model_name, got {type(item)}"
                f"模型列表中的项目无效 - 期望字符串或包含model_name的字典，实际获得 {type(item)}"
            )

    return model_names


@shared_task(
    name=OnyxCeleryTask.CHECK_FOR_LLM_MODEL_UPDATE,
    soft_time_limit=JOB_TIMEOUT,
    trail=False,
    bind=True,
)
def check_for_llm_model_update(self: Task, *, tenant_id: str | None) -> bool | None:
    """
    检查并更新LLM模型列表的Celery任务

    Args:
        self (Task): Celery任务实例
        tenant_id (str | None): 租户ID

    Returns:
        bool | None: 更新成功返回True，失败返回None

    Raises:
        ValueError: 当LLM模型更新API URL未配置时抛出
    """
    if not LLM_MODEL_UPDATE_API_URL:
        raise ValueError("LLM model update API URL not configured"
                       "LLM模型更新API URL未配置")

    # First fetch the models from the API
    # 首先从API获取模型列表
    try:
        response = requests.get(LLM_MODEL_UPDATE_API_URL)
        response.raise_for_status()
        available_models = _process_model_list_response(response.json())
        task_logger.info(f"Found available models: {available_models}"
                        f"发现可用模型：{available_models}")

    except Exception:
        task_logger.exception("Failed to fetch models from API."
                            "从API获取模型失败。")
        return None

    # Then update the database with the fetched models
    # 然后使用获取的模型更新数据库
    with get_session_with_tenant(tenant_id) as db_session:
        # Get the default LLM provider
        # 获取默认LLM提供商
        default_provider = (
            db_session.query(LLMProvider)
            .filter(LLMProvider.is_default_provider.is_(True))
            .first()
        )

        if not default_provider:
            task_logger.warning("No default LLM provider found"
                              "未找到默认LLM提供商")
            return None

        # log change if any
        # 记录变更（如果有）
        old_models = set(default_provider.model_names or [])
        new_models = set(available_models)
        added_models = new_models - old_models
        removed_models = old_models - new_models

        if added_models:
            task_logger.info(f"Adding models: {sorted(added_models)}"
                           f"添加模型：{sorted(added_models)}")
        if removed_models:
            task_logger.info(f"Removing models: {sorted(removed_models)}"
                           f"移除模型：{sorted(removed_models)}")

        # Update the provider's model list
        # 更新提供商的模型列表
        default_provider.model_names = available_models
        # if the default model is no longer available, set it to the first model in the list
        # 如果默认模型不再可用，设置为列表中的第一个模型
        if default_provider.default_model_name not in available_models:
            task_logger.info(
                f"Default model {default_provider.default_model_name} not "
                f"available, setting to first model in list: {available_models[0]}"
                f"默认模型 {default_provider.default_model_name} 不可用，"
                f"设置为列表中的第一个模型：{available_models[0]}"
            )
            default_provider.default_model_name = available_models[0]
        if default_provider.fast_default_model_name not in available_models:
            task_logger.info(
                f"Fast default model {default_provider.fast_default_model_name} "
                f"not available, setting to first model in list: {available_models[0]}"
                f"快速默认模型 {default_provider.fast_default_model_name} 不可用，"
                f"设置为列表中的第一个模型：{available_models[0]}"
            )
            default_provider.fast_default_model_name = available_models[0]
        db_session.commit()

        if added_models or removed_models:
            task_logger.info("Updated model list for default provider."
                           "已更新默认提供商的模型列表。")

    return True
