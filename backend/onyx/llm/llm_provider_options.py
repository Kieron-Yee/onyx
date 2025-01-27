"""
此模块用于定义和管理LLM（大型语言模型）提供商的配置选项和描述信息。
包含了对OpenAI、Anthropic、Azure OpenAI和AWS Bedrock等主要LLM提供商的支持。
"""

import litellm  # type: ignore
from pydantic import BaseModel


class CustomConfigKey(BaseModel):
    """
    自定义配置键的模型类，用于定义LLM提供商的配置参数。
    
    属性：
        name: 配置键名称
        description: 配置说明
        is_required: 是否必需
        is_secret: 是否为敏感信息
    """
    name: str
    description: str | None = None
    is_required: bool = True
    is_secret: bool = False


class WellKnownLLMProviderDescriptor(BaseModel):
    """
    LLM提供商描述符类，用于描述LLM提供商的基本信息和配置要求。
    
    属性：
        name: 提供商名称
        display_name: 显示名称
        api_key_required: 是否需要API密钥
        api_base_required: 是否需要API基础URL
        api_version_required: 是否需要API版本
        custom_config_keys: 自定义配置键列表
        llm_names: 支持的模型名称列表
        default_model: 默认模型
        default_fast_model: 默认快速模型
        deployment_name_required: 是否需要部署名称
        single_model_supported: 是否支持单一模型部署
    """
    name: str
    display_name: str
    api_key_required: bool
    api_base_required: bool
    api_version_required: bool
    custom_config_keys: list[CustomConfigKey] | None = None
    llm_names: list[str]
    default_model: str | None = None
    default_fast_model: str | None = None
    deployment_name_required: bool = False
    single_model_supported: bool = False


# 定义各提供商的常量名称
OPENAI_PROVIDER_NAME = "openai"
OPEN_AI_MODEL_NAMES = [
    "o1-mini",
    "o1-preview",
    "o1-2024-12-17",
    "gpt-4",
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4-turbo",
    "gpt-4-turbo-preview",
    "gpt-4-1106-preview",
    "gpt-4-vision-preview",
    "gpt-4-0613",
    "gpt-4o-2024-08-06",
    "gpt-4-0314",
    "gpt-4-32k-0314",
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-0125",
    "gpt-3.5-turbo-1106",
    "gpt-3.5-turbo-16k",
    "gpt-3.5-turbo-0613",
    "gpt-3.5-turbo-16k-0613",
    "gpt-3.5-turbo-0301",
]

BEDROCK_PROVIDER_NAME = "bedrock"
# 需要移除所有带有奇怪格式的模型名称，如 "bedrock/eu-central-1/anthropic.claude-v1"
BEDROCK_MODEL_NAMES = [
    model
    for model in litellm.bedrock_models
    if "/" not in model and "embed" not in model
][::-1]

IGNORABLE_ANTHROPIC_MODELS = [
    "claude-2",
    "claude-instant-1",
    "anthropic/claude-3-5-sonnet-20241022",
]
ANTHROPIC_PROVIDER_NAME = "anthropic"
ANTHROPIC_MODEL_NAMES = [
    model
    for model in litellm.anthropic_models
    if model not in IGNORABLE_ANTHROPIC_MODELS
][::-1]

AZURE_PROVIDER_NAME = "azure"


_PROVIDER_TO_MODELS_MAP = {
    OPENAI_PROVIDER_NAME: OPEN_AI_MODEL_NAMES,
    BEDROCK_PROVIDER_NAME: BEDROCK_MODEL_NAMES,
    ANTHROPIC_PROVIDER_NAME: ANTHROPIC_MODEL_NAMES,
}


def fetch_available_well_known_llms() -> list[WellKnownLLMProviderDescriptor]:
    """
    获取所有可用的已知LLM提供商的描述符列表。
    
    返回值：
        list[WellKnownLLMProviderDescriptor]: 包含所有已配置的LLM提供商描述符的列表
    """
    return [
        WellKnownLLMProviderDescriptor(
            name="openai",
            display_name="OpenAI",
            api_key_required=True,
            api_base_required=False,
            api_version_required=False,
            custom_config_keys=[],
            llm_names=fetch_models_for_provider(OPENAI_PROVIDER_NAME),
            default_model="gpt-4",
            default_fast_model="gpt-4o-mini",
        ),
        WellKnownLLMProviderDescriptor(
            name=ANTHROPIC_PROVIDER_NAME,
            display_name="Anthropic",
            api_key_required=True,
            api_base_required=False,
            api_version_required=False,
            custom_config_keys=[],
            llm_names=fetch_models_for_provider(ANTHROPIC_PROVIDER_NAME),
            default_model="claude-3-5-sonnet-20241022",
            default_fast_model="claude-3-5-sonnet-20241022",
        ),
        WellKnownLLMProviderDescriptor(
            name=AZURE_PROVIDER_NAME,
            display_name="Azure OpenAI",
            api_key_required=True,
            api_base_required=True,
            api_version_required=True,
            custom_config_keys=[],
            llm_names=fetch_models_for_provider(AZURE_PROVIDER_NAME),
            deployment_name_required=True,
            single_model_supported=True,
        ),
        WellKnownLLMProviderDescriptor(
            name=BEDROCK_PROVIDER_NAME,
            display_name="AWS Bedrock",
            api_key_required=False,
            api_base_required=False,
            api_version_required=False,
            custom_config_keys=[
                CustomConfigKey(name="AWS_REGION_NAME"),
                CustomConfigKey(
                    name="AWS_ACCESS_KEY_ID",
                    is_required=False,
                    description="If using AWS IAM roles, AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY can be left blank.",
                ),
                CustomConfigKey(
                    name="AWS_SECRET_ACCESS_KEY",
                    is_required=False,
                    is_secret=True,
                    description="If using AWS IAM roles, AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY can be left blank.",
                ),
            ],
            llm_names=fetch_models_for_provider(BEDROCK_PROVIDER_NAME),
            default_model="anthropic.claude-3-5-sonnet-20241022-v2:0",
            default_fast_model="anthropic.claude-3-5-sonnet-20241022-v2:0",
        ),
    ]


def fetch_models_for_provider(provider_name: str) -> list[str]:
    """
    获取指定提供商支持的模型列表。
    
    参数：
        provider_name: 提供商名称
        
    返回值：
        list[str]: 该提供商支持的模型名称列表，如果提供商不存在则返回空列表
    """
    return _PROVIDER_TO_MODELS_MAP.get(provider_name, [])
