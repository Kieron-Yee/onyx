"""
This file contains the Pydantic models used for LLM (Large Language Model) provider management.
此文件包含用于LLM（大型语言模型）提供商管理的Pydantic模型。

主要功能：
1. 定义LLM提供商相关的数据模型
2. 提供模型之间的转换方法
3. 处理LLM提供商配置的验证和序列化
"""

from typing import TYPE_CHECKING

from pydantic import BaseModel
from pydantic import Field

from onyx.llm.llm_provider_options import fetch_models_for_provider


if TYPE_CHECKING:
    from onyx.db.models import LLMProvider as LLMProviderModel


class TestLLMRequest(BaseModel):
    """
    用于测试LLM提供商连接的请求模型
    
    属性说明：
    - provider: LLM提供商名称
    - api_key: API密钥
    - api_base: API基础URL
    - api_version: API版本
    - custom_config: 自定义配置
    - default_model_name: 默认模型名称
    - fast_default_model_name: 快速默认模型名称
    - deployment_name: 部署名称
    """
    # provider level
    provider: str
    api_key: str | None = None
    api_base: str | None = None
    api_version: str | None = None
    custom_config: dict[str, str] | None = None

    # model level
    default_model_name: str
    fast_default_model_name: str | None = None
    deployment_name: str | None = None


class LLMProviderDescriptor(BaseModel):
    """
    A descriptor for an LLM provider that can be safely viewed by non-admin users. 
    Used when giving a list of available LLMs.
    一个可以被非管理员用户安全查看的LLM提供商描述符。用于提供可用LLM列表时使用。
    """

    name: str
    provider: str
    model_names: list[str]
    default_model_name: str
    fast_default_model_name: str | None
    is_default_provider: bool | None
    display_model_names: list[str] | None

    @classmethod
    def from_model(
        cls, llm_provider_model: "LLMProviderModel"
    ) -> "LLMProviderDescriptor":
        """
        从数据库模型创建LLM提供商描述符
        
        参数:
            llm_provider_model: LLM提供商数据库模型实例
        
        返回:
            LLMProviderDescriptor实例
        """
        return cls(
            name=llm_provider_model.name,
            provider=llm_provider_model.provider,
            default_model_name=llm_provider_model.default_model_name,
            fast_default_model_name=llm_provider_model.fast_default_model_name,
            is_default_provider=llm_provider_model.is_default_provider,
            model_names=(
                llm_provider_model.model_names
                or fetch_models_for_provider(llm_provider_model.provider)
                or [llm_provider_model.default_model_name]
            ),
            display_model_names=llm_provider_model.display_model_names,
        )


class LLMProvider(BaseModel):
    """
    LLM提供商的基本模型
    
    属性说明：
    - name: 提供商名称
    - provider: 提供商类型
    - api_key: API密钥
    - api_base: API基础URL
    - api_version: API版本
    - custom_config: 自定义配置
    - default_model_name: 默认模型名称
    - fast_default_model_name: 快速默认模型名称
    - is_public: 是否公开
    - groups: 可访问的用户组ID列表
    - display_model_names: 显示的模型名称列表
    - deployment_name: 部署名称
    """
    name: str
    provider: str
    api_key: str | None = None
    api_base: str | None = None
    api_version: str | None = None
    custom_config: dict[str, str] | None = None
    default_model_name: str
    fast_default_model_name: str | None = None
    is_public: bool = True
    groups: list[int] = Field(default_factory=list)
    display_model_names: list[str] | None = None
    deployment_name: str | None = None


class LLMProviderUpsertRequest(LLMProvider):
    """
    用于创建或更新LLM提供商的请求模型
    
    注意：model_names字段仅用于"自定义"提供商
    对于默认提供商，使用内置的模型名称列表
    """
    # should only be used for a "custom" provider
    # for default providers, the built-in model names are used
    model_names: list[str] | None = None


class FullLLMProvider(LLMProvider):
    """
    完整的LLM提供商模型，包含所有字段
    
    额外属性：
    - id: 提供商ID
    - is_default_provider: 是否为默认提供商
    - model_names: 支持的模型名称列表
    """
    id: int
    is_default_provider: bool | None = None
    model_names: list[str]

    @classmethod
    def from_model(cls, llm_provider_model: "LLMProviderModel") -> "FullLLMProvider":
        """
        从数据库模型创建完整的LLM提供商模型
        
        参数:
            llm_provider_model: LLM提供商数据库模型实例
        
        返回:
            FullLLMProvider实例
        """
        return cls(
            id=llm_provider_model.id,
            name=llm_provider_model.name,
            provider=llm_provider_model.provider,
            api_key=llm_provider_model.api_key,
            api_base=llm_provider_model.api_base,
            api_version=llm_provider_model.api_version,
            custom_config=llm_provider_model.custom_config,
            default_model_name=llm_provider_model.default_model_name,
            fast_default_model_name=llm_provider_model.fast_default_model_name,
            is_default_provider=llm_provider_model.is_default_provider,
            display_model_names=llm_provider_model.display_model_names,
            model_names=(
                llm_provider_model.model_names
                or fetch_models_for_provider(llm_provider_model.provider)
                or [llm_provider_model.default_model_name]
            ),
            is_public=llm_provider_model.is_public,
            groups=[group.id for group in llm_provider_model.groups],
            deployment_name=llm_provider_model.deployment_name,
        )
