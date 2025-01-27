"""
此文件包含了与嵌入(Embedding)服务相关的数据模型定义。
主要包括搜索设置、嵌入测试请求、云端嵌入提供者等数据模型类。
这些模型用于验证和处理与嵌入服务相关的请求和响应数据。
"""

from typing import TYPE_CHECKING

from pydantic import BaseModel

from shared_configs.enums import EmbeddingProvider

if TYPE_CHECKING:
    from onyx.db.models import CloudEmbeddingProvider as CloudEmbeddingProviderModel


class SearchSettingsDeleteRequest(BaseModel):
    """
    搜索设置删除请求的数据模型
    
    属性:
        search_settings_id (int): 要删除的搜索设置的唯一标识符
    """
    search_settings_id: int


class TestEmbeddingRequest(BaseModel):
    """
    嵌入测试请求的数据模型
    
    属性:
        provider_type (EmbeddingProvider): 嵌入服务提供者类型
        api_key (str | None): API密钥，可选
        api_url (str | None): API地址，可选
        model_name (str | None): 模型名称，可选
        api_version (str | None): API版本，可选
        deployment_name (str | None): 部署名称，可选
    """
    provider_type: EmbeddingProvider
    api_key: str | None = None
    api_url: str | None = None
    model_name: str | None = None
    api_version: str | None = None
    deployment_name: str | None = None

    # This disables the "model_" protected namespace for pydantic
    # 这里禁用了pydantic的"model_"保护命名空间
    model_config = {"protected_namespaces": ()}


class CloudEmbeddingProvider(BaseModel):
    """
    云端嵌入服务提供者的数据模型
    
    属性:
        provider_type (EmbeddingProvider): 嵌入服务提供者类型
        api_key (str | None): API密钥，可选
        api_url (str | None): API地址，可选
        api_version (str | None): API版本，可选
        deployment_name (str | None): 部署名称，可选
    """
    provider_type: EmbeddingProvider
    api_key: str | None = None
    api_url: str | None = None
    api_version: str | None = None
    deployment_name: str | None = None

    @classmethod
    def from_request(
        cls, cloud_provider_model: "CloudEmbeddingProviderModel"
    ) -> "CloudEmbeddingProvider":
        """
        从数据库模型创建云端嵌入提供者实例
        
        参数:
            cloud_provider_model (CloudEmbeddingProviderModel): 数据库中的云端嵌入提供者模型实例
        
        返回:
            CloudEmbeddingProvider: 新创建的云端嵌入提供者实例
        """
        return cls(
            provider_type=cloud_provider_model.provider_type,
            api_key=cloud_provider_model.api_key,
            api_url=cloud_provider_model.api_url,
            api_version=cloud_provider_model.api_version,
            deployment_name=cloud_provider_model.deployment_name,
        )


class CloudEmbeddingProviderCreationRequest(BaseModel):
    """
    创建云端嵌入提供者的请求数据模型
    
    属性:
        provider_type (EmbeddingProvider): 嵌入服务提供者类型
        api_key (str | None): API密钥，可选
        api_url (str | None): API地址，可选
        api_version (str | None): API版本，可选
        deployment_name (str | None): 部署名称，可选
    """
    provider_type: EmbeddingProvider
    api_key: str | None = None
    api_url: str | None = None
    api_version: str | None = None
    deployment_name: str | None = None
