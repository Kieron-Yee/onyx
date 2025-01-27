"""
此文件定义了与文档集(Document Set)相关的数据模型和请求/响应类。
包含了文档集的创建、更新、查询等功能所需的数据结构。
"""

from uuid import UUID

from pydantic import BaseModel
from pydantic import Field

from onyx.db.models import DocumentSet as DocumentSetDBModel
from onyx.server.documents.models import ConnectorCredentialPairDescriptor
from onyx.server.documents.models import ConnectorSnapshot
from onyx.server.documents.models import CredentialSnapshot


class DocumentSetCreationRequest(BaseModel):
    """
    文档集创建请求的数据模型
    
    属性说明：
        name: 文档集名称
        description: 文档集描述
        cc_pair_ids: 连接器凭证对ID列表
        is_public: 是否为公开文档集
        users: 私有文档集的授权用户ID列表
        groups: 私有文档集的授权用户组ID列表
    """
    name: str
    description: str
    cc_pair_ids: list[int]
    is_public: bool
    # For Private Document Sets, who should be able to access these
    users: list[UUID] = Field(default_factory=list)
    groups: list[int] = Field(default_factory=list)


class DocumentSetUpdateRequest(BaseModel):
    """
    文档集更新请求的数据模型
    
    属性说明：
        id: 文档集ID
        description: 文档集描述
        cc_pair_ids: 连接器凭证对ID列表
        is_public: 是否为公开文档集
        users: 私有文档集的授权用户ID列表
        groups: 私有文档集的授权用户组ID列表
    """
    id: int
    description: str
    cc_pair_ids: list[int]
    is_public: bool
    # For Private Document Sets, who should be able to access these
    users: list[UUID]
    groups: list[int]


class CheckDocSetPublicRequest(BaseModel):
    """Note that this does not mean that the Document Set itself is to be viewable by everyone
    Rather, this refers to the CC-Pairs in the Document Set, and if every CC-Pair is public
    注意：这并不意味着文档集本身可以被所有人查看，而是指文档集中的CC-Pairs是否都是公开的
    """

    document_set_ids: list[int]


class CheckDocSetPublicResponse(BaseModel):
    """
    检查文档集公开状态的响应模型
    
    属性说明：
        is_public: 是否所有CC-Pairs都是公开的
    """
    is_public: bool


class DocumentSet(BaseModel):
    """
    文档集的数据模型
    
    属性说明：
        id: 文档集ID
        name: 文档集名称
        description: 文档集描述
        cc_pair_descriptors: 连接器凭证对描述符列表
        is_up_to_date: 文档集是否是最新的
        is_public: 是否为公开文档集
        users: 私有文档集的授权用户ID列表
        groups: 私有文档集的授权用户组ID列表
    """
    id: int
    name: str
    description: str | None
    cc_pair_descriptors: list[ConnectorCredentialPairDescriptor]
    is_up_to_date: bool
    is_public: bool
    # For Private Document Sets, who should be able to access these
    users: list[UUID]
    groups: list[int]

    @classmethod
    def from_model(cls, document_set_model: DocumentSetDBModel) -> "DocumentSet":
        """
        从数据库模型创建文档集对象
        
        参数：
            document_set_model: 文档集数据库模型实例
            
        返回值：
            DocumentSet: 转换后的文档集对象
        """
        return cls(
            id=document_set_model.id,
            name=document_set_model.name,
            description=document_set_model.description,
            cc_pair_descriptors=[
                ConnectorCredentialPairDescriptor(
                    id=cc_pair.id,
                    name=cc_pair.name,
                    connector=ConnectorSnapshot.from_connector_db_model(
                        cc_pair.connector
                    ),
                    credential=CredentialSnapshot.from_credential_db_model(
                        cc_pair.credential
                    ),
                )
                for cc_pair in document_set_model.connector_credential_pairs
            ],
            is_up_to_date=document_set_model.is_up_to_date,
            is_public=document_set_model.is_public,
            users=[user.id for user in document_set_model.users],
            groups=[group.id for group in document_set_model.groups],
        )
