"""
此模块定义了文档访问控制相关的数据模型类。
主要包含了外部访问控制、文档外部访问包装器以及文档访问控制类的实现。
用于管理文档的访问权限、用户组权限等功能。
"""

from dataclasses import dataclass

from onyx.access.utils import prefix_external_group
from onyx.access.utils import prefix_user_email
from onyx.access.utils import prefix_user_group
from onyx.configs.constants import PUBLIC_DOC_PAT


@dataclass(frozen=True)
class ExternalAccess:
    """
    外部访问控制类，用于定义文档的外部访问权限
    
    属性:
        external_user_emails: 有权访问文档的外部用户邮箱集合
        external_user_group_ids: 有权访问文档的外部用户组ID集合
        is_public: 文档在外部系统或Onyx中是否公开
    """
    # Emails of external users with access to the doc externally
    external_user_emails: set[str]
    # Names or external IDs of groups with access to the doc
    external_user_group_ids: set[str]
    # Whether the document is public in the external system or Onyx
    is_public: bool


@dataclass(frozen=True)
class DocExternalAccess:
    """
    文档外部访问包装类，用于将外部访问权限与文档ID关联
    This is just a class to wrap the external access and the document ID
    together. It's used for syncing document permissions to Redis.
    这只是一个将外部访问和文档ID包装在一起的类。用于同步文档权限到Redis。
    
    属性:
        external_access: 外部访问控制对象
        doc_id: 文档ID
    """

    external_access: ExternalAccess
    # The document ID
    doc_id: str

    def to_dict(self) -> dict:
        """
        将对象转换为字典格式
        
        返回值:
            dict: 包含外部访问权限和文档ID的字典
        """
        return {
            "external_access": {
                "external_user_emails": list(self.external_access.external_user_emails),
                "external_user_group_ids": list(
                    self.external_access.external_user_group_ids
                ),
                "is_public": self.external_access.is_public,
            },
            "doc_id": self.doc_id,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DocExternalAccess":
        """
        从字典创建DocExternalAccess对象
        
        参数:
            data: 包含外部访问权限和文档ID的字典
        
        返回值:
            DocExternalAccess: 新创建的对象
        """
        external_access = ExternalAccess(
            external_user_emails=set(
                data["external_access"].get("external_user_emails", [])
            ),
            external_user_group_ids=set(
                data["external_access"].get("external_user_group_ids", [])
            ),
            is_public=data["external_access"]["is_public"],
        )
        return cls(
            external_access=external_access,
            doc_id=data["doc_id"],
        )


@dataclass(frozen=True)
class DocumentAccess(ExternalAccess):
    """
    文档访问控制类，继承自ExternalAccess，用于管理文档的所有访问权限
    
    属性:
        user_emails: Onyx用户邮箱集合，None表示管理员
        user_groups: 与文档关联的用户组名称集合
        继承自ExternalAccess的所有属性
    """
    # User emails for Onyx users, None indicates admin
    user_emails: set[str | None]
    # Names of user groups associated with this document
    user_groups: set[str]

    def to_acl(self) -> set[str]:
        """
        生成访问控制列表
        
        返回值:
            set[str]: 包含所有授权用户和组的访问控制字符串集合
        """
        return set(
            [
                prefix_user_email(user_email)
                for user_email in self.user_emails
                if user_email
            ]
            + [prefix_user_group(group_name) for group_name in self.user_groups]
            + [
                prefix_user_email(user_email)
                for user_email in self.external_user_emails
            ]
            + [
                # The group names are already prefixed by the source type
                # This adds an additional prefix of "external_group:"
                prefix_external_group(group_name)
                for group_name in self.external_user_group_ids
            ]
            + ([PUBLIC_DOC_PAT] if self.is_public else [])
        )

    @classmethod
    def build(
        cls,
        user_emails: list[str | None],
        user_groups: list[str],
        external_user_emails: list[str],
        external_user_group_ids: list[str],
        is_public: bool,
    ) -> "DocumentAccess":
        """
        构建DocumentAccess对象的工厂方法
        
        参数:
            user_emails: 用户邮箱列表
            user_groups: 用户组列表
            external_user_emails: 外部用户邮箱列表
            external_user_group_ids: 外部用户组ID列表
            is_public: 是否公开
            
        返回值:
            DocumentAccess: 新创建的文档访问控制对象
        """
        return cls(
            external_user_emails={
                prefix_user_email(external_email)
                for external_email in external_user_emails
            },
            external_user_group_ids={
                prefix_external_group(external_group_id)
                for external_group_id in external_user_group_ids
            },
            user_emails={
                prefix_user_email(user_email)
                for user_email in user_emails
                if user_email
            },
            user_groups=set(user_groups),
            is_public=is_public,
        )


"""
默认的公开访问权限对象，允许所有人访问
"""
default_public_access = DocumentAccess(
    external_user_emails=set(),
    external_user_group_ids=set(),
    user_emails=set(),
    user_groups=set(),
    is_public=True,
)
