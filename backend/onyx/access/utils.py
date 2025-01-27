"""
此模块提供用户访问控制相关的工具函数。
主要包含各种前缀处理函数,用于区分和处理不同类型的用户标识(用户邮箱、用户组、外部组等)。
"""

from onyx.configs.constants import DocumentSource


def prefix_user_email(user_email: str) -> str:
    """Prefixes a user email to eliminate collision with group names.
    This applies to both a Onyx user and an External user, this is to make the query time
    more efficient
    
    为用户邮箱添加前缀,以避免与组名发生冲突。这适用于Onyx用户和外部用户,可以提高查询效率。

    参数:
        user_email (str): 用户邮箱地址
    返回:
        str: 添加了前缀的用户邮箱
    """
    return f"user_email:{user_email}"


def prefix_user_group(user_group_name: str) -> str:
    """Prefixes a user group name to eliminate collision with user emails.
    This assumes that user ids are prefixed with a different prefix.
    
    为用户组名添加前缀,以避免与用户邮箱发生冲突。前提是用户ID使用了不同的前缀。

    参数:
        user_group_name (str): 用户组名称
    返回:
        str: 添加了前缀的用户组名
    """
    return f"group:{user_group_name}"


def prefix_external_group(ext_group_name: str) -> str:
    """Prefixes an external group name to eliminate collision with user emails / Onyx groups.
    
    为外部组名添加前缀,以避免与用户邮箱和Onyx组名发生冲突。

    参数:
        ext_group_name (str): 外部组名称
    返回:
        str: 添加了前缀的外部组名
    """
    return f"external_group:{ext_group_name}"


def prefix_group_w_source(ext_group_name: str, source: DocumentSource) -> str:
    """External groups may collide across sources, every source needs its own prefix.
    
    为不同来源的外部组添加前缀,因为不同来源的外部组可能会发生冲突,每个来源需要自己的前缀。

    参数:
        ext_group_name (str): 外部组名称
        source (DocumentSource): 文档来源
    返回:
        str: 添加了来源前缀的外部组名
    """
    return f"{source.value.upper()}_{ext_group_name}"
