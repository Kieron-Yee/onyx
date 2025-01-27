"""
该文件实现了文档索引相关的API路由功能。
主要提供了查询索引错误的管理接口。
"""

from fastapi import APIRouter
from fastapi import Depends
from sqlalchemy.orm import Session

from onyx.auth.users import current_admin_user
from onyx.db.engine import get_session
from onyx.db.index_attempt import (
    get_index_attempt_errors,
)
from onyx.db.models import User
from onyx.server.documents.models import IndexAttemptError

# 创建一个路由器，设置前缀为"/manage"
router = APIRouter(prefix="/manage")


@router.get("/admin/indexing-errors/{index_attempt_id}")
def get_indexing_errors(
    index_attempt_id: int,
    _: User | None = Depends(current_admin_user),
    db_session: Session = Depends(get_session),
) -> list[IndexAttemptError]:
    """
    获取指定索引尝试的错误信息。
    
    参数:
        index_attempt_id: 索引尝试的ID
        _: 当前管理员用户（用于权限验证）
        db_session: 数据库会话实例
    
    返回:
        list[IndexAttemptError]: 包含索引错误信息的列表
    """
    indexing_errors = get_index_attempt_errors(index_attempt_id, db_session)
    return [IndexAttemptError.from_db_model(e) for e in indexing_errors]
