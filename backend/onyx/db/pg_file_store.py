"""
此模块用于处理PostgreSQL大对象(Large Objects)存储文件的相关操作。
提供了创建、读取、更新和删除文件的功能，以及与PGFileStore表的交互操作。
"""

import tempfile
from io import BytesIO
from typing import IO

from psycopg2.extensions import connection
from sqlalchemy.orm import Session

from onyx.configs.constants import FileOrigin
from onyx.db.models import PGFileStore
from onyx.file_store.constants import MAX_IN_MEMORY_SIZE
from onyx.file_store.constants import STANDARD_CHUNK_SIZE
from onyx.utils.logger import setup_logger

logger = setup_logger()


def get_pg_conn_from_session(db_session: Session) -> connection:
    """
    从SQLAlchemy会话中获取PostgreSQL原生连接对象
    
    Args:
        db_session: SQLAlchemy会话对象
    Returns:
        PostgreSQL原生连接对象
    """
    return db_session.connection().connection.connection  # type: ignore


def get_pgfilestore_by_file_name(
    file_name: str,
    db_session: Session,
) -> PGFileStore:
    """
    根据文件名从数据库中获取PGFileStore对象
    
    Args:
        file_name: 文件名
        db_session: 数据库会话对象
    Returns:
        PGFileStore对象
    Raises:
        RuntimeError: 当文件不存在时抛出异常
    """
    pgfilestore = db_session.query(PGFileStore).filter_by(file_name=file_name).first()

    if not pgfilestore:
        raise RuntimeError(f"File by name {file_name} does not exist or was deleted")
        # 文件名为{file_name}的文件不存在或已被删除

    return pgfilestore


def delete_pgfilestore_by_file_name(
    file_name: str,
    db_session: Session,
) -> None:
    """
    根据文件名删除PGFileStore表中的记录
    
    Args:
        file_name: 要删除的文件名
        db_session: 数据库会话对象
    """
    db_session.query(PGFileStore).filter_by(file_name=file_name).delete()


def create_populate_lobj(
    content: IO,
    db_session: Session,
) -> int:
    """
    创建并填充PostgreSQL大对象
    
    注意：此函数不会提交数据库更改，因为提交应该与PGFileStore行创建一起进行
    这一步会同时完成Large Object和跟踪它的表的创建
    Note, this does not commit the changes to the DB
    This is because the commit should happen with the PGFileStore row creation
    That step finalizes both the Large Object and the table tracking it
    
    Args:
        content: 文件内容的IO对象
        db_session: 数据库会话对象
    Returns:
        创建的大对象的OID
    """
    pg_conn = get_pg_conn_from_session(db_session)
    large_object = pg_conn.lobject()

    # write in multiple chunks to avoid loading the whole file into memory
    while True:
        chunk = content.read(STANDARD_CHUNK_SIZE)
        if not chunk:
            break
        large_object.write(chunk)

    large_object.close()

    return large_object.oid


def read_lobj(
    lobj_oid: int,
    db_session: Session,
    mode: str | None = None,
    use_tempfile: bool = False,
) -> IO:
    """
    读取PostgreSQL大对象的内容
    
    Args:
        lobj_oid: 大对象的OID
        db_session: 数据库会话对象
        mode: 读取模式
        use_tempfile: 是否使用临时文件
    Returns:
        包含文件内容的IO对象
    """
    pg_conn = get_pg_conn_from_session(db_session)
    large_object = (
        pg_conn.lobject(lobj_oid, mode=mode) if mode else pg_conn.lobject(lobj_oid)
    )

    if use_tempfile:
        temp_file = tempfile.SpooledTemporaryFile(max_size=MAX_IN_MEMORY_SIZE)
        while True:
            chunk = large_object.read(STANDARD_CHUNK_SIZE)
            if not chunk:
                break
            temp_file.write(chunk)
        temp_file.seek(0)
        return temp_file
    else:
        return BytesIO(large_object.read())


def delete_lobj_by_id(
    lobj_oid: int,
    db_session: Session,
) -> None:
    """
    根据OID删除PostgreSQL大对象
    
    Args:
        lobj_oid: 要删除的大对象的OID
        db_session: 数据库会话对象
    """
    pg_conn = get_pg_conn_from_session(db_session)
    pg_conn.lobject(lobj_oid).unlink()


def delete_lobj_by_name(
    lobj_name: str,
    db_session: Session,
) -> None:
    """
    根据文件名删除PostgreSQL大对象及其在PGFileStore表中的记录
    
    Args:
        lobj_name: 要删除的文件名
        db_session: 数据库会话对象
    """
    try:
        pgfilestore = get_pgfilestore_by_file_name(lobj_name, db_session)
    except RuntimeError:
        logger.info(f"no file with name {lobj_name} found")  # 未找到名为{lobj_name}的文件
        return

    pg_conn = get_pg_conn_from_session(db_session)
    pg_conn.lobject(pgfilestore.lobj_oid).unlink()

    delete_pgfilestore_by_file_name(lobj_name, db_session)
    db_session.commit()


def upsert_pgfilestore(
    file_name: str,
    display_name: str | None,
    file_origin: FileOrigin,
    file_type: str,
    lobj_oid: int,
    db_session: Session,
    commit: bool = False,
    file_metadata: dict | None = None,
) -> PGFileStore:
    """
    创建或更新PGFileStore表中的文件记录
    
    Args:
        file_name: 文件名
        display_name: 显示名称
        file_origin: 文件来源
        file_type: 文件类型
        lobj_oid: 大对象的OID
        db_session: 数据库会话对象
        commit: 是否立即提交更改
        file_metadata: 文件元数据
    Returns:
        PGFileStore对象
    """
    pgfilestore = db_session.query(PGFileStore).filter_by(file_name=file_name).first()

    if pgfilestore:
        try:
            # 这种情况在正常执行中不应该发生
            # This should not happen in normal execution
            delete_lobj_by_id(lobj_oid=pgfilestore.lobj_oid, db_session=db_session)
        except Exception:
            # 如果删除也失败了，说明大对象不存在，即使删除失败也不会太糟糕，因为大多数文件大小都不显著
            # If the delete fails as well, the large object doesn't exist anyway and even if it
            # fails to delete, it's not too terrible as most files sizes are insignificant
            logger.error(
                f"Failed to delete large object with oid {pgfilestore.lobj_oid}"  # 删除OID为{pgfilestore.lobj_oid}的大对象失败
            )

        pgfilestore.lobj_oid = lobj_oid
    else:
        pgfilestore = PGFileStore(
            file_name=file_name,
            display_name=display_name,
            file_origin=file_origin,
            file_type=file_type,
            file_metadata=file_metadata,
            lobj_oid=lobj_oid,
        )
        db_session.add(pgfilestore)

    if commit:
        db_session.commit()

    return pgfilestore
