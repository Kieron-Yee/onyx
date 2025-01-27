"""
文件存储模块
本模块提供了文件存储的抽象接口和具体实现，主要用于处理大型二进制对象的存储和检索。
主要包含两个类：
1. FileStore: 文件存储的抽象基类，定义了文件操作的基本接口
2. PostgresBackedFileStore: 基于PostgreSQL的具体文件存储实现
"""

from abc import ABC
from abc import abstractmethod
from typing import IO

from sqlalchemy.orm import Session

from onyx.configs.constants import FileOrigin
from onyx.db.models import PGFileStore
from onyx.db.pg_file_store import create_populate_lobj
from onyx.db.pg_file_store import delete_lobj_by_id
from onyx.db.pg_file_store import delete_pgfilestore_by_file_name
from onyx.db.pg_file_store import get_pgfilestore_by_file_name
from onyx.db.pg_file_store import read_lobj
from onyx.db.pg_file_store import upsert_pgfilestore


class FileStore(ABC):
    """
    An abstraction for storing files and large binary objects.
    文件和大型二进制对象存储的抽象接口。
    """

    @abstractmethod
    def save_file(
        self,
        file_name: str,
        content: IO,
        display_name: str | None,
        file_origin: FileOrigin,
        file_type: str,
        file_metadata: dict | None = None,
    ) -> None:
        """
        Save a file to the blob store
        将文件保存到二进制对象存储中

        参数说明:
        - file_name: 要保存的文件名
        - content: 文件内容
        - display_name: 文件显示名称
        - file_origin: 文件来源
        - file_type: 文件类型
        - file_metadata: 文件元数据（可选）
        """
        raise NotImplementedError

    @abstractmethod
    def read_file(
        self, file_name: str, mode: str | None, use_tempfile: bool = False
    ) -> IO:
        """
        Read the content of a given file by the name
        通过文件名读取指定文件的内容

        参数说明:
        - file_name: 要读取的文件名
        - mode: 文件打开模式（如 'b' 表示二进制模式）
        - use_tempfile: 是否使用临时文件存储内容，避免将整个文件加载到内存中

        返回值:
            文件内容和元数据字典
        """
        raise NotImplementedError

    @abstractmethod
    def read_file_record(self, file_name: str) -> PGFileStore:
        """
        Read the file record by the name
        通过文件名读取文件记录

        参数说明:
        - file_name: 要读取的文件名

        返回值:
            PGFileStore对象
        """
        raise NotImplementedError

    @abstractmethod
    def delete_file(self, file_name: str) -> None:
        """
        Delete a file by its name.
        通过文件名删除文件

        参数说明:
        - file_name: 要删除的文件名
        """
        raise NotImplementedError


class PostgresBackedFileStore(FileStore):
    """
    PostgreSQL文件存储实现类
    提供了基于PostgreSQL大对象（Large Object）的文件存储实现
    """

    def __init__(self, db_session: Session):
        """
        初始化PostgreSQL文件存储

        参数说明:
        - db_session: 数据库会话对象
        """
        self.db_session = db_session

    def save_file(
        self,
        file_name: str,
        content: IO,
        display_name: str | None,
        file_origin: FileOrigin,
        file_type: str,
        file_metadata: dict | None = None,
    ) -> None:
        """
        将文件保存到PostgreSQL存储中

        参数说明与基类相同
        """
        try:
            # The large objects in postgres are saved as special objects can be listed with
            # SELECT * FROM pg_largeobject_metadata;
            # PostgreSQL中的大对象作为特殊对象保存，可以通过以下SQL语句列出：
            # SELECT * FROM pg_largeobject_metadata;
            obj_id = create_populate_lobj(content=content, db_session=self.db_session)
            upsert_pgfilestore(
                file_name=file_name,
                display_name=display_name or file_name,
                file_origin=file_origin,
                file_type=file_type,
                lobj_oid=obj_id,
                db_session=self.db_session,
                file_metadata=file_metadata,
            )
            self.db_session.commit()
        except Exception:
            self.db_session.rollback()
            raise

    def read_file(
        self, file_name: str, mode: str | None = None, use_tempfile: bool = False
    ) -> IO:
        """
        通过文件名读取指定文件的内容

        参数说明:
        - file_name: 要读取的文件名
        - mode: 文件打开模式（如 'b' 表示二进制模式）
        - use_tempfile: 是否使用临时文件存储内容，避免将整个文件加载到内存中

        返回值:
            文件内容和元数据字典
        """
        file_record = get_pgfilestore_by_file_name(
            file_name=file_name, db_session=self.db_session
        )
        return read_lobj(
            lobj_oid=file_record.lobj_oid,
            db_session=self.db_session,
            mode=mode,
            use_tempfile=use_tempfile,
        )

    def read_file_record(self, file_name: str) -> PGFileStore:
        """
        通过文件名读取文件记录

        参数说明:
        - file_name: 要读取的文件名

        返回值:
            PGFileStore对象
        """
        file_record = get_pgfilestore_by_file_name(
            file_name=file_name, db_session=self.db_session
        )

        return file_record

    def delete_file(self, file_name: str) -> None:
        """
        通过文件名删除文件

        参数说明:
        - file_name: 要删除的文件名
        """
        try:
            file_record = get_pgfilestore_by_file_name(
                file_name=file_name, db_session=self.db_session
            )
            delete_lobj_by_id(file_record.lobj_oid, db_session=self.db_session)
            delete_pgfilestore_by_file_name(
                file_name=file_name, db_session=self.db_session
            )
            self.db_session.commit()
        except Exception:
            self.db_session.rollback()
            raise


def get_default_file_store(db_session: Session) -> FileStore:
    """
    获取默认文件存储实现

    参数说明:
    - db_session: 数据库会话对象

    返回值:
        FileStore接口的实现类实例（目前仅支持PostgreSQL实现）
    """
    # The only supported file store now is the Postgres File Store
    # 目前仅支持PostgreSQL文件存储
    return PostgresBackedFileStore(db_session=db_session)
