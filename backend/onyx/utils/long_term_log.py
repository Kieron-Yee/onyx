"""
此模块提供长期日志记录功能。
主要用于：
1. 支持大量数据的持久化存储
2. 通过后台线程实现快速的日志写入
3. 支持按类别和时间范围查询日志
"""

import json
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from onyx.utils.logger import setup_logger
from onyx.utils.special_types import JSON_ro

logger = setup_logger()

_LOG_FILE_NAME_TIMESTAMP_FORMAT = "%Y-%m-%d_%H-%M-%S-%f"


class LongTermLogger:
    """NOTE: should support a LOT of data AND should be extremely fast,
    ideally done in a background thread.
    注：应该支持大量数据且要求极快的处理速度，理想情况下在后台线程中执行。"""

    def __init__(
        self,
        metadata: dict[str, str] | None = None,
        log_file_path: str = "/tmp/long_term_log",
        max_files_per_category: int = 1000,
    ):
        """
        初始化长期日志记录器
        
        参数:
            metadata: 要记录的元数据字典
            log_file_path: 日志文件存储路径
            max_files_per_category: 每个类别最大文件数量
        """
        self.metadata = metadata
        self.log_file_path = Path(log_file_path)
        self.max_files_per_category = max_files_per_category
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
        except Exception as e:
            logger.error(f"Error creating directory for long-term logs: {e}")

    def _cleanup_old_files(self, category_path: Path) -> None:
        """
        清理超出数量限制的旧日志文件
        
        参数:
            category_path: 日志类别目录路径
        """
        try:
            files = sorted(
                category_path.glob("*.json"),
                key=lambda x: x.stat().st_mtime,  # Sort by modification time
                reverse=True,
            )
            # Delete oldest files that exceed the limit
            for file in files[self.max_files_per_category :]:
                try:
                    file.unlink()
                except Exception as e:
                    logger.error(f"Error deleting old log file {file}: {e}")
        except Exception as e:
            logger.error(f"Error during log rotation cleanup: {e}")

    def _record(self, message: Any, category: str) -> None:
        """
        记录单条日志信息到文件
        
        参数:
            message: 要记录的日志信息
            category: 日志类别
        """
        category_path = self.log_file_path / category
        try:
            # Create directory if it doesn't exist
            os.makedirs(category_path, exist_ok=True)

            # Perform cleanup before writing new file
            self._cleanup_old_files(category_path)

            final_record = {
                "metadata": self.metadata,
                "record": message,
            }

            file_path = (
                category_path
                / f"{datetime.now().strftime(_LOG_FILE_NAME_TIMESTAMP_FORMAT)}.json"
            )
            with open(file_path, "w+") as f:
                # default allows us to "ignore" unserializable objects
                json.dump(final_record, f, default=lambda x: str(x))
        except Exception as e:
            logger.error(f"Error recording log: {e}")

    def record(self, message: JSON_ro, category: str = "default") -> None:
        """
        在后台线程中异步记录日志信息
        
        参数:
            message: 要记录的JSON格式日志信息
            category: 日志类别，默认为'default'
        """
        try:
            # Run in separate thread to have minimal overhead in main flows
            # 在单独的线程中运行以最小化主流程的开销
            thread = threading.Thread(
                target=self._record, args=(message, category), daemon=True
            )
            thread.start()
        except Exception:
            # Should never interfere with normal functions of Onyx
            # 不应该影响Onyx的正常功能
            pass

    def fetch_category(
        self,
        category: str,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int = 100,
    ) -> list[JSON_ro]:
        """
        获取指定类别和时间范围内的日志记录
        
        参数:
            category: 日志类别
            start_time: 开始时间，可选
            end_time: 结束时间，可选
            limit: 返回记录的最大数量
        
        返回:
            符合条件的日志记录列表
        """
        category_path = self.log_file_path / category
        files = list(category_path.glob("*.json"))

        results: list[JSON_ro] = []
        for file in files:
            # Parse timestamp from filename (YYYY-MM-DD_HH-MM-SS.json)
            try:
                file_time = datetime.strptime(
                    file.stem, _LOG_FILE_NAME_TIMESTAMP_FORMAT
                )

                # Skip if outside time range
                if start_time and file_time < start_time:
                    continue
                if end_time and file_time > end_time:
                    continue

                results.append(json.loads(file.read_text()))
            except ValueError:
                # Skip files that don't match expected format
                continue

        return results
