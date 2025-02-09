"""
此模块实现了一个内存跟踪器，用于监控和分析程序的内存使用情况。
主要功能包括：
- 跟踪内存分配
- 创建内存快照
- 比较不同时间点的内存使用差异
- 输出内存使用统计信息
"""

import tracemalloc

from onyx.utils.logger import setup_logger

logger = setup_logger()

# 设置跟踪的调用帧数量
DANSWER_TRACEMALLOC_FRAMES = 10


class OnyxTracer:
    """
    内存跟踪器类，用于跟踪和分析程序的内存使用情况。
    """
    
    def __init__(self) -> None:
        """
        初始化跟踪器，创建用于存储不同时间点内存快照的属性。
        """
        self.snapshot_first: tracemalloc.Snapshot | None = None  # 第一次快照
        self.snapshot_prev: tracemalloc.Snapshot | None = None   # 上一次快照
        self.snapshot: tracemalloc.Snapshot | None = None        # 当前快照

    def start(self) -> None:
        """
        启动内存跟踪。
        """
        tracemalloc.start(DANSWER_TRACEMALLOC_FRAMES)

    def stop(self) -> None:
        """
        停止内存跟踪。
        """
        tracemalloc.stop()

    def snap(self) -> None:
        """
        创建当前内存使用的快照，并更新快照历史记录。
        """
        snapshot = tracemalloc.take_snapshot()
        # Filter out irrelevant frames (e.g., from tracemalloc itself or importlib)
        # 过滤掉无关的调用帧（例如：来自tracemalloc本身或importlib的帧）
        snapshot = snapshot.filter_traces(
            (
                tracemalloc.Filter(False, tracemalloc.__file__),  # Exclude tracemalloc / 排除tracemalloc
                tracemalloc.Filter(
                    False, "<frozen importlib._bootstrap>"
                ),  # Exclude importlib / 排除importlib
                tracemalloc.Filter(
                    False, "<frozen importlib._bootstrap_external>"
                ),  # Exclude external importlib / 排除外部importlib
            )
        )

        if not self.snapshot_first:
            self.snapshot_first = snapshot

        if self.snapshot:
            self.snapshot_prev = self.snapshot

        self.snapshot = snapshot

    def log_snapshot(self, numEntries: int) -> None:
        """
        记录当前快照的内存使用统计信息。

        参数:
            numEntries: int - 要显示的统计条目数量
        """
        if not self.snapshot:
            return

        stats = self.snapshot.statistics("traceback")
        for s in stats[:numEntries]:
            logger.debug(f"Tracer snap: {s}")
            for line in s.traceback:
                logger.debug(f"* {line}")

    @staticmethod
    def log_diff(
        snap_current: tracemalloc.Snapshot,
        snap_previous: tracemalloc.Snapshot,
        numEntries: int,
    ) -> None:
        """
        比较并记录两个快照之间的内存使用差异。

        参数:
            snap_current: tracemalloc.Snapshot - 当前快照
            snap_previous: tracemalloc.Snapshot - 之前的快照
            numEntries: int - 要显示的差异条目数量
        """
        stats = snap_current.compare_to(snap_previous, "traceback")
        for s in stats[:numEntries]:
            logger.debug(f"Tracer diff: {s}")
            for line in s.traceback.format():
                logger.debug(f"* {line}")

    def log_previous_diff(self, numEntries: int) -> None:
        """
        记录当前快照与上一个快照之间的内存使用差异。

        参数:
            numEntries: int - 要显示的差异条目数量
        """
        if not self.snapshot or not self.snapshot_prev:
            return

        OnyxTracer.log_diff(self.snapshot, self.snapshot_prev, numEntries)

    def log_first_diff(self, numEntries: int) -> None:
        """
        记录当前快照与第一个快照之间的内存使用差异。

        参数:
            numEntries: int - 要显示的差异条目数量
        """
        if not self.snapshot or not self.snapshot_first:
            return

        OnyxTracer.log_diff(self.snapshot, self.snapshot_first, numEntries)
