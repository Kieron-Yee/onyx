"""
这个文件包含了一系列用于在Redis中处理Celery任务的辅助函数。
主要功能包括：
- 检查未确认任务的数量和ID
- 获取任务队列长度
- 查找特定任务
- 检查worker状态和任务状态
"""

# These are helper objects for tracking the keys we need to write in redis
# 这些是用于跟踪我们需要在redis中写入的键的辅助对象
import json
from typing import Any
from typing import cast

from celery import Celery
from redis import Redis

from onyx.background.celery.configs.base import CELERY_SEPARATOR
from onyx.configs.constants import OnyxCeleryPriority


def celery_get_unacked_length(r: Redis) -> int:
    """Checking the unacked queue is useful because a non-zero length tells us there
    may be prefetched tasks.

    There can be other tasks in here besides indexing tasks, so this is mostly useful
    just to see if the task count is non zero.

    检查未确认队列很有用，因为非零长度表明可能存在预取的任务。

    除了索引任务之外，这里可能还有其他任务，所以这主要用于查看任务计数是否非零。

    ref: https://blog.hikaru.run/2022/08/29/get-waiting-tasks-count-in-celery.html

    参数:
        r: Redis连接实例
    返回值:
        未确认队列的长度
    """
    length = cast(int, r.hlen("unacked"))
    return length


def celery_get_unacked_task_ids(queue: str, r: Redis) -> set[str]:
    """Gets the set of task id's matching the given queue in the unacked hash.

    Unacked entries belonging to the indexing queue are "prefetched", so this gives
    us crucial visibility as to what tasks are in that state.

    获取未确认哈希中与给定队列匹配的任务ID集合。

    属于索引队列的未确认条目是"预取"的，因此这让我们能够清楚地了解处于该状态的任务。

    参数:
        queue: 队列名称
        r: Redis连接实例
    返回值:
        未确认任务ID的集合
    """
    tasks: set[str] = set()

    for _, v in r.hscan_iter("unacked"):
        v_bytes = cast(bytes, v)
        v_str = v_bytes.decode("utf-8")
        task = json.loads(v_str)

        task_description = task[0]
        task_queue = task[2]

        if task_queue != queue:
            continue

        task_id = task_description.get("headers", {}).get("id")
        if not task_id:
            continue

        # if the queue matches and we see the task_id, add it
        tasks.add(task_id)
    return tasks


def celery_get_queue_length(queue: str, r: Redis) -> int:
    """This is a redis specific way to get the length of a celery queue.
    It is priority aware and knows how to count across the multiple redis lists
    used to implement task prioritization.
    This operation is not atomic.

    这是一个Redis特定的方法，用于获取Celery队列的长度。
    它能够识别优先级，并知道如何统计用于实现任务优先级的多个Redis列表。
    这个操作不是原子性的。

    参数:
        queue: 队列名称
        r: Redis连接实例
    返回值:
        队列的总长度
    """
    total_length = 0
    for i in range(len(OnyxCeleryPriority)):
        queue_name = queue
        if i > 0:
            queue_name += CELERY_SEPARATOR
            queue_name += str(i)

        length = r.llen(queue_name)
        total_length += cast(int, length)

    return total_length


def celery_find_task(task_id: str, queue: str, r: Redis) -> int:
    """This is a redis specific way to find a task for a particular queue in redis.
    It is priority aware and knows how to look through the multiple redis lists
    used to implement task prioritization.
    This operation is not atomic.

    这是一个Redis特定的方法，用于在Redis中查找特定队列的任务。
    它能够识别优先级，并知道如何查找用于实现任务优先级的多个Redis列表。
    这个操作不是原子性的。

    参数:
        task_id: 要查找的任务ID
        queue: 队列名称
        r: Redis连接实例
    返回值:
        如果任务ID在队列中存在则返回True，否则返回False
    """
    for priority in range(len(OnyxCeleryPriority)):
        queue_name = f"{queue}{CELERY_SEPARATOR}{priority}" if priority > 0 else queue

        tasks = cast(list[bytes], r.lrange(queue_name, 0, -1))
        for task in tasks:
            task_dict: dict[str, Any] = json.loads(task.decode("utf-8"))
            if task_dict.get("headers", {}).get("id") == task_id:
                return True

    return False


def celery_inspect_get_workers(name_filter: str | None, app: Celery) -> list[str]:
    """Returns a list of current workers containing name_filter, or all workers if
    name_filter is None.

    返回包含name_filter的当前worker列表，如果name_filter为None则返回所有worker。

    We've empirically discovered that the celery inspect API is potentially unstable
    and may hang or return empty results when celery is under load. Suggest using this
    more to debug and troubleshoot than in production code.

    我们通过经验发现，当Celery负载较高时，inspect API可能不稳定，可能会挂起或返回空结果。
    建议将其更多地用于调试和故障排除，而不是在生产代码中使用。

    参数:
        name_filter: worker名称过滤器
        app: Celery应用实例
    返回值:
        匹配的worker名称列表
    """
    worker_names: list[str] = []

    # filter for and create an indexing specific inspect object
    inspect = app.control.inspect()
    workers: dict[str, Any] = inspect.ping()  # type: ignore
    if workers:
        for worker_name in list(workers.keys()):
            # if the name filter not set, return all worker names
            if not name_filter:
                worker_names.append(worker_name)
                continue

            # if the name filter is set, return only worker names that contain the name filter
            if name_filter not in worker_name:
                continue

            worker_names.append(worker_name)

    return worker_names


def celery_inspect_get_reserved(worker_names: list[str], app: Celery) -> set[str]:
    """Returns a list of reserved tasks on the specified workers.

    返回指定worker上的预留任务列表。

    We've empirically discovered that the celery inspect API is potentially unstable
    and may hang or return empty results when celery is under load. Suggest using this
    more to debug and troubleshoot than in production code.

    我们通过经验发现，当Celery负载较高时，inspect API可能不稳定，可能会挂起或返回空结果。
    建议将其更多地用于调试和故障排除，而不是在生产代码中使用。

    参数:
        worker_names: worker名称列表
        app: Celery应用实例
    返回值:
        预留任务ID的集合
    """
    reserved_task_ids: set[str] = set()

    inspect = app.control.inspect(destination=worker_names)

    # get the list of reserved tasks
    reserved_tasks: dict[str, list] | None = inspect.reserved()  # type: ignore
    if reserved_tasks:
        for _, task_list in reserved_tasks.items():
            for task in task_list:
                reserved_task_ids.add(task["id"])

    return reserved_task_ids


def celery_inspect_get_active(worker_names: list[str], app: Celery) -> set[str]:
    """Returns a list of active tasks on the specified workers.

    返回指定worker上的活动任务列表。

    We've empirically discovered that the celery inspect API is potentially unstable
    and may hang or return empty results when celery is under load. Suggest using this
    more to debug and troubleshoot than in production code.

    我们通过经验发现，当Celery负载较高时，inspect API可能不稳定，可能会挂起或返回空结果。
    建议将其更多地用于调试和故障排除，而不是在生产代码中使用。

    参数:
        worker_names: worker名称列表
        app: Celery应用实例
    返回值:
        活动任务ID的集合
    """
    active_task_ids: set[str] = set()

    inspect = app.control.inspect(destination=worker_names)

    # get the list of reserved tasks
    active_tasks: dict[str, list] | None = inspect.active()  # type: ignore
    if active_tasks:
        for _, task_list in active_tasks.items():
            for task in task_list:
                active_task_ids.add(task["id"])

    return active_task_ids
