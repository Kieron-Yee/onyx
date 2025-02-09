"""
此脚本用于开发环境下运行 Celery 后台任务。
主要功能：
1. 启动多个 Celery worker 进程处理不同类型的任务
2. 启动 Celery beat 进程处理定时任务
3. 监控所有进程的输出并打印到控制台

进程说明：
- primary worker: 处理主要的后台任务队列
- light worker: 处理轻量级、高频率的任务
- heavy worker: 处理重量级、耗时的任务
- indexing worker: 专门处理索引相关的任务
- beat: 负责定时任务的调度
"""

import subprocess
import threading


def monitor_process(process_name: str, process: subprocess.Popen) -> None:
    """
    监控指定进程的输出并将其打印到控制台。
    
    参数:
        process_name: str - 进程名称，用于在输出中标识不同进程
        process: subprocess.Popen - 需要监控的子进程对象
    
    返回:
        None
    """
    assert process.stdout is not None

    while True:
        output = process.stdout.readline()
        if isinstance(output, bytes):
            output = output.decode('utf-8')  # 或者根据实际情况选择其他编码如'gbk'
        if output:
            print(f"{process_name}: {output.strip()}")

        if process.poll() is not None:
            break


def run_jobs() -> None:
    """
    启动并管理所有 Celery 相关进程。
    
    功能：
    1. 设置并启动四个不同类型的 worker 进程：
       - primary: 处理主要任务
       - light: 处理轻量级任务
       - heavy: 处理重量级任务
       - indexing: 处理索引相关任务
    2. 启动 beat 进程处理定时任务
    3. 为每个进程创建监控线程
    4. 等待所有进程完成
    
    返回:
        None
    """
    # 配置 primary worker
    # --pool=threads: 使用线程池
    # --concurrency=6: 同时运行6个线程
    # --prefetch-multiplier=1: 每个线程一次只预取一个任务，避免任务堆积
    cmd_worker_primary = [
        "celery",
        "-A",
        "onyx.background.celery.versioned_apps.primary",
        "worker",
        "--pool=threads",
        "--concurrency=6",
        "--prefetch-multiplier=1",
        "--loglevel=INFO",
        "--hostname=primary@%n",
        "-Q",
        "celery",
    ]

    # 配置 light worker
    # --concurrency=16: 较高的并发数，适合处理大量轻量级任务
    # --prefetch-multiplier=8: 较高的预取倍数，提高吞吐量
    # 处理队列：vespa元数据同步、连接器删除、文档权限更新
    cmd_worker_light = [
        "celery",
        "-A",
        "onyx.background.celery.versioned_apps.light",
        "worker",
        "--pool=threads",
        "--concurrency=16",
        "--prefetch-multiplier=8",
        "--loglevel=INFO",
        "--hostname=light@%n",
        "-Q",
        "vespa_metadata_sync,connector_deletion,doc_permissions_upsert",
    ]

    # 配置 heavy worker
    # 较低的并发数和预取倍数，适合处理资源密集型任务
    # 处理队列：连接器清理、权限同步、外部组同步等重量级任务
    cmd_worker_heavy = [
        "celery",
        "-A",
        "onyx.background.celery.versioned_apps.heavy",
        "worker",
        "--pool=threads",
        "--concurrency=6",
        "--prefetch-multiplier=1",
        "--loglevel=INFO",
        "--hostname=heavy@%n",
        "-Q",
        "connector_pruning,connector_doc_permissions_sync,connector_external_group_sync",
    ]

    # 配置 indexing worker
    # 单线程处理索引任务，确保顺序性和资源控制
    cmd_worker_indexing = [
        "celery",
        "-A",
        "onyx.background.celery.versioned_apps.indexing",
        "worker",
        "--pool=threads",
        "--concurrency=1",
        "--prefetch-multiplier=1",
        "--loglevel=INFO",
        "--hostname=indexing@%n",
        "--queues=connector_indexing",
    ]

    # 配置 beat 进程
    # 负责定时任务的调度管理
    cmd_beat = [
        "celery",
        "-A",
        "onyx.background.celery.versioned_apps.beat",
        "beat",
        "--loglevel=INFO",
    ]

    # 创建子进程，并将标准输出和错误输出重定向到管道
    # text=True 确保输出为字符串而不是字节
    worker_primary_process = subprocess.Popen(
        cmd_worker_primary, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8'
    )

    worker_light_process = subprocess.Popen(
        cmd_worker_light, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8'
    )

    worker_heavy_process = subprocess.Popen(
        cmd_worker_heavy, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,encoding='utf-8'
    )

    worker_indexing_process = subprocess.Popen(
        cmd_worker_indexing, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,encoding='utf-8'
    )

    beat_process = subprocess.Popen(
        cmd_beat, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,encoding='utf-8'
    )

    # 创建监控线程
    # 每个线程负责监控一个进程的输出并打印到控制台
    worker_primary_thread = threading.Thread(
        target=monitor_process, args=("PRIMARY", worker_primary_process)
    )
    worker_light_thread = threading.Thread(
        target=monitor_process, args=("LIGHT", worker_light_process)
    )
    worker_heavy_thread = threading.Thread(
        target=monitor_process, args=("HEAVY", worker_heavy_process)
    )
    worker_indexing_thread = threading.Thread(
        target=monitor_process, args=("INDEX", worker_indexing_process)
    )
    beat_thread = threading.Thread(target=monitor_process, args=("BEAT", beat_process))

    # 启动所有监控线程
    # 注意：线程启动顺序不影响实际任务处理顺序
    worker_primary_thread.start()
    worker_light_thread.start()
    worker_heavy_thread.start()
    worker_indexing_thread.start()
    beat_thread.start()

    # 等待所有线程结束
    # join() 会阻塞直到对应线程结束
    worker_primary_thread.join()
    worker_light_thread.join()
    worker_heavy_thread.join()
    worker_indexing_thread.join()
    beat_thread.join()


if __name__ == "__main__":
    run_jobs()
