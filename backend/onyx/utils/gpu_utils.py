"""
GPU状态检查工具模块

本模块提供了用于检查GPU可用状态的功能函数。
通过HTTP请求访问模型服务器，获取GPU的当前状态。
"""

import requests
from retry import retry

from onyx.utils.logger import setup_logger
from shared_configs.configs import INDEXING_MODEL_SERVER_HOST
from shared_configs.configs import INDEXING_MODEL_SERVER_PORT
from shared_configs.configs import MODEL_SERVER_HOST
from shared_configs.configs import MODEL_SERVER_PORT

logger = setup_logger()


@retry(tries=5, delay=5)
def gpu_status_request(indexing: bool = True) -> bool:
    """
    检查GPU状态的函数
    
    通过HTTP请求检查模型服务器上GPU的可用状态。如果请求失败会自动重试5次，每次重试间隔5秒。
    
    参数:
        indexing (bool): 是否检查索引服务器的GPU。
                        True表示检查索引服务器，False表示检查普通模型服务器。
                        默认值为True。
    
    返回:
        bool: GPU是否可用。True表示可用，False表示不可用。
    
    异常:
        requests.RequestException: 当HTTP请求失败时抛出。
    """
    if indexing:
        model_server_url = f"{INDEXING_MODEL_SERVER_HOST}:{INDEXING_MODEL_SERVER_PORT}"
    else:
        model_server_url = f"{MODEL_SERVER_HOST}:{MODEL_SERVER_PORT}"

    # 确保URL包含http前缀
    if "http" not in model_server_url:
        model_server_url = f"http://{model_server_url}"

    try:
        response = requests.get(f"{model_server_url}/api/gpu-status", timeout=10)
        response.raise_for_status()
        gpu_status = response.json()
        return gpu_status["gpu_available"]
    except requests.RequestException as e:
        # Error: Unable to fetch GPU status.
        # 错误：无法获取GPU状态
        logger.error(f"Error: Unable to fetch GPU status. Error: {str(e)}")
        raise  # 重新抛出异常以触发重试
