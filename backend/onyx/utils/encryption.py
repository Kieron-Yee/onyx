"""
加密工具模块
本模块提供了字符串加密和解密的基本功能。
在MIT版本中，这些功能仅提供基础的编码和解码操作。
完整的加密功能请参考Onyx企业版。
"""

from onyx.configs.app_configs import ENCRYPTION_KEY_SECRET
from onyx.utils.logger import setup_logger
from onyx.utils.variable_functionality import fetch_versioned_implementation

logger = setup_logger()


def _encrypt_string(input_str: str) -> bytes:
    """
    将输入字符串进行加密
    
    Args:
        input_str (str): 需要加密的字符串
    
    Returns:
        bytes: 加密后的字节数据
    """
    if ENCRYPTION_KEY_SECRET:
        logger.warning("MIT version of Onyx does not support encryption of secrets.")
        # MIT版本的Onyx不支持密钥加密
    return input_str.encode()


def _decrypt_bytes(input_bytes: bytes) -> str:
    """
    解密字节数据为字符串
    
    Args:
        input_bytes (bytes): 需要解密的字节数据
    
    Returns:
        str: 解密后的字符串
    """
    # No need to double warn. If you wish to learn more about encryption features
    # refer to the Onyx EE code
    # 无需重复警告。如果想了解更多加密功能，请参考Onyx企业版代码
    return input_bytes.decode()


def encrypt_string_to_bytes(intput_str: str) -> bytes:
    """
    调用版本相关的加密函数对字符串进行加密
    
    Args:
        intput_str (str): 需要加密的字符串
    
    Returns:
        bytes: 加密后的字节数据
    """
    versioned_encryption_fn = fetch_versioned_implementation(
        "onyx.utils.encryption", "_encrypt_string"
    )
    return versioned_encryption_fn(intput_str)


def decrypt_bytes_to_string(intput_bytes: bytes) -> str:
    """
    调用版本相关的解密函数对字节数据进行解密
    
    Args:
        intput_bytes (bytes): 需要解密的字节数据
    
    Returns:
        str: 解密后的字符串
    """
    versioned_decryption_fn = fetch_versioned_implementation(
        "onyx.utils.encryption", "_decrypt_bytes"
    )
    return versioned_decryption_fn(intput_bytes)
