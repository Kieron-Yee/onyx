"""
此文件定义了生成式AI相关的异常类。
主要包含生成式AI功能被禁用时抛出的异常。
"""

class GenAIDisabledException(Exception):
    """
    生成式AI功能被禁用时抛出的异常类
    
    属性:
        message (str): 异常信息
    
    继承:
        Exception: Python标准异常基类
    """
    def __init__(self, message: str = "Generative AI has been turned off") -> None:
        """
        初始化GenAIDisabledException异常实例
        
        参数:
            message (str): 异常信息，默认为"Generative AI has been turned off"
            
        返回:
            None
        """
        self.message = message
        super().__init__(self.message)
