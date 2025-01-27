"""
这个文件定义了互联网搜索功能相关的数据模型。
主要包含搜索结果的数据结构和响应格式的定义。
"""

from pydantic import BaseModel


class InternetSearchResult(BaseModel):
    """
    互联网搜索结果的数据模型，表示单条搜索结果
    
    属性:
        title (str): 搜索结果的标题
        link (str): 搜索结果的链接URL
        snippet (str): 搜索结果的摘要内容
    """
    title: str    # 搜索结果标题
    link: str     # 搜索结果链接
    snippet: str  # 搜索结果摘要


class InternetSearchResponse(BaseModel):
    """
    互联网搜索响应的数据模型，包含完整的搜索响应内容
    
    属性:
        revised_query (str): 经过修正的搜索查询词
        internet_results (list[InternetSearchResult]): 搜索结果列表，每个元素都是InternetSearchResult类型
    """
    revised_query: str                           # 修正后的搜索查询词
    internet_results: list[InternetSearchResult] # 搜索结果列表
