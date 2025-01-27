"""
GPT搜索API模块
本模块提供了GPT文档搜索相关的API接口，实现了基于用户查询的文档检索功能
"""

import math
from datetime import datetime

from fastapi import APIRouter
from fastapi import Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from onyx.context.search.models import SearchRequest
from onyx.context.search.pipeline import SearchPipeline
from onyx.db.engine import get_session
from onyx.db.models import User
from onyx.llm.factory import get_default_llms
from onyx.server.onyx_api.ingestion import api_key_dep
from onyx.utils.logger import setup_logger


logger = setup_logger()


router = APIRouter(prefix="/gpts")


def time_ago(dt: datetime) -> str:
    """
    计算给定时间到现在的时间差，并返回人性化的表示
    
    参数:
        dt: 需要计算的日期时间
    
    返回:
        str: 人性化的时间差表示，如"~5 minutes"、"~2 hours"等
    """
    # Calculate time difference / 计算时间差
    now = datetime.now()
    diff = now - dt

    # Convert difference to minutes / 将时间差转换为分钟
    minutes = diff.total_seconds() / 60

    # Determine the appropriate unit and calculate the age
    # 确定适当的时间单位并计算时间差
    if minutes < 60:
        return f"~{math.floor(minutes)} minutes"
    hours = minutes / 60
    if hours < 24:
        return f"~{math.floor(hours)} hours"
    days = hours / 24
    if days < 7:
        return f"~{math.floor(days)} days"
    weeks = days / 7
    if weeks < 4:
        return f"~{math.floor(weeks)} weeks"
    months = days / 30
    return f"~{math.floor(months)} months"


class GptSearchRequest(BaseModel):
    """
    GPT搜索请求模型
    
    属性:
        query: 用户的搜索查询字符串
    """
    query: str


class GptDocChunk(BaseModel):
    """
    GPT文档块模型，表示搜索结果中的一个文档片段
    
    属性:
        title: 文档标题
        content: 文档内容
        source_type: 文档来源类型
        link: 文档链接
        metadata: 文档元数据
        document_age: 文档年龄（多久之前创建或更新）
    """
    title: str
    content: str
    source_type: str
    link: str
    metadata: dict[str, str | list[str]]
    document_age: str


class GptSearchResponse(BaseModel):
    """
    GPT搜索响应模型
    
    属性:
        matching_document_chunks: 匹配的文档块列表
    """
    matching_document_chunks: list[GptDocChunk]


@router.post("/gpt-document-search")
def gpt_search(
    search_request: GptSearchRequest,
    _: User | None = Depends(api_key_dep),
    db_session: Session = Depends(get_session),
) -> GptSearchResponse:
    """
    处理GPT文档搜索请求的API端点
    
    参数:
        search_request: 搜索请求对象
        _: 用户对象（用于API认证）
        db_session: 数据库会话对象
    
    返回:
        GptSearchResponse: 搜索结果响应对象
    """
    llm, fast_llm = get_default_llms()
    top_sections = SearchPipeline(
        search_request=SearchRequest(
            query=search_request.query,
        ),
        user=None,
        llm=llm,
        fast_llm=fast_llm,
        db_session=db_session,
    ).reranked_sections

    return GptSearchResponse(
        matching_document_chunks=[
            GptDocChunk(
                title=section.center_chunk.semantic_identifier,
                content=section.center_chunk.content,
                source_type=section.center_chunk.source_type,
                link=section.center_chunk.source_links.get(0, "")
                if section.center_chunk.source_links
                else "",
                metadata=section.center_chunk.metadata,
                document_age=time_ago(section.center_chunk.updated_at)
                if section.center_chunk.updated_at
                else "Unknown",
            )
            for section in top_sections
        ],
    )
