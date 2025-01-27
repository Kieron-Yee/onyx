"""
此模块用于处理和提取查询中的时间过滤条件。
主要功能：
1. 解析时间字符串为datetime对象
2. 从用户查询中提取时间过滤条件
3. 处理相对时间（如"最近两个季度"）和绝对时间（如"2022年2月"）
"""

import json
from datetime import datetime
from datetime import timedelta
from datetime import timezone

from dateutil.parser import parse

from onyx.llm.interfaces import LLM
from onyx.llm.utils import dict_based_prompt_to_langchain_prompt
from onyx.llm.utils import message_to_string
from onyx.prompts.filter_extration import TIME_FILTER_PROMPT
from onyx.prompts.prompt_utils import get_current_llm_day_time
from onyx.utils.logger import setup_logger

logger = setup_logger()


def best_match_time(time_str: str) -> datetime | None:
    """
    尝试将时间字符串转换为datetime对象
    
    参数:
        time_str: 需要解析的时间字符串
    
    返回:
        datetime: 解析成功后的datetime对象(UTC时区)
        None: 解析失败时返回None
    """
    preferred_formats = ["%m/%d/%Y", "%m-%d-%Y"]

    for fmt in preferred_formats:
        try:
            # As we don't know if the user is interacting with the API server from
            # the same timezone as the API server, just assume the queries are UTC time
            # the few hours offset (if any) shouldn't make any significant difference
            # 由于不知道用户与API服务器是否在同一时区，我们假设查询都是UTC时间
            # 几个小时的偏差(如果有的话)不会造成显著影响
            dt = datetime.strptime(time_str, fmt)
            return dt.replace(tzinfo=timezone.utc)
        except ValueError:
            continue

    # If the above formats don't match, try using dateutil's parser
    # 如果上述格式不匹配，尝试使用dateutil的解析器
    try:
        dt = parse(time_str)
        return (
            dt.astimezone(timezone.utc)
            if dt.tzinfo
            else dt.replace(tzinfo=timezone.utc)
        )
    except ValueError:
        return None


def extract_time_filter(query: str, llm: LLM) -> tuple[datetime | None, bool]:
    """
    从给定查询中提取时间过滤条件

    参数:
        query: 用户输入的查询字符串
        llm: LLM模型实例

    返回:
        tuple: (datetime对象或None, 布尔值)
            - datetime: 如果应该应用硬时间过滤，返回具体时间；否则为None
            - bool: 如果应该优先考虑最近更新的文档，返回True；否则False
    """

    def _get_time_filter_messages(query: str) -> list[dict[str, str]]:
        """
        生成用于时间过滤的消息列表
        
        参数:
            query: 用户查询字符串
        
        返回:
            list: 包含系统提示和示例的消息列表
        """
        messages = [
            {
                "role": "system",
                "content": TIME_FILTER_PROMPT.format(
                    current_day_time_str=get_current_llm_day_time()
                ),
            },
            {
                "role": "user",
                "content": "What documents in Confluence were written in the last two quarters",
            },
            {
                "role": "assistant",
                "content": json.dumps(
                    {
                        "filter_type": "hard cutoff",
                        "filter_value": "quarter",
                        "value_multiple": 2,
                    }
                ),
            },
            {"role": "user", "content": "What's the latest on project Corgies?"},
            {
                "role": "assistant",
                "content": json.dumps({"filter_type": "favor recent"}),
            },
            {
                "role": "user",
                "content": "Which customer asked about security features in February of 2022?",
            },
            {
                "role": "assistant",
                "content": json.dumps(
                    {"filter_type": "hard cutoff", "date": "02/01/2022"}
                ),
            },
            {"role": "user", "content": query},
        ]
        return messages

    def _extract_time_filter_from_llm_out(
        model_out: str,
    ) -> tuple[datetime | None, bool]:
        """
        从LLM输出中提取时间过滤信息
        
        参数:
            model_out: LLM模型的输出字符串
            
        返回:
            tuple: (datetime对象或None, 布尔值)
                - datetime: 硬截止时间
                - bool: 是否优先考虑最近文档
        """
        try:
            model_json = json.loads(model_out, strict=False)
        except json.JSONDecodeError:
            return None, False

        # If filter type is not present, just assume something has gone wrong
        # Potentially model has identified a date and just returned that but
        # better to be conservative and not identify the wrong filter.
        # 如果未提供过滤类型，假设出现了错误
        # 模型可能识别到日期并直接返回，但为了保险起见，最好不要识别错误的过滤条件
        if "filter_type" not in model_json:
            return None, False

        if "hard" in model_json["filter_type"] or "recent" in model_json["filter_type"]:
            favor_recent = "recent" in model_json["filter_type"]

            if "date" in model_json:
                extracted_time = best_match_time(model_json["date"])
                if extracted_time is not None:
                    # LLM struggles to understand the concept of not sensitive within a time range
                    # So if a time is extracted, just go with that alone
                    # LLM难以理解在时间范围内不敏感的概念
                    # 因此，如果提取到时间，就单独使用它
                    return extracted_time, False

            time_diff = None
            multiplier = 1.0

            if "value_multiple" in model_json:
                try:
                    multiplier = float(model_json["value_multiple"])
                except ValueError:
                    pass

            if "filter_value" in model_json:
                filter_value = model_json["filter_value"]
                if "day" in filter_value:
                    time_diff = timedelta(days=multiplier)
                elif "week" in filter_value:
                    time_diff = timedelta(weeks=multiplier)
                elif "month" in filter_value:
                    # Have to just use the average here, too complicated to calculate exact day
                    # based on current day etc.
                    # 这里只能使用平均值，基于当前日期计算确切日期太复杂
                    time_diff = timedelta(days=multiplier * 30.437)
                elif "quarter" in filter_value:
                    time_diff = timedelta(days=multiplier * 91.25)
                elif "year" in filter_value:
                    time_diff = timedelta(days=multiplier * 365)

            if time_diff is not None:
                current = datetime.now(timezone.utc)
                # LLM struggles to understand the concept of not sensitive within a time range
                # So if a time is extracted, just go with that alone
                # LLM难以理解在时间范围内不敏感的概念
                # 因此，如果提取到时间，就单独使用它
                return current - time_diff, False

            # If we failed to extract a hard filter, just pass back the value of favor recent
            # 如果我们未能提取硬过滤器，只需返回favor recent的值
            return None, favor_recent

        return None, False

    messages = _get_time_filter_messages(query)
    filled_llm_prompt = dict_based_prompt_to_langchain_prompt(messages)
    model_output = message_to_string(llm.invoke(filled_llm_prompt))
    logger.debug(model_output)

    return _extract_time_filter_from_llm_out(model_output)


if __name__ == "__main__":
    # Just for testing purposes, too tedious to unit test as it relies on an LLM
    # 仅用于测试目的，因为依赖LLM所以单元测试比较繁琐
    from onyx.llm.factory import get_default_llms, get_main_llm_from_tuple

    while True:
        user_input = input("Query to Extract Time: ")
        cutoff, recency_bias = extract_time_filter(
            user_input, get_main_llm_from_tuple(get_default_llms())
        )
        print(f"Time Cutoff: {cutoff}")
        print(f"Favor Recent: {recency_bias}")
