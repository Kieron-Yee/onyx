"""
此文件用于验证LLM模型生成的答案的有效性。
主要功能包括：
1. 检查答案的合法性
2. 验证答案是否满足指定的标准
3. 处理答案验证的相关逻辑
"""

from onyx.llm.exceptions import GenAIDisabledException
from onyx.llm.factory import get_default_llms
from onyx.llm.utils import dict_based_prompt_to_langchain_prompt
from onyx.llm.utils import message_to_string
from onyx.prompts.answer_validation import ANSWER_VALIDITY_PROMPT
from onyx.utils.logger import setup_logger
from onyx.utils.timing import log_function_time

logger = setup_logger()


@log_function_time()
def get_answer_validity(
    query: str,
    answer: str,
) -> bool:
    """
    验证给定查询的答案是否有效。

    参数:
        query (str): 用户的原始查询
        answer (str): 需要验证的答案

    返回:
        bool: 答案是否有效的布尔值
    """
    def _get_answer_validation_messages(
        query: str, answer: str
    ) -> list[dict[str, str]]:
        """
        生成用于答案验证的消息列表。

        参数:
            query (str): 用户查询
            answer (str): 需要验证的答案

        返回:
            list[dict[str, str]]: 包含验证提示的消息列表
        """
        # Below COT block is unused, keeping for reference. Chain of Thought here significantly increases the time to
        # answer, we can get most of the way there but just having the model evaluate each individual condition with
        # a single True/False.
        # 以下思维链(COT)代码块未使用，仅作参考。在这里使用思维链会显著增加回答时间，我们可以通过让模型用单个True/False评估每个条件来实现大部分功能。

        messages = [
            {
                "role": "user",
                "content": ANSWER_VALIDITY_PROMPT.format(
                    user_query=query, llm_answer=answer
                ),
            },
        ]

        return messages

    def _extract_validity(model_output: str) -> bool:
        """
        从模型输出中提取验证结果。

        参数:
            model_output (str): 模型的输出文本

        返回:
            bool: True表示有效，False表示无效
        """
        if model_output.strip().strip("```").strip().split()[-1].lower() == "invalid":
            return False
        # If something is wrong, let's not toss away the answer
        # 如果出现问题，让我们不要丢弃答案
        return True  

    try:
        llm, _ = get_default_llms()
    except GenAIDisabledException:
        return True

    if not answer:
        return False

    messages = _get_answer_validation_messages(query, answer)
    filled_llm_prompt = dict_based_prompt_to_langchain_prompt(messages)
    model_output = message_to_string(llm.invoke(filled_llm_prompt))
    logger.debug(model_output)

    validity = _extract_validity(model_output)

    return validity
