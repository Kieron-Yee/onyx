"""
此模块包含用于直接问答系统的提示词模板。
主要功能：
1. 定义了系统提示词、任务提示词等基础模板
2. 提供了JSON格式和引用格式的响应模板
3. 支持上下文文档和对话历史的提示词组装
"""

# The following prompts are used for the initial response before a chat history exists
# It is used also for the one shot direct QA flow
# 以下提示词用于在聊天历史记录存在之前的初始响应
# 也用于一次性直接问答流程
import json

from onyx.prompts.constants import DEFAULT_IGNORE_STATEMENT
from onyx.prompts.constants import FINAL_QUERY_PAT
from onyx.prompts.constants import GENERAL_SEP_PAT
from onyx.prompts.constants import QUESTION_PAT
from onyx.prompts.constants import THOUGHT_PAT

# 一次性问答的系统提示词
# 你是一个不断学习和改进的问答系统。
# 你可以处理和理解大量文本，并利用这些知识为各种查询提供准确和详细的答案。
ONE_SHOT_SYSTEM_PROMPT = """
You are a question answering system that is constantly learning and improving.
You can process and comprehend vast amounts of text and utilize this knowledge to provide \
accurate and detailed answers to diverse queries.
""".strip()

# 一次性问答的任务提示词
# 回答下面的最终查询，考虑上述相关上下文。
# 忽略与查询无关的任何提供的上下文。
ONE_SHOT_TASK_PROMPT = """
Answer the final query below taking into account the context above where relevant. \
Ignore any provided context that is not relevant to the query.
""".strip()

# 用于弱模型的系统提示词
# 使用以下参考文档回答用户查询。
WEAK_MODEL_SYSTEM_PROMPT = """
Respond to the user query using the following reference document.
""".lstrip()

# 用于弱模型的任务提示词
# 根据上面的参考文档回答下面的用户查询。
WEAK_MODEL_TASK_PROMPT = """
Answer the user query below based on the reference document above.
"""

# 要求JSON格式响应的提示词
# 你始终只能使用包含答案和支持答案引用的JSON格式进行响应。
REQUIRE_JSON = """
You ALWAYS responds with ONLY a JSON containing an answer and quotes that support the answer.
""".strip()

# JSON格式响应的提示词帮助提示
# 提示：使答案尽可能详细，并以JSON格式响应！
# 引用必须是提供的文档中的精确子字符串！
JSON_HELPFUL_HINT = """
Hint: Make the answer as DETAILED as possible and respond in JSON format! \
Quotes MUST be EXACT substrings from provided documents!
""".strip()

# 上下文文档块
CONTEXT_BLOCK = f"""
REFERENCE DOCUMENTS:
{GENERAL_SEP_PAT}
{{context_docs_str}}
{GENERAL_SEP_PAT}
"""

# 对话历史块
HISTORY_BLOCK = f"""
CONVERSATION HISTORY:
{GENERAL_SEP_PAT}
{{history_str}}
{GENERAL_SEP_PAT}
"""

# 这是一个双重转义的空样本JSON
# This has to be doubly escaped due to json containing { } which are also used for format strings
EMPTY_SAMPLE_JSON = {
    "answer": "Place your final answer here. It should be as DETAILED and INFORMATIVE as possible.",
    "quotes": [
        "each quote must be UNEDITED and EXACTLY as shown in the context documents!",
        "HINT, quotes are not shown to the user!",
    ],
}

# 默认的JSON提示词，可以引用多个文档并提供答案和引用
# Default json prompt which can reference multiple docs and provide answer + quotes
# system_like_header 类似于系统消息，可以由用户提供或默认为 QA_HEADER
# context/history blocks 是上下文文档和对话历史记录，可以为空
# task prompt 是提示词的任务消息，可以为空，没有默认值
JSON_PROMPT = f"""
{{system_prompt}}
{REQUIRE_JSON}
{{context_block}}{{history_block}}
{{task_prompt}}

SAMPLE RESPONSE:
```
{{{json.dumps(EMPTY_SAMPLE_JSON)}}}
```

{FINAL_QUERY_PAT.upper()}
{{user_query}}

{JSON_HELPFUL_HINT}
{{language_hint_or_none}}
""".strip()

# 类似于聊天流程，但可以包含“对话历史”块
# similar to the chat flow, but with the option of including a
# "conversation history" block
CITATIONS_PROMPT = f"""
Refer to the following context documents when responding to me.{DEFAULT_IGNORE_STATEMENT}

CONTEXT:
{GENERAL_SEP_PAT}
{{context_docs_str}}
{GENERAL_SEP_PAT}

{{history_block}}{{task_prompt}}

{QUESTION_PAT.upper()}
{{user_query}}
"""

# 用于工具调用的引用提示词，文档在单独的“工具”消息中
# with tool calling, the documents are in a separate "tool" message
# 注意：需要添加额外的关于“直奔主题”的行，因为
# OpenAI 的工具调用模型往往更冗长
# NOTE: need to add the extra line about "getting right to the point" since the
# tool calling models from OpenAI tend to be more verbose
CITATIONS_PROMPT_FOR_TOOL_CALLING = f"""
Refer to the provided context documents when responding to me.{DEFAULT_IGNORE_STATEMENT} \
You should always get right to the point, and never use extraneous language.

{{history_block}}{{task_prompt}}

{QUESTION_PAT.upper()}
{{user_query}}
"""

# 这仅用于用户指定其自己的提示词的可视化
# 实际流程不是这样工作的
# This is only for visualization for the users to specify their own prompts
# The actual flow does not work like this
PARAMATERIZED_PROMPT = f"""
{{system_prompt}}

CONTEXT:
{GENERAL_SEP_PAT}
{{context_docs_str}}
{GENERAL_SEP_PAT}

{{task_prompt}}

{QUESTION_PAT.upper()} {{user_query}}
RESPONSE:
""".strip()

PARAMATERIZED_PROMPT_WITHOUT_CONTEXT = f"""
{{system_prompt}}

{{task_prompt}}

{QUESTION_PAT.upper()} {{user_query}}
RESPONSE:
""".strip()

# 当前禁用，不能使用这个
# CURRENTLY DISABLED, CANNOT USE THIS ONE
# 默认的链式思维风格的JSON提示词，使用多个文档
# Default chain-of-thought style json prompt which uses multiple docs
# 这个有一个部分用于LLM输出一些非答案的“思考”
# COT（链式思维）流程基本上
# This one has a section for the LLM to output some non-answer "thoughts"
# COT (chain-of-thought) flow basically
COT_PROMPT = f"""
{ONE_SHOT_SYSTEM_PROMPT}

CONTEXT:
{GENERAL_SEP_PAT}
{{context_docs_str}}
{GENERAL_SEP_PAT}

You MUST respond in the following format:
```
{THOUGHT_PAT} Use this section as a scratchpad to reason through the answer.

{{{json.dumps(EMPTY_SAMPLE_JSON)}}}
```

{QUESTION_PAT.upper()} {{user_query}}
{JSON_HELPFUL_HINT}
{{language_hint_or_none}}
""".strip()

# 使用以下内容轻松查看提示词
# User the following for easy viewing of prompts
if __name__ == "__main__":
    print(JSON_PROMPT)  # Default prompt used in the Onyx UI flow
