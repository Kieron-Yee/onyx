"""
This module contains constant patterns and strings used in prompt engineering and text processing.
此模块包含在提示工程和文本处理中使用的常量模式和字符串。
"""

# Separator pattern for general use, matches Langchain's separator length
# 通用分隔符模式，长度与Langchain的分隔符相同
GENERAL_SEP_PAT = "--------------"

# Pattern for code block formatting
# 代码块格式化模式
CODE_BLOCK_PAT = "```\n{}\n```"

# Triple backtick constant for code blocks
# 代码块的三重反引号常量
TRIPLE_BACKTICK = "```"

# Patterns for different sections in prompt responses
# 提示响应中不同部分的模式
QUESTION_PAT = "Query:"  # 查询模式
FINAL_QUERY_PAT = "Final Query:"  # 最终查询模式
THOUGHT_PAT = "Thought:"  # 思考过程模式
ANSWER_PAT = "Answer:"  # 答案模式
ANSWERABLE_PAT = "Answerable:"  # 可回答状态模式
FINAL_ANSWER_PAT = "Final Answer:"  # 最终答案模式
QUOTE_PAT = "Quote:"  # 引用模式
QUOTES_PAT_PLURAL = "Quotes:"  # 多个引用模式
INVALID_PAT = "Invalid:"  # 无效标记模式

# Key for source references in responses
# 响应中源引用的键名
SOURCES_KEY = "sources"

# Default statement for context filtering
# 上下文过滤的默认声明
DEFAULT_IGNORE_STATEMENT = " Ignore any context documents that are not relevant."
