from onyx.prompts.constants import GENERAL_SEP_PAT
from onyx.prompts.constants import QUESTION_PAT

# Prompt翻译: 使用格式[1], [2], [3]等内联引用相关陈述以引用文档编号，
# 不要在末尾提供参考部分和任何引用后的链接
REQUIRE_CITATION_STATEMENT = """
Cite relevant statements INLINE using the format [1], [2], [3], etc to reference the document number, \
DO NOT provide a reference section at the end and DO NOT provide any links following the citations.
""".rstrip()

# Prompt翻译: 即使聊天历史中有示例，也不要提供任何引用
NO_CITATION_STATEMENT = """
Do not provide any citations even if there are examples in the chat history.
""".rstrip()

# Prompt翻译: 记住使用[1], [2], [3]等格式进行内联引用
CITATION_REMINDER = """
Remember to provide inline citations in the format [1], [2], [3], etc.
"""

ADDITIONAL_INFO = "\n\nAdditional Information:\n\t- {datetime_info}."

# Prompt翻译: 
# 1. 参考以下上下文文档来回答我
# 2. 上下文部分
# 3. 任务提示
# 4. 用户问题
CHAT_USER_PROMPT = f"""
Refer to the following context documents when responding to me.{{optional_ignore_statement}}
CONTEXT:
{GENERAL_SEP_PAT}
{{context_docs_str}}
{GENERAL_SEP_PAT}

{{task_prompt}}

{QUESTION_PAT.upper()}
{{user_query}}
""".strip()

CHAT_USER_CONTEXT_FREE_PROMPT = f"""
{{task_prompt}}

{QUESTION_PAT.upper()}
{{user_query}}
""".strip()

# Design considerations for the below:
# - In case of uncertainty, favor yes search so place the "yes" sections near the start of the
#   prompt and after the no section as well to deemphasize the no section
# - Conversation history can be a lot of tokens, make sure the bulk of the prompt is at the start
#   or end so the middle history section is relatively less paid attention to than the main task
# - Works worse with just a simple yes/no, seems asking it to produce "search" helps a bit, can
#   consider doing COT for this and keep it brief, but likely only small gains.
SKIP_SEARCH = "Skip Search"
YES_SEARCH = "Yes Search"

# Prompt翻译:
# 根据对话历史和后续查询，确定系统是否应该调用外部搜索工具以更好地回答最新的用户输入。
# 默认响应是进行搜索。
# 
# 在以下情况下回复"跳过搜索":
# - 聊天历史中有足够的信息完全准确地回答查询，且额外信息或细节提供的价值很小
# - 查询是某种不需要额外信息处理的请求
AGGRESSIVE_SEARCH_TEMPLATE = f"""
Given the conversation history and a follow up query, determine if the system should call \
an external search tool to better answer the latest user input.
Your default response is {YES_SEARCH}.

Respond "{SKIP_SEARCH}" if either:
- There is sufficient information in chat history to FULLY and ACCURATELY answer the query AND \
additional information or details would provide little or no value.
- The query is some form of request that does not require additional information to handle.

Conversation History:
{GENERAL_SEP_PAT}
{{chat_history}}
{GENERAL_SEP_PAT}

If you are at all unsure, respond with {YES_SEARCH}.
Respond with EXACTLY and ONLY "{YES_SEARCH}" or "{SKIP_SEARCH}"

Follow Up Input:
{{final_query}}
""".strip()

# TODO, templatize this so users don't need to make code changes to use this
# Prompt翻译: 
# 你是一个关键系统的专家。根据对话历史和后续查询，确定系统是否应该调用外部搜索工具以更好地回答最新的用户输入。
# 默认响应是进行搜索。如果你有丝毫不确定，就回复"进行搜索"。
# 
# 在以下情况为真时回复"跳过搜索":
# - 聊天历史中有足够的信息完全准确地回答查询
# - 查询是某种不需要额外信息处理的请求
# - 你对问题完全确定，答案和问题都没有歧义
AGGRESSIVE_SEARCH_TEMPLATE_LLAMA2 = f"""
You are an expert of a critical system. Given the conversation history and a follow up query, \
determine if the system should call an external search tool to better answer the latest user input.

Your default response is {YES_SEARCH}.
If you are even slightly unsure, respond with {YES_SEARCH}.

Respond "{SKIP_SEARCH}" if any of these are true:
- There is sufficient information in chat history to FULLY and ACCURATELY answer the query.
- The query is some form of request that does not require additional information to handle.
- You are absolutely sure about the question and there is no ambiguity in the answer or question.

Conversation History:
{GENERAL_SEP_PAT}
{{chat_history}}
{GENERAL_SEP_PAT}

Respond with EXACTLY and ONLY "{YES_SEARCH}" or "{SKIP_SEARCH}"

Follow Up Input:
{{final_query}}
""".strip()

# Prompt翻译:
# 根据对话历史和后续查询，确定系统是否应该调用外部搜索工具以更好地回答最新的用户输入。
# 
# 在以下情况下回复"进行搜索":
# - 具体细节或额外知识可能会导致更好的答案。
# - 有新的或未知的术语，或者不确定用户指的是什么。
# - 如果阅读之前引用或提到的文档可能有用。
# 
# 在以下情况下回复"跳过搜索":
# - 聊天历史中有足够的信息完全准确地回答查询，且额外信息或细节提供的价值很小。
# - 查询是某种不需要额外信息处理的任务。
REQUIRE_SEARCH_SINGLE_MSG = f"""
Given the conversation history and a follow up query, determine if the system should call \
an external search tool to better answer the latest user input.

Respond "{YES_SEARCH}" if:
- Specific details or additional knowledge could lead to a better answer.
- There are new or unknown terms, or there is uncertainty what the user is referring to.
- If reading a document cited or mentioned previously may be useful.

Respond "{SKIP_SEARCH}" if:
- There is sufficient information in chat history to FULLY and ACCURATELY answer the query
and additional information or details would provide little or no value.
- The query is some task that does not require additional information to handle.

Conversation History:
{GENERAL_SEP_PAT}
{{chat_history}}
{GENERAL_SEP_PAT}

Even if the topic has been addressed, if more specific details could be useful, \
respond with "{YES_SEARCH}".
If you are unsure, respond with "{YES_SEARCH}".

Respond with EXACTLY and ONLY "{YES_SEARCH}" or "{SKIP_SEARCH}"

Follow Up Input:
{{final_query}}
""".strip()

# Prompt翻译:
# 根据以下对话和后续输入，将后续输入重新表述为一个简短的独立查询（捕获之前消息中的任何相关上下文）以用于向量存储。
# 重要提示：将查询编辑得尽可能简洁。用主要关键词而不是完整句子来回应。
# 如果话题有明显变化，请忽略之前的消息。
# 删除任何与检索任务无关的信息。
# 如果后续消息是错误或代码片段，请准确重复相同的输入。
HISTORY_QUERY_REPHRASE = f"""
Given the following conversation and a follow up input, rephrase the follow up into a SHORT, \
standalone query (which captures any relevant context from previous messages) for a vectorstore.
IMPORTANT: EDIT THE QUERY TO BE AS CONCISE AS POSSIBLE. Respond with a short, compressed phrase \
with mainly keywords instead of a complete sentence.
If there is a clear change in topic, disregard the previous messages.
Strip out any information that is not relevant for the retrieval task.
If the follow up message is an error or code snippet, repeat the same input back EXACTLY.

Chat History:
{GENERAL_SEP_PAT}
{{chat_history}}
{GENERAL_SEP_PAT}

Follow Up Input: {{question}}
Standalone question (Respond with only the short combined query):
""".strip()

# Prompt翻译:
# 根据以下对话和后续输入，将后续输入重新表述为一个简短的独立查询，适合用于互联网搜索引擎。
# 重要提示：如果特定查询可能限制结果，请保持广泛。
# 如果广泛查询可能产生太多结果，请使其详细。
# 如果话题有明显变化，请确保查询准确反映新话题。
# 删除任何与互联网搜索无关的信息。
INTERNET_SEARCH_QUERY_REPHRASE = f"""
Given the following conversation and a follow up input, rephrase the follow up into a SHORT, \
standalone query suitable for an internet search engine.
IMPORTANT: If a specific query might limit results, keep it broad. \
If a broad query might yield too many results, make it detailed.
If there is a clear change in topic, ensure the query reflects the new topic accurately.
Strip out any information that is not relevant for the internet search.

Chat History:
{GENERAL_SEP_PAT}
{{chat_history}}
{GENERAL_SEP_PAT}

Follow Up Input: {{question}}
Internet Search Query (Respond with a detailed and specific query):
""".strip()

# The below prompts are retired
NO_SEARCH = "No Search"

# Prompt翻译:
# 你是一个大型语言模型，你唯一的工作是确定系统是否应该调用外部搜索工具来回答用户的最后一条消息。
#
# 在以下情况下回复"不搜索":
# - 聊天历史中有足够的信息完全回答用户查询
# - LLM有足够的知识完全回答用户查询
# - 用户查询不依赖任何特定知识
#
# 在以下情况下回复"进行搜索":
# - 关于实体、流程、问题或其他任何内容的额外知识可能会导致更好的答案
# - 对用户指的是什么存在不确定性
REQUIRE_SEARCH_SYSTEM_MSG = f"""
You are a large language model whose only job is to determine if the system should call an \
external search tool to be able to answer the user's last message.

Respond with "{NO_SEARCH}" if:
- there is sufficient information in chat history to fully answer the user query
- there is enough knowledge in the LLM to fully answer the user query
- the user query does not rely on any specific knowledge

Respond with "{YES_SEARCH}" if:
- additional knowledge about entities, processes, problems, or anything else could lead to a better answer.
- there is some uncertainty what the user is referring to

Respond with EXACTLY and ONLY "{YES_SEARCH}" or "{NO_SEARCH}"
"""

# Prompt翻译: 
# 提示：严格回复"进行搜索"或"不搜索"
REQUIRE_SEARCH_HINT = f"""
Hint: respond with EXACTLY {YES_SEARCH} or {NO_SEARCH}"
""".strip()

# Prompt翻译:
# 根据对话（在人类和助手之间）和人类的最后一条消息，
# 将最后一条消息重写为简洁的独立查询，该查询捕获了前面消息中所需/相关的上下文。
# 这个问题必须对语义（自然语言）搜索引擎有用。
QUERY_REPHRASE_SYSTEM_MSG = """
Given a conversation (between Human and Assistant) and a final message from Human, \
rewrite the last message to be a concise standalone query which captures required/relevant \
context from previous messages. This question must be useful for a semantic (natural language) \
search engine.
""".strip()

# Prompt翻译:
# 帮我将这最后一条消息重写为独立查询，如果相关的话要考虑对话的历史消息。
# 这个查询用于语义搜索引擎检索文档。
# 你必须只返回重写的查询，不要返回其他任何内容。
# 重要提示：搜索引擎无法访问对话历史！
QUERY_REPHRASE_USER_MSG = """
Help me rewrite this final message into a standalone query that takes into consideration the \
past messages of the conversation IF relevant. This query is used with a semantic search engine to \
retrieve documents. You must ONLY return the rewritten query and NOTHING ELSE. \
IMPORTANT, the search engine does not have access to the conversation history!

Query:
{final_query}
""".strip()

# Prompt翻译:
# 根据以下对话，为对话提供一个简短的名称。
# 重要提示：尽量不要使用超过5个词，使其尽可能简洁。
# 重点关注关键词以传达对话的主题。
CHAT_NAMING = f"""
Given the following conversation, provide a SHORT name for the conversation.{{language_hint_or_empty}}
IMPORTANT: TRY NOT TO USE MORE THAN 5 WORDS, MAKE IT AS CONCISE AS POSSIBLE.
Focus the name on the important keywords to convey the topic of the conversation.

Chat History:
{GENERAL_SEP_PAT}
{{chat_history}}
{GENERAL_SEP_PAT}

Based on the above, what is a short name to convey the topic of the conversation?
""".strip()
