"""
此模块包含了支持工具调用的提示模板。
这些提示模板目前未在主流程或任何配置中使用。
当前一代LLM对于这项任务的可靠性还不够。
"""

# These prompts are to support tool calling. Currently not used in the main flow or via any configs
# The current generation of LLM is too unreliable for this task.
# 这些提示用于支持工具调用。目前未在主流程或通过任何配置使用。
# 当前一代的LLM对于这项任务的可靠性还不够。

# Onyx retrieval call as a tool option
# Onyx检索调用作为工具选项
DANSWER_TOOL_NAME = "Current Search"
DANSWER_TOOL_DESCRIPTION = (
    "A search tool that can find information on any topic "
    "including up to date and proprietary knowledge."
)
# 一个可以查找任何主题信息的搜索工具，包括最新的和专有的知识

# Tool calling format inspired from LangChain
# 工具调用格式灵感来自LangChain
TOOL_TEMPLATE = """
TOOLS
------
You can use tools to look up information that may be helpful in answering the user's \
original question. The available tools are:

{tool_overviews}

RESPONSE FORMAT INSTRUCTIONS
----------------------------
When responding to me, please output a response in one of two formats:

**Option 1:**
Use this if you want to use a tool. Markdown code snippet formatted in the following schema:

```json
{{
    "action": string, \\ The action to take. {tool_names}
    "action_input": string \\ The input to the action
}}
```

**Option #2:**
Use this if you want to respond directly to the user. Markdown code snippet formatted in the following schema:

```json
{{
    "action": "Final Answer",
    "action_input": string \\ You should put what you want to return to use here
}}
```
"""
# 定义了工具使用的模板，包括工具列表和响应格式说明

# For the case where the user has not configured any tools to call, but still using the tool-flow
# expected format
# 无工具情况下的提示格式
TOOL_LESS_PROMPT = """
Respond with a markdown code snippet in the following schema:

```json
{{
    "action": "Final Answer",
    "action_input": string \\ You should put what you want to return to use here
}}
```
"""

# Second part of the prompt to include the user query
# 用户输入的提示格式
USER_INPUT = """
USER'S INPUT
--------------------
Here is the user's input \
(remember to respond with a markdown code snippet of a json blob with a single action, and NOTHING else):

{user_input}
"""

# After the tool call, this is the following message to get a final answer
# Tools are not chained currently, the system must provide an answer after calling a tool
# 工具调用后的后续提示
TOOL_FOLLOWUP = """
TOOL RESPONSE:
---------------------
{tool_output}

USER'S INPUT
--------------------
Okay, so what is the response to my last comment? If using information obtained from the tools you must \
mention it explicitly without mentioning the tool names - I have forgotten all TOOL RESPONSES!
If the tool response is not useful, ignore it completely.
{optional_reminder}{hint}
IMPORTANT! You MUST respond with a markdown code snippet of a json blob with a single action, and NOTHING else.
"""

# If no tools were used, but retrieval is enabled, then follow up with this message to get the final answer
# 无工具但启用检索时的后续提示
TOOL_LESS_FOLLOWUP = """
Refer to the following documents when responding to my final query. Ignore any documents that are not relevant.

CONTEXT DOCUMENTS:
---------------------
{context_str}

FINAL QUERY:
--------------------
{user_query}

{hint_text}
"""
