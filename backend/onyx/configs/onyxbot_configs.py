"""
This file contains the configuration settings for the Onyx Slack Bot.
It manages various aspects of the bot's behavior including API retries, context handling,
document display, feedback systems, and rate limiting.
这个文件包含 Onyx Slack 机器人的配置设置。
它管理机器人的各个方面，包括 API 重试、上下文处理、文档显示、反馈系统和速率限制。
"""

import os

#####
# Onyx Slack Bot Configs
# Onyx Slack 机器人配置
#####

# Number of retries for Onyx Slack bot API calls
# API 调用失败时的重试次数
# 控制在调用 Slack API 失败时最多重试多少次
DANSWER_BOT_NUM_RETRIES = int(os.environ.get("DANSWER_BOT_NUM_RETRIES", "5"))

# How much of the available input context can be used for thread context
# 用于存储对话上下文的最大输入上下文比例
# 在 3072 个 token 中，保留 2/3 用于存储对话历史
MAX_THREAD_CONTEXT_PERCENTAGE = 512 * 2 / 3072

# Number of docs to display in "Reference Documents" 
# 在"参考文档"中显示的文档数量
DANSWER_BOT_NUM_DOCS_TO_DISPLAY = int(
    os.environ.get("DANSWER_BOT_NUM_DOCS_TO_DISPLAY", "5")
)

# If the LLM fails to answer, Onyx can still show the "Reference Documents"
# 如果 LLM 无法回答，Onyx 仍然可以显示"参考文档"
DANSWER_BOT_DISABLE_DOCS_ONLY_ANSWER = os.environ.get(
    "DANSWER_BOT_DISABLE_DOCS_ONLY_ANSWER", ""
).lower() not in ["false", ""]

# When Onyx is considering a message, what emoji does it react with
# 当Onyx处理消息时使用的表情符号
DANSWER_REACT_EMOJI = os.environ.get("DANSWER_REACT_EMOJI") or "eyes"

# When User needs more help, what should the emoji be  
# 当用户需要更多帮助时使用的表情符号
DANSWER_FOLLOWUP_EMOJI = os.environ.get("DANSWER_FOLLOWUP_EMOJI") or "sos"

# What kind of message should be shown when someone gives an AI answer feedback to OnyxBot
# 当有人对OnyxBot的AI回答提供反馈时显示的消息类型
# Private: Only visible to user clicking the feedback
# 私密: 只对点击反馈的用户可见
# Anonymous: Public but anonymous  
# 匿名: 公开但匿名
# Public: Visible with the user name who submitted the feedback
# 公开: 显示提交反馈的用户名
DANSWER_BOT_FEEDBACK_VISIBILITY = (
    os.environ.get("DANSWER_BOT_FEEDBACK_VISIBILITY") or "private"
)

# Should OnyxBot send an apology message if it's not able to find an answer
# That way the user isn't confused as to why OnyxBot reacted but then said nothing
# Off by default to be less intrusive (don't want to give a notif that just says we couldnt help)
# 如果 OnyxBot 无法找到答案，是否应该发送道歉消息
# 这样用户就不会困惑为什么 OnyxBot 有反应但没有说任何话
# 默认关闭以减少干扰（不希望发送仅表示我们无法帮助的通知）
NOTIFY_SLACKBOT_NO_ANSWER = (
    os.environ.get("NOTIFY_SLACKBOT_NO_ANSWER", "").lower() == "true"
)

# Mostly for debugging purposes but it's for explaining what went wrong
# if OnyxBot couldn't find an answer
# 主要用于调试目的，但也用于解释 OnyxBot 无法找到答案的原因
DANSWER_BOT_DISPLAY_ERROR_MSGS = os.environ.get(
    "DANSWER_BOT_DISPLAY_ERROR_MSGS", ""
).lower() not in [
    "false",
    "",
]

# Default is only respond in channels that are included by a slack config set in the UI
# 默认情况下只在 UI 中配置的 Slack 频道中响应
# 控制机器人是否可以在所有频道中响应消息，默认为 false
DANSWER_BOT_RESPOND_EVERY_CHANNEL = (
    os.environ.get("DANSWER_BOT_RESPOND_EVERY_CHANNEL", "").lower() == "true"
)

# Maximum Questions Per Minute, Default Uncapped
# 每分钟最大问题数，默认无上限
# 控制机器人每分钟可以处理的最大问题数量
DANSWER_BOT_MAX_QPM = int(os.environ.get("DANSWER_BOT_MAX_QPM") or 0) or None

# Maximum time to wait when a question is queued
# 问题排队时的最大等待时间（秒）
# 当系统繁忙时，问题最多可以等待多长时间
DANSWER_BOT_MAX_WAIT_TIME = int(os.environ.get("DANSWER_BOT_MAX_WAIT_TIME") or 180)

# Time (in minutes) after which a Slack message is sent to the user to remind him to give feedback.
# Set to 0 to disable it (default)
# 在指定时间（分钟）后向用户发送 Slack 消息提醒其提供反馈
# 设置为 0 以禁用（默认）
DANSWER_BOT_FEEDBACK_REMINDER = int(
    os.environ.get("DANSWER_BOT_FEEDBACK_REMINDER") or 0
)

# Set to True to rephrase the Slack users messages
# 设置为 True 时会重新措辞用户的消息
# 控制是否对用户输入的问题进行重新措辞后再处理
DANSWER_BOT_REPHRASE_MESSAGE = (
    os.environ.get("DANSWER_BOT_REPHRASE_MESSAGE", "").lower() == "true"
)

# DANSWER_BOT_RESPONSE_LIMIT_PER_TIME_PERIOD is the number of responses OnyxBot can send in a given time period
# 在指定时间段内 OnyxBot 可以发送的最大响应数
# 用于限制机器人在一定时间内的总响应次数，防止过度使用
DANSWER_BOT_RESPONSE_LIMIT_PER_TIME_PERIOD = int(
    os.environ.get("DANSWER_BOT_RESPONSE_LIMIT_PER_TIME_PERIOD", "5000")
)

# DANSWER_BOT_RESPONSE_LIMIT_TIME_PERIOD_SECONDS is the number of seconds until the response limit is reset
# 响应限制重置的时间周期（秒）
# 定义多少秒后重置响应次数计数器
DANSWER_BOT_RESPONSE_LIMIT_TIME_PERIOD_SECONDS = int(
    os.environ.get("DANSWER_BOT_RESPONSE_LIMIT_TIME_PERIOD_SECONDS", "86400")
)
