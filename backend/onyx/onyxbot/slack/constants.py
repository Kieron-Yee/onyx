"""
This file contains constants used in Slack bot integration
此文件包含 Slack 机器人集成中使用的常量
"""

from enum import Enum

# Block action IDs for different feedback interactions in Slack
# Slack 中不同反馈交互的块动作 ID
LIKE_BLOCK_ACTION_ID = "feedback-like"  # 点赞反馈动作ID
DISLIKE_BLOCK_ACTION_ID = "feedback-dislike"  # 点踩反馈动作ID
CONTINUE_IN_WEB_UI_ACTION_ID = "continue-in-web-ui"  # 在网页界面继续操作的动作ID
FEEDBACK_DOC_BUTTON_BLOCK_ACTION_ID = "feedback-doc-button"  # 文档反馈按钮动作ID
IMMEDIATE_RESOLVED_BUTTON_ACTION_ID = "immediate-resolved-button"  # 立即解决按钮动作ID
FOLLOWUP_BUTTON_ACTION_ID = "followup-button"  # 跟进按钮动作ID
FOLLOWUP_BUTTON_RESOLVED_ACTION_ID = "followup-resolved-button"  # 跟进已解决按钮动作ID
VIEW_DOC_FEEDBACK_ID = "view-doc-feedback"  # 查看文档反馈ID
GENERATE_ANSWER_BUTTON_ACTION_ID = "generate-answer-button"  # 生成答案按钮动作ID


class FeedbackVisibility(str, Enum):
    """
    Enum class defining visibility options for feedback
    定义反馈可见性选项的枚举类
    
    Attributes:
        PRIVATE: 私密的，仅对特定用户可见
        ANONYMOUS: 匿名的，不显示提供反馈的用户信息
        PUBLIC: 公开的，对所有用户可见
    """
    PRIVATE = "private"
    ANONYMOUS = "anonymous"
    PUBLIC = "public"
