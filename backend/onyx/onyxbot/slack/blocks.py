"""
这个文件主要用于处理 Slack 消息块(blocks)的相关功能。
主要包含:
- 构建反馈提醒消息块
- 构建文档反馈块
- 构建问答响应块
- 构建引用和来源块
- 处理消息格式化和显示
"""

# 导入标准库
from datetime import datetime

# 导入第三方库
import pytz
import timeago  # type: ignore
# Slack SDK相关导入
from slack_sdk.models.blocks import ActionsBlock
from slack_sdk.models.blocks import Block
from slack_sdk.models.blocks import ButtonElement
from slack_sdk.models.blocks import ContextBlock
from slack_sdk.models.blocks import DividerBlock
from slack_sdk.models.blocks import HeaderBlock
from slack_sdk.models.blocks import Option
from slack_sdk.models.blocks import RadioButtonsElement
from slack_sdk.models.blocks import SectionBlock
from slack_sdk.models.blocks.basic_components import MarkdownTextObject
from slack_sdk.models.blocks.block_elements import ImageElement

# 内部模块导入
from onyx.chat.models import ChatOnyxBotResponse
from onyx.configs.app_configs import DISABLE_GENERATIVE_AI
from onyx.configs.app_configs import WEB_DOMAIN
from onyx.configs.constants import DocumentSource
from onyx.configs.constants import SearchFeedbackType
from onyx.configs.onyxbot_configs import DANSWER_BOT_NUM_DOCS_TO_DISPLAY
from onyx.context.search.models import SavedSearchDoc
from onyx.db.chat import get_chat_session_by_message_id
from onyx.db.engine import get_session_with_tenant
from onyx.db.models import ChannelConfig
from onyx.onyxbot.slack.constants import CONTINUE_IN_WEB_UI_ACTION_ID
from onyx.onyxbot.slack.constants import DISLIKE_BLOCK_ACTION_ID
from onyx.onyxbot.slack.constants import FEEDBACK_DOC_BUTTON_BLOCK_ACTION_ID
from onyx.onyxbot.slack.constants import FOLLOWUP_BUTTON_ACTION_ID
from onyx.onyxbot.slack.constants import FOLLOWUP_BUTTON_RESOLVED_ACTION_ID
from onyx.onyxbot.slack.constants import IMMEDIATE_RESOLVED_BUTTON_ACTION_ID
from onyx.onyxbot.slack.constants import LIKE_BLOCK_ACTION_ID
from onyx.onyxbot.slack.formatting import format_slack_message
from onyx.onyxbot.slack.icons import source_to_github_img_link
from onyx.onyxbot.slack.models import SlackMessageInfo
from onyx.onyxbot.slack.utils import build_continue_in_web_ui_id
from onyx.onyxbot.slack.utils import build_feedback_id
from onyx.onyxbot.slack.utils import remove_slack_text_interactions
from onyx.onyxbot.slack.utils import translate_vespa_highlight_to_slack
from onyx.utils.text_processing import decode_escapes

# 定义文本块的最大长度
_MAX_BLURB_LEN = 45  # 文档标题的最大显示长度


def get_feedback_reminder_blocks(thread_link: str, include_followup: bool) -> Block:
    """
    构建反馈提醒消息块
    
    参数:
        thread_link: 消息线程链接
        include_followup: 是否包含后续跟进选项
    
    返回:
        Block: Slack消息块对象
    """
    text = (
        f"Please provide feedback on <{thread_link}|this answer>. "
        "This is essential to help us to improve the quality of the answers. "
        "Please rate it by clicking the `Helpful` or `Not helpful` button. "
    )
    if include_followup:
        text += "\n\nIf you need more help, click the `I need more help from a human!` button. "

    text += "\n\nThanks!"

    return SectionBlock(text=text)


def _split_text(text: str, limit: int = 3000) -> list[str]:
    """
    将长文本按照指定长度限制分割成多个块
    
    参数:
        text: 需要分割的文本
        limit: 每个块的最大长度限制，默认3000字符
        
    返回:
        list[str]: 分割后的文本块列表
    
    说明:
        - 会在空格处进行分割，避免单词被截断
        - 如果找不到合适的分割点，将强制在limit处分割
    """
    if len(text) <= limit:
        return [text]

    chunks = []
    while text:
        if len(text) <= limit:
            chunks.append(text)
            break

        # 在限制长度之前找到最近的空格，以避免分割单词
        split_at = text.rfind(" ", 0, limit)
        if split_at == -1:  # 没有找到空格，强制分割
            split_at = limit

        chunk = text[:split_at]
        chunks.append(chunk)
        text = text[split_at:].lstrip()  # 删除下一个块的前导空格

    return chunks


def _clean_markdown_link_text(text: str) -> str:
    """
    清理Markdown链接文本中的换行符
    
    参数:
        text: 需要清理的文本
    
    返回:
        str: 清理后的文本
    """
    # 删除文本中的所有换行符
    return text.replace("\n", " ").strip()


def _build_qa_feedback_block(
    message_id: int, feedback_reminder_id: str | None = None
) -> Block:
    """
    构建问答反馈块，包含有用和无用两个按钮
    
    参数:
        message_id: 消息ID
        feedback_reminder_id: 反馈提醒ID
        
    返回:
        Block: Slack消息块对象
    """
    return ActionsBlock(
        block_id=build_feedback_id(message_id),
        elements=[
            ButtonElement(
                action_id=LIKE_BLOCK_ACTION_ID,
                text="👍 Helpful",
                value=feedback_reminder_id,
            ),
            ButtonElement(
                action_id=DISLIKE_BLOCK_ACTION_ID,
                text="👎 Not helpful",
                value=feedback_reminder_id,
            ),
        ],
    )


def get_document_feedback_blocks() -> Block:
    """
    构建文档反馈块
    
    返回:
        Block: Slack消息块对象
    """
    return SectionBlock(
        text=(
            "- 'Up-Boost' if this document is a good source of information and should be "
            "shown more often.\n"
            "- 'Down-boost' if this document is a poor source of information and should be "
            "shown less often.\n"
            "- 'Hide' if this document is deprecated and should never be shown anymore."
        ),
        accessory=RadioButtonsElement(
            options=[
                Option(
                    text=":thumbsup: Up-Boost",
                    value=SearchFeedbackType.ENDORSE.value,
                ),
                Option(
                    text=":thumbsdown: Down-Boost",
                    value=SearchFeedbackType.REJECT.value,
                ),
                Option(
                    text=":x: Hide",
                    value=SearchFeedbackType.HIDE.value,
                ),
            ]
        ),
    )


def _build_doc_feedback_block(
    message_id: int,
    document_id: str,
    document_rank: int,
) -> ButtonElement:
    """
    构建文档反馈按钮
    
    参数:
        message_id: 消息ID
        document_id: 文档ID
        document_rank: 文档排名
        
    返回:
        ButtonElement: Slack按钮元素
    """
    feedback_id = build_feedback_id(message_id, document_id, document_rank)
    return ButtonElement(
        action_id=FEEDBACK_DOC_BUTTON_BLOCK_ACTION_ID,
        value=feedback_id,
        text="Give Feedback",
    )


def get_restate_blocks(
    msg: str,
    is_bot_msg: bool,
) -> list[Block]:
    """
    构建重新陈述问题的消息块
    
    参数:
        msg: 消息内容
        is_bot_msg: 是否为机器人消息
    
    返回:
        list[Block]: Slack消息块列表
    """
    # Only the slash command needs this context because the user doesn't see their own input
    if not is_bot_msg:
        return []

    return [
        HeaderBlock(text="Responding to the Query"),
        SectionBlock(text=f"```{msg}```"),
    ]


def _build_documents_blocks(
    documents: list[SavedSearchDoc],
    message_id: int | None,
    num_docs_to_display: int = DANSWER_BOT_NUM_DOCS_TO_DISPLAY,
) -> list[Block]:
    """
    构建文档消息块
    
    参数:
        documents: 文档列表
        message_id: 消息ID
        num_docs_to_display: 显示的文档数量
        
    返回:
        list[Block]: Slack消息块列表
    """
    header_text = (
        "Retrieved Documents" if DISABLE_GENERATIVE_AI else "Reference Documents"
    )
    seen_docs_identifiers = set()
    section_blocks: list[Block] = [HeaderBlock(text=header_text)]
    included_docs = 0
    for rank, d in enumerate(documents):
        if d.document_id in seen_docs_identifiers:
            continue
        seen_docs_identifiers.add(d.document_id)

        # Strip newlines from the semantic identifier for Slackbot formatting
        doc_sem_id = d.semantic_identifier.replace("\n", " ")
        if d.source_type == DocumentSource.SLACK.value:
            doc_sem_id = "#" + doc_sem_id

        used_chars = len(doc_sem_id) + 3
        match_str = translate_vespa_highlight_to_slack(d.match_highlights, used_chars)

        included_docs += 1

        header_line = f"{doc_sem_id}\n"
        if d.link:
            header_line = f"<{d.link}|{doc_sem_id}>\n"

        updated_at_line = ""
        if d.updated_at is not None:
            updated_at_line = (
                f"_Updated {timeago.format(d.updated_at, datetime.now(pytz.utc))}_\n"
            )

        body_text = f">{remove_slack_text_interactions(match_str)}"

        block_text = header_line + updated_at_line + body_text

        feedback: ButtonElement | dict = {}
        if message_id is not None:
            feedback = _build_doc_feedback_block(
                message_id=message_id,
                document_id=d.document_id,
                document_rank=rank,
            )

        section_blocks.append(
            SectionBlock(text=block_text, accessory=feedback),
        )

        section_blocks.append(DividerBlock())

        if included_docs >= num_docs_to_display:
            break

    return section_blocks


def _build_sources_blocks(
    cited_documents: list[tuple[int, SavedSearchDoc]],
    num_docs_to_display: int = DANSWER_BOT_NUM_DOCS_TO_DISPLAY,
) -> list[Block]:
    """
    构建引用和来源消息块
    
    参数:
        cited_documents: 引用的文档列表
        num_docs_to_display: 显示的文档数量
        
    返回:
        list[Block]: Slack消息块列表
    """
    if not cited_documents:
        return [
            SectionBlock(
                text="*Warning*: no sources were cited for this answer, so it may be unreliable 😔"
            )
        ]

    seen_docs_identifiers = set()
    section_blocks: list[Block] = [SectionBlock(text="*Sources:*")]
    included_docs = 0
    for citation_num, d in cited_documents:
        if d.document_id in seen_docs_identifiers:
            continue
        seen_docs_identifiers.add(d.document_id)

        doc_sem_id = d.semantic_identifier
        if d.source_type == DocumentSource.SLACK.value:
            # 由于历史原因，在切换到Slack语义标识符的构造方式之前
            if "#" not in doc_sem_id:
                doc_sem_id = "#" + doc_sem_id

        # 这是为了防止行溢出，如果溢出，图像会被放置在标题上方，看起来很糟糕
        doc_sem_id = (
            doc_sem_id[:_MAX_BLURB_LEN] + "..."
            if len(doc_sem_id) > _MAX_BLURB_LEN
            else doc_sem_id
        )

        owner_str = f"By {d.primary_owners[0]}" if d.primary_owners else None
        days_ago_str = (
            timeago.format(d.updated_at, datetime.now(pytz.utc))
            if d.updated_at
            else None
        )
        final_metadata_str = " | ".join(
            ([owner_str] if owner_str else [])
            + ([days_ago_str] if days_ago_str else [])
        )

        document_title = _clean_markdown_link_text(doc_sem_id)
        img_link = source_to_github_img_link(d.source_type)

        section_blocks.append(
            ContextBlock(
                elements=(
                    [
                        ImageElement(
                            image_url=img_link,
                            alt_text=f"{d.source_type.value} logo",
                        )
                    ]
                    if img_link
                    else []
                )
                + [
                    MarkdownTextObject(text=f"{document_title}")
                    if d.link == ""
                    else MarkdownTextObject(
                        text=f"*<{d.link}|[{citation_num}] {document_title}>*\n{final_metadata_str}"
                    ),
                ]
            )
        )

        if included_docs >= num_docs_to_display:
            break

    return section_blocks


def _priority_ordered_documents_blocks(
    answer: ChatOnyxBotResponse,
) -> list[Block]:
    """
    构建优先排序的文档消息块
    
    参数:
        answer: 聊天机器人的响应对象
        
    返回:
        list[Block]: Slack消息块列表
    """
    docs_response = answer.docs if answer.docs else None
    top_docs = docs_response.top_documents if docs_response else []
    llm_doc_inds = answer.llm_selected_doc_indices or []
    llm_docs = [top_docs[i] for i in llm_doc_inds]
    remaining_docs = [
        doc for idx, doc in enumerate(top_docs) if idx not in llm_doc_inds
    ]
    priority_ordered_docs = llm_docs + remaining_docs
    if not priority_ordered_docs:
        return []

    document_blocks = _build_documents_blocks(
        documents=priority_ordered_docs,
        message_id=answer.chat_message_id,
    )
    if document_blocks:
        document_blocks = [DividerBlock()] + document_blocks
    return document_blocks


def _build_citations_blocks(
    answer: ChatOnyxBotResponse,
) -> list[Block]:
    """
    构建引用消息块
    
    参数:
        answer: 聊天机器人的响应对象
        
    返回:
        list[Block]: Slack消息块列表
    """
    docs_response = answer.docs if answer.docs else None
    top_docs = docs_response.top_documents if docs_response else []
    citations = answer.citations or []
    cited_docs = []
    for citation in citations:
        matching_doc = next(
            (d for d in top_docs if d.document_id == citation.document_id),
            None,
        )
        if matching_doc:
            cited_docs.append((citation.citation_num, matching_doc))

    cited_docs.sort()
    citations_block = _build_sources_blocks(cited_documents=cited_docs)
    return citations_block


def _build_qa_response_blocks(
    answer: ChatOnyxBotResponse,
) -> list[Block]:
    """
    构建问答响应消息块
    
    参数:
        answer: 聊天机器人的响应对象
        
    返回:
        list[Block]: Slack消息块列表
    """
    retrieval_info = answer.docs
    if not retrieval_info:
        # 这种情况不应该发生，即使没有检索到文档，仍然会返回信息
        raise RuntimeError("Failed to retrieve docs, cannot answer question.")

    if DISABLE_GENERATIVE_AI:
        return []

    filter_block: Block | None = None
    if (
        retrieval_info.applied_time_cutoff
        or retrieval_info.recency_bias_multiplier > 1
        or retrieval_info.applied_source_filters
    ):
        filter_text = "Filters: "
        if retrieval_info.applied_source_filters:
            sources_str = ", ".join(
                [s.value for s in retrieval_info.applied_source_filters]
            )
            filter_text += f"`Sources in [{sources_str}]`"
            if (
                retrieval_info.applied_time_cutoff
                or retrieval_info.recency_bias_multiplier > 1
            ):
                filter_text += " and "
        if retrieval_info.applied_time_cutoff is not None:
            time_str = retrieval_info.applied_time_cutoff.strftime("%b %d, %Y")
            filter_text += f"`Docs Updated >= {time_str}` "
        if retrieval_info.recency_bias_multiplier > 1:
            if retrieval_info.applied_time_cutoff is not None:
                filter_text += "+ "
            filter_text += "`Prioritize Recently Updated Docs`"

        filter_block = SectionBlock(text=f"_{filter_text}_")

    if not answer.answer:
        answer_blocks = [
            SectionBlock(
                text="Sorry, I was unable to find an answer, but I did find some potentially relevant docs 🤓"
            )
        ]
    else:
        # 将markdown链接替换为slack格式的链接
        formatted_answer = format_slack_message(answer.answer)
        answer_processed = decode_escapes(
            remove_slack_text_interactions(formatted_answer)
        )
        answer_blocks = [
            SectionBlock(text=text) for text in _split_text(answer_processed)
        ]

    response_blocks: list[Block] = []

    if filter_block is not None:
        response_blocks.append(filter_block)

    response_blocks.extend(answer_blocks)

    return response_blocks


def _build_continue_in_web_ui_block(
    tenant_id: str | None,
    message_id: int | None,
) -> Block:
    """
    构建继续在Web UI中进行的消息块
    
    参数:
        tenant_id: 租户ID
        message_id: 消息ID
        
    返回:
        Block: Slack消息块对象
    """
    if message_id is None:
        raise ValueError("No message id provided to build continue in web ui block")
    with get_session_with_tenant(tenant_id) as db_session:
        chat_session = get_chat_session_by_message_id(
            db_session=db_session,
            message_id=message_id,
        )
        return ActionsBlock(
            block_id=build_continue_in_web_ui_id(message_id),
            elements=[
                ButtonElement(
                    action_id=CONTINUE_IN_WEB_UI_ACTION_ID,
                    text="Continue Chat in Onyx!",
                    style="primary",
                    url=f"{WEB_DOMAIN}/chat?slackChatId={chat_session.id}",
                ),
            ],
        )


def _build_follow_up_block(message_id: int | None) -> ActionsBlock:
    """
    构建后续跟进消息块
    
    参数:
        message_id: 消息ID
        
    返回:
        ActionsBlock: Slack操作块对象
    """
    return ActionsBlock(
        block_id=build_feedback_id(message_id) if message_id is not None else None,
        elements=[
            ButtonElement(
                action_id=IMMEDIATE_RESOLVED_BUTTON_ACTION_ID,
                style="primary",
                text="I'm all set!",
            ),
            ButtonElement(
                action_id=FOLLOWUP_BUTTON_ACTION_ID,
                style="danger",
                text="I need more help from a human!",
            ),
        ],
    )


def build_follow_up_resolved_blocks(
    tag_ids: list[str], group_ids: list[str]
) -> list[Block]:
    """
    构建后续跟进已解决消息块
    
    参数:
        tag_ids: 标签ID列表
        group_ids: 组ID列表
        
    返回:
        list[Block]: Slack消息块列表
    """
    tag_str = " ".join([f"<@{tag}>" for tag in tag_ids])
    if tag_str:
        tag_str += " "

    group_str = " ".join([f"<!subteam^{group_id}|>" for group_id in group_ids])
    if group_str:
        group_str += " "

    text = (
        tag_str
        + group_str
        + "Someone has requested more help.\n\n:point_down:Please mark this resolved after answering!"
    )
    text_block = SectionBlock(text=text)
    button_block = ActionsBlock(
        elements=[
            ButtonElement(
                action_id=FOLLOWUP_BUTTON_RESOLVED_ACTION_ID,
                style="primary",
                text="Mark Resolved",
            )
        ]
    )
    return [text_block, button_block]


def build_slack_response_blocks(
    answer: ChatOnyxBotResponse,
    tenant_id: str | None,
    message_info: SlackMessageInfo,
    channel_conf: ChannelConfig | None,
    use_citations: bool,
    feedback_reminder_id: str | None,
    skip_ai_feedback: bool = False,
) -> list[Block]:
    """
    构建完整的Slack响应块
    This function is a top level function that builds all the blocks for the Slack response.
    It also handles combining all the blocks together.
    这个函数是一个顶层函数，用于构建Slack响应的所有块，并处理所有块的组合。
    
    参数:
        answer: 聊天机器人的响应对象
        tenant_id: 租户ID
        message_info: Slack消息信息
        channel_conf: 频道配置
        use_citations: 是否使用引用
        feedback_reminder_id: 反馈提醒ID
        skip_ai_feedback: 是否跳过AI反馈
        
    返回:
        list[Block]: Slack消息块列表
    """
    # 如果使用OnyxBot斜杠命令调用，问题会丢失，所以我们必须重新显示它
    restate_question_block = get_restate_blocks(
        message_info.thread_messages[-1].message, message_info.is_bot_msg
    )

    answer_blocks = _build_qa_response_blocks(
        answer=answer,
    )

    web_follow_up_block = []
    if channel_conf and channel_conf.get("show_continue_in_web_ui"):
        web_follow_up_block.append(
            _build_continue_in_web_ui_block(
                tenant_id=tenant_id,
                message_id=answer.chat_message_id,
            )
        )

    follow_up_block = []
    if channel_conf and channel_conf.get("follow_up_tags") is not None:
        follow_up_block.append(
            _build_follow_up_block(message_id=answer.chat_message_id)
        )

    ai_feedback_block = []
    if answer.chat_message_id is not None and not skip_ai_feedback:
        ai_feedback_block.append(
            _build_qa_feedback_block(
                message_id=answer.chat_message_id,
                feedback_reminder_id=feedback_reminder_id,
            )
        )

    citations_blocks = []
    document_blocks = []
    if use_citations and answer.citations:
        citations_blocks = _build_citations_blocks(answer)
    else:
        document_blocks = _priority_ordered_documents_blocks(answer)

    citations_divider = [DividerBlock()] if citations_blocks else []
    buttons_divider = [DividerBlock()] if web_follow_up_block or follow_up_block else []

    all_blocks = (
        restate_question_block
        + answer_blocks
        + ai_feedback_block
        + citations_divider
        + citations_blocks
        + document_blocks
        + buttons_divider
        + web_follow_up_block
        + follow_up_block
    )

    return all_blocks
