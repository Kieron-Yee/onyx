"""
此文件用于管理和提供不同文档来源的图标链接。
主要功能是根据文档来源返回对应的 GitHub 图标图片链接。
"""

from onyx.configs.constants import DocumentSource


def source_to_github_img_link(source: DocumentSource) -> str | None:
    """
    根据文档来源返回对应的 GitHub 图标图片链接。

    Args:
        source (DocumentSource): 文档来源枚举值，表示不同的文档来源类型
        
    Returns:
        str | None: 返回对应文档来源的图标URL字符串，如果没有匹配的图标则返回默认文件图标
    """
    # TODO: store these images somewhere better
    # TODO: 需要找一个更好的位置存储这些图片
    if source == DocumentSource.WEB.value:
        return "https://raw.githubusercontent.com/onyx-dot-app/onyx/main/backend/slackbot_images/Web.png"
    if source == DocumentSource.FILE.value:
        return "https://raw.githubusercontent.com/onyx-dot-app/onyx/main/backend/slackbot_images/File.png"
    if source == DocumentSource.GOOGLE_SITES.value:
        return "https://raw.githubusercontent.com/onyx-dot-app/onyx/main/web/public/GoogleSites.png"
    if source == DocumentSource.SLACK.value:
        return "https://raw.githubusercontent.com/onyx-dot-app/onyx/main/web/public/Slack.png"
    if source == DocumentSource.GMAIL.value:
        return "https://raw.githubusercontent.com/onyx-dot-app/onyx/main/web/public/Gmail.png"
    if source == DocumentSource.GOOGLE_DRIVE.value:
        return "https://raw.githubusercontent.com/onyx-dot-app/onyx/main/web/public/GoogleDrive.png"
    if source == DocumentSource.GITHUB.value:
        return "https://raw.githubusercontent.com/onyx-dot-app/onyx/main/web/public/Github.png"
    if source == DocumentSource.GITLAB.value:
        return "https://raw.githubusercontent.com/onyx-dot-app/onyx/main/web/public/Gitlab.png"
    if source == DocumentSource.CONFLUENCE.value:
        return "https://raw.githubusercontent.com/onyx-dot-app/onyx/main/backend/slackbot_images/Confluence.png"
    if source == DocumentSource.JIRA.value:
        return "https://raw.githubusercontent.com/onyx-dot-app/onyx/main/backend/slackbot_images/Jira.png"
    if source == DocumentSource.NOTION.value:
        return "https://raw.githubusercontent.com/onyx-dot-app/onyx/main/web/public/Notion.png"
    if source == DocumentSource.ZENDESK.value:
        return "https://raw.githubusercontent.com/onyx-dot-app/onyx/main/backend/slackbot_images/Zendesk.png"
    if source == DocumentSource.GONG.value:
        return "https://raw.githubusercontent.com/onyx-dot-app/onyx/main/web/public/Gong.png"
    if source == DocumentSource.LINEAR.value:
        return "https://raw.githubusercontent.com/onyx-dot-app/onyx/main/web/public/Linear.png"
    if source == DocumentSource.PRODUCTBOARD.value:
        return "https://raw.githubusercontent.com/onyx-dot-app/onyx/main/web/public/Productboard.webp"
    if source == DocumentSource.SLAB.value:
        return "https://raw.githubusercontent.com/onyx-dot-app/onyx/main/web/public/SlabLogo.png"
    if source == DocumentSource.ZULIP.value:
        return "https://raw.githubusercontent.com/onyx-dot-app/onyx/main/web/public/Zulip.png"
    if source == DocumentSource.GURU.value:
        return "https://raw.githubusercontent.com/onyx-dot-app/onyx/main/backend/slackbot_images/Guru.png"
    if source == DocumentSource.HUBSPOT.value:
        return "https://raw.githubusercontent.com/onyx-dot-app/onyx/main/web/public/HubSpot.png"
    if source == DocumentSource.DOCUMENT360.value:
        return "https://raw.githubusercontent.com/onyx-dot-app/onyx/main/web/public/Document360.png"
    if source == DocumentSource.BOOKSTACK.value:
        return "https://raw.githubusercontent.com/onyx-dot-app/onyx/main/web/public/Bookstack.png"
    if source == DocumentSource.LOOPIO.value:
        return "https://raw.githubusercontent.com/onyx-dot-app/onyx/main/web/public/Loopio.png"
    if source == DocumentSource.SHAREPOINT.value:
        return "https://raw.githubusercontent.com/onyx-dot-app/onyx/main/web/public/Sharepoint.png"
    if source == DocumentSource.REQUESTTRACKER.value:
        # just use file icon for now
        return "https://raw.githubusercontent.com/onyx-dot-app/onyx/main/backend/slackbot_images/File.png"
    if source == DocumentSource.INGESTION_API.value:
        return "https://raw.githubusercontent.com/onyx-dot-app/onyx/main/backend/slackbot_images/File.png"

    # 如果没有匹配的来源，返回默认的文件图标
    return "https://raw.githubusercontent.com/onyx-dot-app/onyx/main/backend/slackbot_images/File.png"
