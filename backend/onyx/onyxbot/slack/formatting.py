"""
此模块用于将Markdown格式的文本转换为Slack消息格式。
主要提供了Markdown到Slack特定格式的转换功能，包括文本格式化、链接处理、列表处理等。
"""

from mistune import Markdown  # type: ignore
from mistune import Renderer  # type: ignore


def format_slack_message(message: str | None) -> str:
    """
    将Markdown格式的消息转换为Slack格式的消息。
    
    参数:
        message: 需要转换的Markdown格式消息文本，可以为None
        
    返回:
        str: 转换后的Slack格式消息文本
    """
    renderer = Markdown(renderer=SlackRenderer())
    return renderer.render(message)


class SlackRenderer(Renderer):
    """
    Slack消息渲染器类，用于将Markdown语法转换为Slack支持的格式。
    继承自mistune的Renderer类。
    """
    
    SPECIALS: dict[str, str] = {"&": "&amp;", "<": "&lt;", ">": "&gt;"}

    def escape_special(self, text: str) -> str:
        """
        转义特殊字符。
        
        参数:
            text: 需要转义的文本
            
        返回:
            str: 转义后的文本
        """
        for special, replacement in self.SPECIALS.items():
            text = text.replace(special, replacement)
        return text

    def header(self, text: str, level: int, raw: str | None = None) -> str:
        """
        转换标题格式。
        
        参数:
            text: 标题文本
            level: 标题层级
            raw: 原始文本（可选）
            
        返回:
            str: 转换后的Slack格式标题
        """
        return f"*{text}*\n"

    def emphasis(self, text: str) -> str:
        return f"_{text}_"

    def double_emphasis(self, text: str) -> str:
        return f"*{text}*"

    def strikethrough(self, text: str) -> str:
        return f"~{text}~"

    def list(self, body: str, ordered: bool = True) -> str:
        """
        转换列表格式。
        
        参数:
            body: 列表内容
            ordered: 是否为有序列表
            
        返回:
            str: 转换后的Slack格式列表
        """
        lines = body.split("\n")
        count = 0
        for i, line in enumerate(lines):
            if line.startswith("li: "):
                count += 1
                prefix = f"{count}. " if ordered else "• "
                lines[i] = f"{prefix}{line[4:]}"
        return "\n".join(lines)

    def list_item(self, text: str) -> str:
        """
        转换列表项格式。
        
        参数:
            text: 列表项文本
            
        返回:
            str: 转换后的列表项格式
        """
        return f"li: {text}\n"

    def link(self, link: str, title: str | None, content: str | None) -> str:
        """
        转换链接格式。
        
        参数:
            link: 链接URL
            title: 链接标题（可选）
            content: 链接内容（可选）
            
        返回:
            str: 转换后的Slack格式链接
        """
        escaped_link = self.escape_special(link)
        if content:
            return f"<{escaped_link}|{content}>"
        if title:
            return f"<{escaped_link}|{title}>"
        return f"<{escaped_link}>"

    def image(self, src: str, title: str | None, text: str | None) -> str:
        escaped_src = self.escape_special(src)
        display_text = title or text
        return f"<{escaped_src}|{display_text}>" if display_text else f"<{escaped_src}>"

    def codespan(self, text: str) -> str:
        return f"`{text}`"

    def block_code(self, text: str, lang: str | None) -> str:
        """
        转换代码块格式。
        
        参数:
            text: 代码块内容
            lang: 编程语言（可选）
            
        返回:
            str: 转换后的Slack格式代码块
        """
        return f"```\n{text}\n```\n"

    def paragraph(self, text: str) -> str:
        """
        转换段落格式。
        
        参数:
            text: 段落文本
            
        返回:
            str: 转换后的段落格式
        """
        return f"{text}\n"

    def autolink(self, link: str, is_email: bool) -> str:
        """
        转换自动链接格式。
        
        参数:
            link: 链接文本
            is_email: 是否为邮件地址
            
        返回:
            str: 转换后的自动链接格式
        """
        return link if is_email else self.link(link, None, None)
