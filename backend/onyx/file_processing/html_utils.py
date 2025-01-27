"""
此模块用于处理和解析HTML内容，提供了一系列用于清理和格式化HTML文档的工具函数。
主要功能包括：
- HTML文档的解析和清理
- 文本格式化和规范化
- 特定HTML元素的处理（如链接、表格等）
- 移除不需要的HTML元素和类
"""

import re
from copy import copy
from dataclasses import dataclass
from typing import IO

import bs4
import trafilatura  # type: ignore
from trafilatura.settings import use_config  # type: ignore

from onyx.configs.app_configs import HTML_BASED_CONNECTOR_TRANSFORM_LINKS_STRATEGY
from onyx.configs.app_configs import PARSE_WITH_TRAFILATURA
from onyx.configs.app_configs import WEB_CONNECTOR_IGNORED_CLASSES
from onyx.configs.app_configs import WEB_CONNECTOR_IGNORED_ELEMENTS
from onyx.file_processing.enums import HtmlBasedConnectorTransformLinksStrategy
from onyx.utils.logger import setup_logger

logger = setup_logger()

MINTLIFY_UNWANTED = ["sticky", "hidden"]


@dataclass
class ParsedHTML:
    """
    解析后的HTML数据结构类
    
    属性：
        title: 页面标题，可以为空
        cleaned_text: 清理后的文本内容
    """
    title: str | None
    cleaned_text: str


def strip_excessive_newlines_and_spaces(document: str) -> str:
    """
    移除文档中多余的换行符和空格
    
    参数：
        document: 需要处理的文档字符串
    
    返回：
        处理后的文档字符串，去除了多余的空白字符
    """
    # collapse repeated spaces into one
    # 将多个空格压缩为一个
    document = re.sub(r" +", " ", document)
    # remove trailing spaces
    # 移除行尾空格
    document = re.sub(r" +[\n\r]", "\n", document)
    # remove repeated newlines
    # 移除重复的换行符
    document = re.sub(r"[\n\r]+", "\n", document)
    return document.strip()


def strip_newlines(document: str) -> str:
    """
    移除文档中的所有换行符
    
    参数：
        document: 需要处理的文档字符串
    
    返回：
        去除换行符后的文档字符串
    """
    # HTML might contain newlines which are just whitespaces to a browser
    # HTML中的换行符在浏览器中只是空白字符
    return re.sub(r"[\n\r]+", " ", document)


def format_element_text(element_text: str, link_href: str | None) -> str:
    """
    格式化元素文本，处理链接格式
    
    参数：
        element_text: 需要格式化的元素文本
        link_href: 链接地址，可以为空
    
    返回：
        格式化后的文本
    """
    element_text_no_newlines = strip_newlines(element_text)

    if (
        not link_href
        or HTML_BASED_CONNECTOR_TRANSFORM_LINKS_STRATEGY
        == HtmlBasedConnectorTransformLinksStrategy.STRIP
    ):
        return element_text_no_newlines

    return f"[{element_text_no_newlines}]({link_href})"


def parse_html_with_trafilatura(html_content: str) -> str:
    """
    使用trafilatura库解析HTML内容
    
    参数：
        html_content: 需要解析的HTML内容
    
    返回：
        解析后的文本内容
    """
    config = use_config()
    config.set("DEFAULT", "include_links", "True")
    config.set("DEFAULT", "include_tables", "True")
    config.set("DEFAULT", "include_images", "True")
    config.set("DEFAULT", "include_formatting", "True")

    extracted_text = trafilatura.extract(html_content, config=config)
    return strip_excessive_newlines_and_spaces(extracted_text) if extracted_text else ""


def format_document_soup(
    document: bs4.BeautifulSoup, table_cell_separator: str = "\t"
) -> str:
    """
    将BeautifulSoup解析的HTML文档转换为格式化的纯文本
    
    参数：
        document: BeautifulSoup解析后的文档对象
        table_cell_separator: 表格单元格分隔符，默认为制表符
    
    返回：
        格式化后的文本内容
    
    处理规则：
    - 移除HTML中的换行符（浏览器会忽略）
    - 移除重复的换行符和空格
    - 仅在标题、段落前后或显式要求时（br或pre标签）添加换行
    - 表格的行列使用换行符分隔
    - 列表元素用换行符分隔并以连字符开头
    """
    text = ""
    list_element_start = False
    verbatim_output = 0
    in_table = False
    last_added_newline = False
    link_href: str | None = None

    for e in document.descendants:
        verbatim_output -= 1
        if isinstance(e, bs4.element.NavigableString):
            if isinstance(e, (bs4.element.Comment, bs4.element.Doctype)):
                continue
            element_text = e.text
            if in_table:
                # Tables are represented in natural language with rows separated by newlines
                # 表格以自然语言形式表示，行之间用换行符分隔
                # Can't have newlines then in the table elements
                # 表格元素中不能有换行符
                element_text = element_text.replace("\n", " ").strip()

            # Some tags are translated to spaces but in the logic underneath this section, we
            # translate them to newlines as a browser should render them such as with br
            # 一些标签被转换为空格，但在下面的逻辑中，我们将它们转换为换行符，就像浏览器渲染br标签那样
            # This logic here avoids a space after newline when it shouldn't be there.
            # 这个逻辑避免了在不应该有空格的换行符后出现空格
            if last_added_newline and element_text.startswith(" "):
                element_text = element_text[1:]
                last_added_newline = False

            if element_text:
                content_to_add = (
                    element_text
                    if verbatim_output > 0
                    else format_element_text(element_text, link_href)
                )

                # Don't join separate elements without any spacing
                # 不要在没有任何间距的情况下连接独立的元素
                if (text and not text[-1].isspace()) and (
                    content_to_add and not content_to_add[0].isspace()
                ):
                    text += " "

                text += content_to_add

                list_element_start = False
        elif isinstance(e, bs4.element.Tag):
            # table is standard HTML element
            # table是标准HTML元素
            if e.name == "table":
                in_table = True
            # tr is for rows
            # tr用于表示行
            elif e.name == "tr" and in_table:
                text += "\n"
            # td for data cell, th for header
            # td用于数据单元格，th用于表头
            elif e.name in ["td", "th"] and in_table:
                text += table_cell_separator
            elif e.name == "/table":
                in_table = False
            elif in_table:
                # don't handle other cases while in table
                # 在表格中不处理其他情况
                pass
            elif e.name == "a":
                href_value = e.get("href", None)
                # mostly for typing, having multiple hrefs is not valid HTML
                # 主要用于类型检查，多个href在HTML中是无效的
                link_href = (
                    href_value[0] if isinstance(href_value, list) else href_value
                )
            elif e.name == "/a":
                link_href = None
            elif e.name in ["p", "div"]:
                if not list_element_start:
                    text += "\n"
            elif e.name in ["h1", "h2", "h3", "h4"]:
                text += "\n"
                list_element_start = False
                last_added_newline = True
            elif e.name == "br":
                text += "\n"
                list_element_start = False
                last_added_newline = True
            elif e.name == "li":
                text += "\n- "
                list_element_start = True
            elif e.name == "pre":
                if verbatim_output <= 0:
                    verbatim_output = len(list(e.childGenerator()))
    return strip_excessive_newlines_and_spaces(text)


def parse_html_page_basic(text: str | IO[bytes]) -> str:
    """
    基本的HTML页面解析函数
    
    参数：
        text: HTML文本内容或字节流
    
    返回：
        解析后的文本内容
    """
    soup = bs4.BeautifulSoup(text, "html.parser")
    return format_document_soup(soup)


def web_html_cleanup(
    page_content: str | bs4.BeautifulSoup,
    mintlify_cleanup_enabled: bool = True,
    additional_element_types_to_discard: list[str] | None = None,
) -> ParsedHTML:
    """
    清理网页HTML内容，移除不需要的元素
    
    参数：
        page_content: HTML内容或BeautifulSoup对象
        mintlify_cleanup_enabled: 是否启用mintlify相关的清理，默认为True
        additional_element_types_to_discard: 额外需要移除的HTML元素类型列表
    
    返回：
        ParsedHTML对象，包含清理后的标题和文本内容
    """
    if isinstance(page_content, str):
        soup = bs4.BeautifulSoup(page_content, "html.parser")
    else:
        soup = page_content

    title_tag = soup.find("title")
    title = None
    if title_tag and title_tag.text:
        title = title_tag.text
        title_tag.extract()

    # Heuristics based cleaning of elements based on css classes
    # 基于CSS类的启发式元素清理
    unwanted_classes = copy(WEB_CONNECTOR_IGNORED_CLASSES)
    if mintlify_cleanup_enabled:
        unwanted_classes.extend(MINTLIFY_UNWANTED)
    for undesired_element in unwanted_classes:
        [
            tag.extract()
            for tag in soup.find_all(
                class_=lambda x: x and undesired_element in x.split()
            )
        ]

    for undesired_tag in WEB_CONNECTOR_IGNORED_ELEMENTS:
        [tag.extract() for tag in soup.find_all(undesired_tag)]

    if additional_element_types_to_discard:
        for undesired_tag in additional_element_types_to_discard:
            [tag.extract() for tag in soup.find_all(undesired_tag)]

    soup_string = str(soup)
    page_text = ""

    if PARSE_WITH_TRAFILATURA:
        try:
            page_text = parse_html_with_trafilatura(soup_string)
            if not page_text:
                raise ValueError("Empty content returned by trafilatura.")
                # trafilatura返回了空内容
        except Exception as e:
            logger.info(f"Trafilatura parsing failed: {e}. Falling back on bs4.")
            # Trafilatura解析失败，回退使用bs4
            page_text = format_document_soup(soup)
    else:
        page_text = format_document_soup(soup)

    # 200B is ZeroWidthSpace which we don't care for
    # 200B是零宽度空格，我们不需要它
    cleaned_text = page_text.replace("\u200B", "")

    return ParsedHTML(title=title, cleaned_text=cleaned_text)
