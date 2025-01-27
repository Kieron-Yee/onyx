"""
此文件主要用于处理聊天过程中的引用内容流式处理。
主要功能包括：
- 从模型输出中提取答案和引用内容
- 将引用内容与文档进行匹配
- 处理流式输出的引用内容
"""

# THIS IS NO LONGER IN USE
# 此文件已不再使用
import math
import re
from collections.abc import Generator
from json import JSONDecodeError
from typing import Optional

import regex
from pydantic import BaseModel

from onyx.chat.models import LlmDoc
from onyx.chat.models import OnyxAnswer
from onyx.chat.models import OnyxAnswerPiece
from onyx.configs.chat_configs import QUOTE_ALLOWED_ERROR_PERCENT
from onyx.context.search.models import InferenceChunk
from onyx.prompts.constants import ANSWER_PAT
from onyx.prompts.constants import QUOTE_PAT
from onyx.utils.logger import setup_logger
from onyx.utils.text_processing import clean_model_quote
from onyx.utils.text_processing import clean_up_code_blocks
from onyx.utils.text_processing import extract_embedded_json
from onyx.utils.text_processing import shared_precompare_cleanup


logger = setup_logger()
answer_pattern = re.compile(r'{\s*"answer"\s*:\s*"', re.IGNORECASE)


class OnyxQuote(BaseModel):
    """
    表示单个引用的模型类
    
    属性:
        quote: str - 引用的具体内容
        document_id: str - 引用来源文档的ID
        link: str | None - 引用的链接
        source_type: str - 引用来源的类型
        semantic_identifier: str - 语义标识符
        blurb: str - 引用的简短描述
    """
    # This is during inference so everything is a string by this point
    # 在推理阶段，所有内容都以字符串形式存储
    quote: str
    document_id: str
    link: str | None
    source_type: str
    semantic_identifier: str
    blurb: str


class OnyxQuotes(BaseModel):
    """
    包含多个引用的容器类
    
    属性:
        quotes: list[OnyxQuote] - 引用列表
    """
    quotes: list[OnyxQuote]


def _extract_answer_quotes_freeform(
    answer_raw: str,
) -> tuple[Optional[str], Optional[list[str]]]:
    """
    从原始文本中提取答案和引用部分
    
    参数:
        answer_raw: str - 原始的模型输出文本
    
    返回:
        tuple[Optional[str], Optional[list[str]]] - (答案文本, 引用列表)
    """
    """Splits the model output into an Answer and 0 or more Quote sections.
    Splits by the Quote pattern, if not exist then assume it's all answer and no quotes
    """
    # If no answer section, don't care about the quote
    if answer_raw.lower().strip().startswith(QUOTE_PAT.lower()):
        return None, None

    # Sometimes model regenerates the Answer: pattern despite it being provided in the prompt
    if answer_raw.lower().startswith(ANSWER_PAT.lower()):
        answer_raw = answer_raw[len(ANSWER_PAT) :]

    # Accept quote sections starting with the lower case version
    answer_raw = answer_raw.replace(
        f"\n{QUOTE_PAT}".lower(), f"\n{QUOTE_PAT}"
    )  # Just in case model unreliable

    sections = re.split(rf"(?<=\n){QUOTE_PAT}", answer_raw)
    sections_clean = [
        str(section).strip() for section in sections if str(section).strip()
    ]
    if not sections_clean:
        return None, None

    answer = str(sections_clean[0])
    if len(sections) == 1:
        return answer, None
    return answer, sections_clean[1:]


def _extract_answer_quotes_json(
    answer_dict: dict[str, str | list[str]]
) -> tuple[Optional[str], Optional[list[str]]]:
    """
    从JSON格式的字典中提取答案和引用
    
    参数:
        answer_dict: dict - 包含答案和引用的字典
        
    返回:
        tuple[Optional[str], Optional[list[str]]] - (答案文本, 引用列表)
    """
    answer_dict = {k.lower(): v for k, v in answer_dict.items()}
    answer = str(answer_dict.get("answer"))
    quotes = answer_dict.get("quotes") or answer_dict.get("quote")
    if isinstance(quotes, str):
        quotes = [quotes]
    return answer, quotes


def _extract_answer_json(raw_model_output: str) -> dict:
    try:
        answer_json = extract_embedded_json(raw_model_output)
    except (ValueError, JSONDecodeError):
        # LLMs get confused when handling the list in the json. Sometimes it doesn't attend
        # enough to the previous { token so it just ends the list of quotes and stops there
        # here, we add logic to try to fix this LLM error.
        answer_json = extract_embedded_json(raw_model_output + "}")

    if "answer" not in answer_json:
        raise ValueError("Model did not output an answer as expected.")

    return answer_json


def match_quotes_to_docs(
    quotes: list[str],
    docs: list[LlmDoc] | list[InferenceChunk],
    max_error_percent: float = QUOTE_ALLOWED_ERROR_PERCENT,
    fuzzy_search: bool = False,
    prefix_only_length: int = 100,
) -> OnyxQuotes:
    """
    将引用内容与原始文档进行匹配
    
    参数:
        quotes: list[str] - 引用列表
        docs: list - 文档列表
        max_error_percent: float - 允许的最大错误百分比
        fuzzy_search: bool - 是否使用模糊匹配
        prefix_only_length: int - 仅匹配前缀的长度
        
    返回:
        OnyxQuotes - 匹配后的引用对象
    """
    onyx_quotes: list[OnyxQuote] = []
    for quote in quotes:
        max_edits = math.ceil(float(len(quote)) * max_error_percent)

        for doc in docs:
            if not doc.source_links:
                continue

            quote_clean = shared_precompare_cleanup(
                clean_model_quote(quote, trim_length=prefix_only_length)
            )
            chunk_clean = shared_precompare_cleanup(doc.content)

            # Finding the offset of the quote in the plain text
            if fuzzy_search:
                re_search_str = (
                    r"(" + re.escape(quote_clean) + r"){e<=" + str(max_edits) + r"}"
                )
                found = regex.search(re_search_str, chunk_clean)
                if not found:
                    continue
                offset = found.span()[0]
            else:
                if quote_clean not in chunk_clean:
                    continue
                offset = chunk_clean.index(quote_clean)

            # Extracting the link from the offset
            curr_link = None
            for link_offset, link in doc.source_links.items():
                # Should always find one because offset is at least 0 and there
                # must be a 0 link_offset
                if int(link_offset) <= offset:
                    curr_link = link
                else:
                    break

            onyx_quotes.append(
                OnyxQuote(
                    quote=quote,
                    document_id=doc.document_id,
                    link=curr_link,
                    source_type=doc.source_type,
                    semantic_identifier=doc.semantic_identifier,
                    blurb=doc.blurb,
                )
            )
            break

    return OnyxQuotes(quotes=onyx_quotes)


def separate_answer_quotes(
    answer_raw: str, is_json_prompt: bool = False
) -> tuple[Optional[str], Optional[list[str]]]:
    """Takes in a raw model output and pulls out the answer and the quotes sections."""
    if is_json_prompt:
        model_raw_json = _extract_answer_json(answer_raw)
        return _extract_answer_quotes_json(model_raw_json)

    return _extract_answer_quotes_freeform(clean_up_code_blocks(answer_raw))


def _process_answer(
    answer_raw: str,
    docs: list[LlmDoc],
    is_json_prompt: bool = True,
) -> tuple[OnyxAnswer, OnyxQuotes]:
    """Used (1) in the non-streaming case to process the model output
    into an Answer and Quotes AND (2) after the complete streaming response
    has been received to process the model output into an Answer and Quotes."""
    answer, quote_strings = separate_answer_quotes(answer_raw, is_json_prompt)
    if not answer:
        logger.debug("No answer extracted from raw output")
        return OnyxAnswer(answer=None), OnyxQuotes(quotes=[])

    logger.notice(f"Answer: {answer}")
    if not quote_strings:
        logger.debug("No quotes extracted from raw output")
        return OnyxAnswer(answer=answer), OnyxQuotes(quotes=[])
    logger.debug(f"All quotes (including unmatched): {quote_strings}")
    quotes = match_quotes_to_docs(quote_strings, docs)
    logger.debug(f"Final quotes: {quotes}")

    return OnyxAnswer(answer=answer), quotes


def _stream_json_answer_end(answer_so_far: str, next_token: str) -> bool:
    next_token = next_token.replace('\\"', "")
    # If the previous character is an escape token, don't consider the first character of next_token
    # This does not work if it's an escaped escape sign before the " but this is rare, not worth handling
    if answer_so_far and answer_so_far[-1] == "\\":
        next_token = next_token[1:]
    if '"' in next_token:
        return True
    return False


def _extract_quotes_from_completed_token_stream(
    model_output: str, context_docs: list[LlmDoc], is_json_prompt: bool = True
) -> OnyxQuotes:
    answer, quotes = _process_answer(model_output, context_docs, is_json_prompt)
    if answer:
        logger.notice(answer)
    elif model_output:
        logger.warning("Answer extraction from model output failed.")

    return quotes


class QuotesProcessor:
    """
    引用处理器类，用于处理流式输出中的引用内容
    
    属性:
        context_docs: list[LlmDoc] - 上下文文档列表
        is_json_prompt: bool - 是否为JSON格式的提示
        found_answer_start: bool - 是否找到答案开始
        found_answer_end: bool - 是否找到答案结束
        hold_quote: str - 临时存储的引用内容
        model_output: str - 模型输出内容
        hold: str - 临时存储区
    """
    
    def __init__(
        self,
        context_docs: list[LlmDoc],
        is_json_prompt: bool = True,
    ):
        """
        初始化引用处理器
        
        参数:
            context_docs: list[LlmDoc] - 上下文文档列表
            is_json_prompt: bool - 是否为JSON格式的提示
        """
        self.context_docs = context_docs
        self.is_json_prompt = is_json_prompt

        self.found_answer_start = False if is_json_prompt else True
        self.found_answer_end = False
        self.hold_quote = ""
        self.model_output = ""
        self.hold = ""

    def process_token(
        self, token: str | None
    ) -> Generator[OnyxAnswerPiece | OnyxQuotes, None, None]:
        """
        处理单个token，生成答案片段或引用
        
        参数:
            token: str | None - 输入的token，None表示流结束
            
        返回:
            Generator - 生成答案片段或引用对象
        """
        # None -> end of stream
        # None表示流结束
        if token is None:
            if self.model_output:
                yield _extract_quotes_from_completed_token_stream(
                    model_output=self.model_output,
                    context_docs=self.context_docs,
                    is_json_prompt=self.is_json_prompt,
                )
            return

        model_previous = self.model_output
        self.model_output += token
        if not self.found_answer_start:
            m = answer_pattern.search(self.model_output)
            if m:
                self.found_answer_start = True

                # Prevent heavy cases of hallucinations
                if self.is_json_prompt and len(self.model_output) > 400:
                    self.found_answer_end = True
                    logger.warning("LLM did not produce json as prompted")
                    logger.debug("Model output thus far:", self.model_output)
                    return

                remaining = self.model_output[m.end() :]

                # Look for an unescaped quote, which means the answer is entirely contained
                # in this token e.g. if the token is `{"answer": "blah", "qu`
                quote_indices = [i for i, char in enumerate(remaining) if char == '"']
                for quote_idx in quote_indices:
                    # Check if quote is escaped by counting backslashes before it
                    num_backslashes = 0
                    pos = quote_idx - 1
                    while pos >= 0 and remaining[pos] == "\\":
                        num_backslashes += 1
                        pos -= 1
                    # If even number of backslashes, quote is not escaped
                    if num_backslashes % 2 == 0:
                        yield OnyxAnswerPiece(answer_piece=remaining[:quote_idx])
                        return

                # If no unescaped quote found, yield the remaining string
                if len(remaining) > 0:
                    yield OnyxAnswerPiece(answer_piece=remaining)
                return

        if self.found_answer_start and not self.found_answer_end:
            if self.is_json_prompt and _stream_json_answer_end(model_previous, token):
                self.found_answer_end = True

                if token:
                    try:
                        answer_token_section = token.index('"')
                        yield OnyxAnswerPiece(
                            answer_piece=self.hold_quote + token[:answer_token_section]
                        )
                    except ValueError:
                        logger.error("Quotation mark not found in token")
                        yield OnyxAnswerPiece(answer_piece=self.hold_quote + token)
                yield OnyxAnswerPiece(answer_piece=None)
                return

            elif not self.is_json_prompt:
                quote_pat = f"\n{QUOTE_PAT}"
                quote_loose = f"\n{quote_pat[:-1]}\n"
                quote_pat_full = f"\n{quote_pat}"

                if (
                    quote_pat in self.hold_quote + token
                    or quote_loose in self.hold_quote + token
                ):
                    self.found_answer_end = True
                    yield OnyxAnswerPiece(answer_piece=None)
                    return
                if self.hold_quote + token in quote_pat_full:
                    self.hold_quote += token
                    return

            yield OnyxAnswerPiece(answer_piece=self.hold_quote + token)
            self.hold_quote = ""
