"""
文本处理工具模块
该模块提供了一系列用于文本处理的实用函数，包括转义序列处理、URL编码、JSON处理、
文本清理等功能。主要用于处理各种文本格式化和清理任务。
"""

import codecs
import json
import re
import string
from urllib.parse import quote

from onyx.utils.logger import setup_logger


logger = setup_logger(__name__)

ESCAPE_SEQUENCE_RE = re.compile(
    r"""
    ( \\U........      # 8-digit hex escapes
    | \\u....          # 4-digit hex escapes
    | \\x..            # 2-digit hex escapes
    | \\[0-7]{1,3}     # Octal escapes
    | \\N\{[^}]+\}     # Unicode characters by name
    | \\[\\'"abfnrtv]  # Single-character escapes
    )""",
    re.UNICODE | re.VERBOSE,
)

"""
解码字符串中的转义序列
参数:
    s (str): 包含转义序列的字符串
返回:
    str: 解码后的字符串
"""
def decode_escapes(s: str) -> str:
    def decode_match(match: re.Match) -> str:
        return codecs.decode(match.group(0), "unicode-escape")

    return ESCAPE_SEQUENCE_RE.sub(decode_match, s)

"""
将字符串转换为URL兼容格式
参数:
    s (str): 需要转换的字符串
返回:
    str: URL编码后的字符串
"""
def make_url_compatible(s: str) -> str:
    s_with_underscores = s.replace(" ", "_")
    return quote(s_with_underscores, safe="")

"""
检查字符串中是否包含未转义的引号
参数:
    s (str): 待检查的字符串
返回:
    bool: 如果存在未转义的引号返回True，否则返回False
"""
def has_unescaped_quote(s: str) -> bool:
    pattern = r'(?<!\\)"'
    return bool(re.search(pattern, s))

"""
转义字符串中的换行符
参数:
    s (str): 需要处理的字符串
返回:
    str: 转义换行符后的字符串
"""
def escape_newlines(s: str) -> str:
    return re.sub(r"(?<!\\)\n", "\\\\n", s)

"""
将所有空白字符替换为单个空格
参数:
    s (str): 需要处理的字符串
返回:
    str: 替换后的字符串
"""
def replace_whitespaces_w_space(s: str) -> str:
    return re.sub(r"\s", " ", s)

"""
移除字符串中的标点符号
参数:
    s (str): 需要处理的字符串
返回:
    str: 移除标点符号后的字符串
"""
def remove_punctuation(s: str) -> str:
    return s.translate(str.maketrans("", "", string.punctuation))

"""
转义JSON字符串中的引号
参数:
    original_json_str (str): 原始JSON字符串
返回:
    str: 转义引号后的字符串
"""
def escape_quotes(original_json_str: str) -> str:
    result = []
    in_string = False
    for i, char in enumerate(original_json_str):
        if char == '"':
            if not in_string:
                in_string = True
                result.append(char)
            else:
                next_char = (
                    original_json_str[i + 1] if i + 1 < len(original_json_str) else None
                )
                if result and result[-1] == "\\":
                    result.append(char)
                elif next_char not in [",", ":", "}", "\n"]:
                    result.append("\\" + char)
                else:
                    result.append(char)
                    in_string = False
        else:
            result.append(char)
    return "".join(result)

"""
从字符串中提取嵌入的JSON数据
参数:
    s (str): 包含JSON数据的字符串
返回:
    dict: 解析后的JSON数据
异常:
    ValueError: 当JSON解析失败时抛出
"""
def extract_embedded_json(s: str) -> dict:
    first_brace_index = s.find("{")
    last_brace_index = s.rfind("}")

    if first_brace_index == -1 or last_brace_index == -1:
        logger.warning("No valid json found, assuming answer is entire string")
        return {"answer": s, "quotes": []}

    json_str = s[first_brace_index : last_brace_index + 1]
    try:
        return json.loads(json_str, strict=False)

    except json.JSONDecodeError:
        try:
            return json.loads(escape_quotes(json_str), strict=False)
        except json.JSONDecodeError as e:
            raise ValueError("Failed to parse JSON, even after escaping quotes") from e

"""
清理代码块中的多余字符
参数:
    model_out_raw (str): 原始代码块字符串
返回:
    str: 清理后的代码块字符串
"""
def clean_up_code_blocks(model_out_raw: str) -> str:
    return model_out_raw.strip().strip("```").strip().replace("\\xa0", "")

"""
清理和修剪引用文本
参数:
    quote (str): 原始引用文本
    trim_length (int): 修剪长度
返回:
    str: 清理后的引用文本
"""
def clean_model_quote(quote: str, trim_length: int) -> str:
    quote_clean = quote.strip()
    if quote_clean[0] == '"':
        quote_clean = quote_clean[1:]
    if quote_clean[-1] == '"':
        quote_clean = quote_clean[:-1]
    if trim_length > 0:
        quote_clean = quote_clean[:trim_length]
    return quote_clean

"""
对文本进行预处理以便于比较
LLMs models sometime restructure whitespaces or edits special characters to fit a more likely
distribution of characters found in its training data, but this hurts exact quote matching
LLM模型有时会重构空白符或编辑特殊字符以适应其训练数据中更可能出现的字符分布，但这会影响精确的引用匹配

参数:
    text (str): 需要处理的文本
返回:
    str: 预处理后的文本
"""
def shared_precompare_cleanup(text: str) -> str:
    text = text.lower()

    text = re.sub(r'\s|\*|\\"|[.,:`"#-]', "", text)

    return text

_INITIAL_FILTER = re.compile(
    "["
    "\U0000FFF0-\U0000FFFF"  # Specials
    "\U0001F000-\U0001F9FF"  # Emoticons
    "\U00002000-\U0000206F"  # General Punctuation
    "\U00002190-\U000021FF"  # Arrows
    "\U00002700-\U000027BF"  # Dingbats
    "]+",
    flags=re.UNICODE,
)

"""
清理文本中的特殊Unicode字符
参数:
    text (str): 需要清理的文本
返回:
    str: 清理后的文本
"""
def clean_text(text: str) -> str:
    cleaned = _INITIAL_FILTER.sub("", text)

    cleaned = "".join(ch for ch in cleaned if ch >= " " or ch in "\n\t")

    return cleaned

"""
验证文本是否为有效的电子邮件地址
Can use a library instead if more detailed checks are needed
如果需要更详细的检查可以使用专门的库

参数:
    text (str): 需要验证的文本
返回:
    bool: 如果是有效的电子邮件地址返回True，否则返回False
"""
def is_valid_email(text: str) -> bool:
    regex = r"^[a-zA-Z0-9._-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"

    if re.match(regex, text):
        return True
    else:
        return False

"""
计算文本中的标点符号数量
参数:
    text (str): 需要计算的文本
返回:
    int: 标点符号的数量
"""
def count_punctuation(text: str) -> int:
    return sum(1 for char in text if char in string.punctuation)
