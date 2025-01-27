"""
这个文件包含了与LLM(大语言模型)交互相关的工具函数。
主要功能包括:
- 异常处理和错误消息转换
- 消息内容处理和格式化
- token计数和限制处理
- 模型配置和参数管理
"""

import copy
import json
from collections.abc import Callable
from collections.abc import Iterator
from typing import Any
from typing import cast

import litellm  # type: ignore
import tiktoken
from langchain.prompts.base import StringPromptValue
from langchain.prompts.chat import ChatPromptValue
from langchain.schema import PromptValue
from langchain.schema.language_model import LanguageModelInput
from langchain.schema.messages import AIMessage
from langchain.schema.messages import BaseMessage
from langchain.schema.messages import HumanMessage
from langchain.schema.messages import SystemMessage
from litellm.exceptions import APIConnectionError  # type: ignore
from litellm.exceptions import APIError  # type: ignore
from litellm.exceptions import AuthenticationError  # type: ignore
from litellm.exceptions import BadRequestError  # type: ignore
from litellm.exceptions import BudgetExceededError  # type: ignore
from litellm.exceptions import ContentPolicyViolationError  # type: ignore
from litellm.exceptions import ContextWindowExceededError  # type: ignore
from litellm.exceptions import NotFoundError  # type: ignore
from litellm.exceptions import PermissionDeniedError  # type: ignore
from litellm.exceptions import RateLimitError  # type: ignore
from litellm.exceptions import Timeout  # type: ignore
from litellm.exceptions import UnprocessableEntityError  # type: ignore

from onyx.configs.app_configs import LITELLM_CUSTOM_ERROR_MESSAGE_MAPPINGS
from onyx.configs.constants import MessageType
from onyx.configs.model_configs import GEN_AI_MAX_TOKENS
from onyx.configs.model_configs import GEN_AI_MODEL_FALLBACK_MAX_TOKENS
from onyx.configs.model_configs import GEN_AI_NUM_RESERVED_OUTPUT_TOKENS
from onyx.file_store.models import ChatFileType
from onyx.file_store.models import InMemoryChatFile
from onyx.llm.interfaces import LLM
from onyx.prompts.constants import CODE_BLOCK_PAT
from onyx.utils.b64 import get_image_type
from onyx.utils.b64 import get_image_type_from_bytes
from onyx.utils.logger import setup_logger
from shared_configs.configs import LOG_LEVEL

logger = setup_logger()


def litellm_exception_to_error_msg(
    e: Exception,
    llm: LLM,
    fallback_to_error_msg: bool = False,
    custom_error_msg_mappings: dict[str, str]
    | None = LITELLM_CUSTOM_ERROR_MESSAGE_MAPPINGS,
) -> str:
    """
    将LiteLLM的异常转换为用户友好的错误消息。

    参数:
        e: 需要处理的异常
        llm: LLM实例
        fallback_to_error_msg: 是否返回原始错误消息
        custom_error_msg_mappings: 自定义错误消息映射

    返回:
        格式化后的错误消息字符串
    """
    error_msg = str(e)

    if custom_error_msg_mappings:
        for error_msg_pattern, custom_error_msg in custom_error_msg_mappings.items():
            if error_msg_pattern in error_msg:
                return custom_error_msg

    if isinstance(e, BadRequestError):
        error_msg = "Bad request: The server couldn't process your request. Please check your input."
    elif isinstance(e, AuthenticationError):
        error_msg = "Authentication failed: Please check your API key and credentials."
    elif isinstance(e, PermissionDeniedError):
        error_msg = (
            "Permission denied: You don't have the necessary permissions for this operation."
            "Ensure you have access to this model."
        )
    elif isinstance(e, NotFoundError):
        error_msg = "Resource not found: The requested resource doesn't exist."
    elif isinstance(e, UnprocessableEntityError):
        error_msg = "Unprocessable entity: The server couldn't process your request due to semantic errors."
    elif isinstance(e, RateLimitError):
        error_msg = (
            "Rate limit exceeded: Please slow down your requests and try again later."
        )
    elif isinstance(e, ContextWindowExceededError):
        error_msg = (
            "Context window exceeded: Your input is too long for the model to process."
        )
        if llm is not None:
            try:
                max_context = get_max_input_tokens(
                    model_name=llm.config.model_name,
                    model_provider=llm.config.model_provider,
                )
                error_msg += f"Your invoked model ({llm.config.model_name}) has a maximum context size of {max_context}"
            except Exception:
                logger.warning(
                    "Unable to get maximum input token for LiteLLM excpetion handling"
                )
    elif isinstance(e, ContentPolicyViolationError):
        error_msg = "Content policy violation: Your request violates the content policy. Please revise your input."
    elif isinstance(e, APIConnectionError):
        error_msg = "API connection error: Failed to connect to the API. Please check your internet connection."
    elif isinstance(e, BudgetExceededError):
        error_msg = (
            "Budget exceeded: You've exceeded your allocated budget for API usage."
        )
    elif isinstance(e, Timeout):
        error_msg = "Request timed out: The operation took too long to complete. Please try again."
    elif isinstance(e, APIError):
        error_msg = f"API error: An error occurred while communicating with the API. Details: {str(e)}"
    elif not fallback_to_error_msg:
        error_msg = "An unexpected error occurred while processing your request. Please try again later."
    return error_msg


def _build_content(
    message: str,
    files: list[InMemoryChatFile] | None = None,
) -> str:
    """
    将非图片文件的内容与消息组合。
    
    参数:
        message: 原始消息
        files: 需要处理的文件列表
        
    返回:
        处理后的消息内容
    """
    if not files:
        return message

    text_files = [
        file
        for file in files
        if file.file_type in (ChatFileType.PLAIN_TEXT, ChatFileType.CSV)
    ]

    if not text_files:
        return message

    final_message_with_files = "FILES:\n\n"
    for file in text_files:
        file_content = file.content.decode("utf-8")
        file_name_section = f"DOCUMENT: {file.filename}\n" if file.filename else ""
        final_message_with_files += (
            f"{file_name_section}{CODE_BLOCK_PAT.format(file_content.strip())}\n\n\n"
        )

    return final_message_with_files + message


def build_content_with_imgs(
    message: str,
    files: list[InMemoryChatFile] | None = None,
    img_urls: list[str] | None = None,
    b64_imgs: list[str] | None = None,
    message_type: MessageType = MessageType.USER,
) -> str | list[str | dict[str, Any]]:
    """
    构建包含图片的消息内容。

    参数:
        message: 原始消息文本
        files: 文件列表
        img_urls: 图片URL列表
        b64_imgs: Base64编码的图片列表
        message_type: 消息类型

    返回:
        处理后的消息内容，可能是字符串或列表形式
    """
    files = files or []

    # Only include image files for user messages
    img_files = (
        [file for file in files if file.file_type == ChatFileType.IMAGE]
        if message_type == MessageType.USER
        else []
    )

    img_urls = img_urls or []
    b64_imgs = b64_imgs or []

    message_main_content = _build_content(message, files)

    if not img_files and not img_urls:
        return message_main_content

    return cast(
        list[str | dict[str, Any]],
        [
            {
                "type": "text",
                "text": message_main_content,
            },
        ]
        + [
            {
                "type": "image_url",
                "image_url": {
                    "url": (
                        f"data:{get_image_type_from_bytes(file.content)};"
                        f"base64,{file.to_base64()}"
                    ),
                },
            }
            for file in img_files
        ]
        + [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{get_image_type(b64_img)};base64,{b64_img}",
                },
            }
            for b64_img in b64_imgs
        ]
        + [
            {
                "type": "image_url",
                "image_url": {
                    "url": url,
                },
            }
            for url in img_urls
        ],
    )


def message_to_prompt_and_imgs(message: BaseMessage) -> tuple[str, list[str]]:
    """
    将消息转换为提示文本和图片列表。

    参数:
        message: 基础消息对象

    返回:
        包含提示文本和图片URL列表的元组
    """
    if isinstance(message.content, str):
        return message.content, []

    imgs = []
    texts = []
    for part in message.content:
        if isinstance(part, dict):
            if part.get("type") == "image_url":
                img_url = part.get("image_url", {}).get("url")
                if img_url:
                    imgs.append(img_url)
            elif part.get("type") == "text":
                text = part.get("text")
                if text:
                    texts.append(text)
        else:
            texts.append(part)

    return "".join(texts), imgs


def dict_based_prompt_to_langchain_prompt(
    messages: list[dict[str, str]]
) -> list[BaseMessage]:
    """
    将字典格式的提示转换为Langchain格式的提示。

    参数:
        messages: 字典格式的消息列表

    返回:
        Langchain格式的消息列表
    """
    prompt: list[BaseMessage] = []
    for message in messages:
        role = message.get("role")
        content = message.get("content")
        if not role:
            raise ValueError(f"Message missing `role`: {message}")
        if not content:
            raise ValueError(f"Message missing `content`: {message}")
        elif role == "user":
            prompt.append(HumanMessage(content=content))
        elif role == "system":
            prompt.append(SystemMessage(content=content))
        elif role == "assistant":
            prompt.append(AIMessage(content=content))
        else:
            raise ValueError(f"Unknown role: {role}")
    return prompt


def str_prompt_to_langchain_prompt(message: str) -> list[BaseMessage]:
    """
    将字符串格式的提示转换为Langchain格式的提示。

    参数:
        message: 字符串格式的消息

    返回:
        Langchain格式的消息列表
    """
    return [HumanMessage(content=message)]


def convert_lm_input_to_basic_string(lm_input: LanguageModelInput) -> str:
    """
    将语言模型输入转换为基础字符串。

    参数:
        lm_input: 语言模型输入

    返回:
        转换后的字符串
    """
    """Heavily inspired by:
    https://github.com/langchain-ai/langchain/blob/master/libs/langchain/langchain/chat_models/base.py#L86
    """
    prompt_value = None
    if isinstance(lm_input, PromptValue):
        prompt_value = lm_input
    elif isinstance(lm_input, str):
        prompt_value = StringPromptValue(text=lm_input)
    elif isinstance(lm_input, list):
        prompt_value = ChatPromptValue(messages=lm_input)

    if prompt_value is None:
        raise ValueError(
            f"Invalid input type {type(lm_input)}. "
            "Must be a PromptValue, str, or list of BaseMessages."
        )

    return prompt_value.to_string()


def message_to_string(message: BaseMessage) -> str:
    """
    将消息对象转换为字符串。

    参数:
        message: 基础消息对象

    返回:
        消息内容字符串
    """
    if not isinstance(message.content, str):
        raise RuntimeError("LLM message not in expected format.")

    return message.content


def message_generator_to_string_generator(
    messages: Iterator[BaseMessage],
) -> Iterator[str]:
    """
    将消息生成器转换为字符串生成器。

    参数:
        messages: 消息生成器

    返回:
        字符串生成器
    """
    for message in messages:
        yield message_to_string(message)


def should_be_verbose() -> bool:
    """
    检查日志级别是否为调试模式。

    返回:
        如果日志级别为调试模式，则返回True，否则返回False
    """
    return LOG_LEVEL == "debug"


# estimate of the number of tokens in an image url
# is correct when downsampling is used. Is very wrong when OpenAI does not downsample
# TODO: improve this
# 图片URL中token数量的估计值
# 在使用降采样时是正确的。当OpenAI不进行降采样时，这个估计值会非常不准确
# TODO: 需要改进这个估计方法
_IMG_TOKENS = 85


def check_message_tokens(
    message: BaseMessage, encode_fn: Callable[[str], list] | None = None
) -> int:
    """
    检查消息中的token数量。

    参数:
        message: 基础消息对象
        encode_fn: 编码函数

    返回:
        消息中的token数量
    """
    if isinstance(message.content, str):
        return check_number_of_tokens(message.content, encode_fn)

    total_tokens = 0
    for part in message.content:
        if isinstance(part, str):
            total_tokens += check_number_of_tokens(part, encode_fn)
            continue

        if part["type"] == "text":
            total_tokens += check_number_of_tokens(part["text"], encode_fn)
        elif part["type"] == "image_url":
            total_tokens += _IMG_TOKENS

    if isinstance(message, AIMessage) and message.tool_calls:
        for tool_call in message.tool_calls:
            total_tokens += check_number_of_tokens(
                json.dumps(tool_call["args"]), encode_fn
            )
            total_tokens += check_number_of_tokens(tool_call["name"], encode_fn)

    return total_tokens


def check_number_of_tokens(
    text: str, encode_fn: Callable[[str], list] | None = None
) -> int:
    """
    获取文本中的token数量。

    参数:
        text: 文本内容
        encode_fn: 编码函数

    返回:
        文本中的token数量
    """
    """Gets the number of tokens in the provided text, using the provided encoding
    function. If none is provided, default to the tiktoken encoder used by GPT-3.5
    and GPT-4.
    """

    if encode_fn is None:
        encode_fn = tiktoken.get_encoding("cl100k_base").encode

    return len(encode_fn(text))


def test_llm(llm: LLM) -> str | None:
    """
    测试LLM实例。

    参数:
        llm: LLM实例

    返回:
        如果测试失败，返回错误消息，否则返回None
    """
    # try for up to 2 timeouts (e.g. 10 seconds in total)
    # 最多尝试2次超时(例如总共10秒)
    error_msg = None
    for _ in range(2):
        try:
            llm.invoke("Do not respond")
            return None
        except Exception as e:
            error_msg = str(e)
            logger.warning(f"Failed to call LLM with the following error: {error_msg}")

    return error_msg


def get_model_map() -> dict:
    """
    获取模型映射。

    返回:
        模型映射字典
    """
    starting_map = copy.deepcopy(cast(dict, litellm.model_cost))

    # NOTE: we could add additional models here in the future,
    # but for now there is no point. Ollama allows the user to
    # to specify their desired max context window, and it's
    # unlikely to be standard across users even for the same model
    # (it heavily depends on their hardware). For now, we'll just
    # rely on GEN_AI_MODEL_FALLBACK_MAX_TOKENS to cover this.
    # 注意：我们将来可以在这里添加更多模型,
    # 但现在没有这个必要。Ollama允许用户指定他们期望的最大上下文窗口,
    # 即使是同一个模型,在不同用户之间也不太可能有标准值
    # (这很大程度上取决于他们的硬件)。目前,我们只需要依靠
    # GEN_AI_MODEL_FALLBACK_MAX_TOKENS来处理这种情况。

    return starting_map


def _strip_extra_provider_from_model_name(model_name: str) -> str:
    """
    从模型名称中去除额外的提供者前缀。

    参数:
        model_name: 模型名称

    返回:
        去除额外提供者前缀的模型名称
    """
    return model_name.split("/")[1] if "/" in model_name else model_name


def _strip_colon_from_model_name(model_name: str) -> str:
    """
    从模型名称中去除冒号及其后缀。

    参数:
        model_name: 模型名称

    返回:
        去除冒号及其后缀的模型名称
    """
    return ":".join(model_name.split(":")[:-1]) if ":" in model_name else model_name


def _find_model_obj(
    model_map: dict, provider: str, model_names: list[str | None]
) -> dict | None:
    """
    在模型映射中查找模型对象。

    参数:
        model_map: 模型映射字典
        provider: 模型提供者
        model_names: 模型名称列表

    返回:
        模型对象字典，如果未找到则返回None
    """
    # Filter out None values and deduplicate model names
    # 过滤掉None值并对模型名称去重
    filtered_model_names = [name for name in model_names if name]

    # First try all model names with provider prefix
    # 首先尝试所有带有提供者前缀的模型名称
    for model_name in filtered_model_names:
        model_obj = model_map.get(f"{provider}/{model_name}")
        if model_obj:
            logger.debug(f"Using model object for {provider}/{model_name}")
            return model_obj

    # Then try all model names without provider prefix
    # 然后尝试所有不带提供者前缀的模型名称
    for model_name in filtered_model_names:
        model_obj = model_map.get(model_name)
        if model_obj:
            logger.debug(f"Using model object for {model_name}")
            return model_obj

    return None


def get_llm_max_tokens(
    model_map: dict,
    model_name: str,
    model_provider: str,
) -> int:
    """
    获取LLM的最大token数量。

    参数:
        model_map: 模型映射字典
        model_name: 模型名称
        model_provider: 模型提供者

    返回:
        最大token数量
    """
    """Best effort attempt to get the max tokens for the LLM"""
    if GEN_AI_MAX_TOKENS:
        # This is an override, so always return this
        # 这是一个覆盖设置，所以始终返回这个值
        logger.info(f"Using override GEN_AI_MAX_TOKENS: {GEN_AI_MAX_TOKENS}")
        return GEN_AI_MAX_TOKENS

    try:
        extra_provider_stripped_model_name = _strip_extra_provider_from_model_name(
            model_name
        )
        model_obj = _find_model_obj(
            model_map,
            model_provider,
            [
                model_name,
                # Remove leading extra provider. Usually for cases where user has a
                # customer model proxy which appends another prefix
                # 移除开头的额外提供者。通常用于用户有添加了其他前缀的自定义模型代理的情况
                extra_provider_stripped_model_name,
                # remove :XXXX from the end, if present. Needed for ollama.
                # 如果存在，移除末尾的:XXXX。这对ollama是必需的
                _strip_colon_from_model_name(model_name),
                _strip_colon_from_model_name(extra_provider_stripped_model_name),
            ],
        )
        if not model_obj:
            raise RuntimeError(
                f"No litellm entry found for {model_provider}/{model_name}"
            )

        if "max_input_tokens" in model_obj:
            max_tokens = model_obj["max_input_tokens"]
            logger.info(
                f"Max tokens for {model_name}: {max_tokens} (from max_input_tokens)"
            )
            return max_tokens

        if "max_tokens" in model_obj:
            max_tokens = model_obj["max_tokens"]
            logger.info(f"Max tokens for {model_name}: {max_tokens} (from max_tokens)")
            return max_tokens

        logger.error(f"No max tokens found for LLM: {model_name}")
        raise RuntimeError("No max tokens found for LLM")
    except Exception:
        logger.exception(
            f"Failed to get max tokens for LLM with name {model_name}. Defaulting to {GEN_AI_MODEL_FALLBACK_MAX_TOKENS}."
        )
        return GEN_AI_MODEL_FALLBACK_MAX_TOKENS


def get_llm_max_output_tokens(
    model_map: dict,
    model_name: str,
    model_provider: str,
) -> int:
    """
    获取LLM的最大输出token数量。

    参数:
        model_map: 模型映射字典
        model名称: 模型名称
        model_provider: 模型提供者

    返回:
        最大输出token数量
    """
    """Best effort attempt to get the max output tokens for the LLM"""
    try:
        model_obj = model_map.get(f"{model_provider}/{model_name}")
        if not model_obj:
            model_obj = model_map[model_name]
            logger.debug(f"Using model object for {model_name}")
        else:
            logger.debug(f"Using model object for {model_provider}/{model_name}")

        if "max_output_tokens" in model_obj:
            max_output_tokens = model_obj["max_output_tokens"]
            logger.info(f"Max output tokens for {model_name}: {max_output_tokens}")
            return max_output_tokens

        # Fallback to a fraction of max_tokens if max_output_tokens is not specified
        # 如果未指定max_output_tokens，则回退到max_tokens的一个比例值
        if "max_tokens" in model_obj:
            max_output_tokens = int(model_obj["max_tokens"] * 0.1)
            logger.info(
                f"Fallback max output tokens for {model_name}: {max_output_tokens} (10% of max_tokens)"
            )
            return max_output_tokens

        logger.error(f"No max output tokens found for LLM: {model_name}")
        raise RuntimeError("No max output tokens found for LLM")
    except Exception:
        default_output_tokens = int(GEN_AI_MODEL_FALLBACK_MAX_TOKENS)
        logger.exception(
            f"Failed to get max output tokens for LLM with name {model_name}. "
            f"Defaulting to {default_output_tokens} (fallback max tokens)."
        )
        return default_output_tokens


def get_max_input_tokens(
    model_name: str,
    model_provider: str,
    output_tokens: int = GEN_AI_NUM_RESERVED_OUTPUT_TOKENS,
) -> int:
    """
    获取LLM的最大输入token数量。

    参数:
        model_name: 模型名称
        model_provider: 模型提供者
        output_tokens: 预留的输出token数量

    返回:
        最大输入token数量
    """
    # NOTE: we previously used `litellm.get_max_tokens()`, but despite the name, this actually
    # returns the max OUTPUT tokens. Under the hood, this uses the `litellm.model_cost` dict,
    # and there is no other interface to get what we want. This should be okay though, since the
    # `model_cost` dict is a named public interface:
    # https://litellm.vercel.app/docs/completion/token_usage#7-model_cost
    # 注意：我们之前使用了`litellm.get_max_tokens()`，但尽管名字如此，它实际上
    # 返回的是最大输出tokens。在底层，它使用的是`litellm.model_cost`字典，
    # 而且没有其他接口可以获取我们想要的内容。不过这应该没问题，因为
    # `model_cost`字典是一个命名的公共接口
    # model_map is  litellm.model_cost
    litellm_model_map = get_model_map()

    input_toks = (
        get_llm_max_tokens(
            model_name=model_name,
            model_provider=model_provider,
            model_map=litellm_model_map,
        )
        - output_tokens
    )

    if input_toks <= 0:
        raise RuntimeError("No tokens for input for the LLM given settings")

    return input_toks
