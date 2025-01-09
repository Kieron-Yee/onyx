import torch
import torch.nn.functional as F
from fastapi import APIRouter
from huggingface_hub import snapshot_download  # type: ignore
from transformers import AutoTokenizer  # type: ignore
from transformers import BatchEncoding  # type: ignore
from transformers import PreTrainedTokenizer  # type: ignore

from model_server.constants import MODEL_WARM_UP_STRING
from model_server.onyx_torch_model import ConnectorClassifier
from model_server.onyx_torch_model import HybridClassifier
from model_server.utils import simple_log_function_time
from onyx.utils.logger import setup_logger
from shared_configs.configs import CONNECTOR_CLASSIFIER_MODEL_REPO
from shared_configs.configs import CONNECTOR_CLASSIFIER_MODEL_TAG
from shared_configs.configs import INDEXING_ONLY
from shared_configs.configs import INTENT_MODEL_TAG
from shared_configs.configs import INTENT_MODEL_VERSION
from shared_configs.model_server_models import ConnectorClassificationRequest
from shared_configs.model_server_models import ConnectorClassificationResponse
from shared_configs.model_server_models import IntentRequest
from shared_configs.model_server_models import IntentResponse

logger = setup_logger()

router = APIRouter(prefix="/custom")

_CONNECTOR_CLASSIFIER_TOKENIZER: AutoTokenizer | None = None
_CONNECTOR_CLASSIFIER_MODEL: ConnectorClassifier | None = None

_INTENT_TOKENIZER: AutoTokenizer | None = None
_INTENT_MODEL: HybridClassifier | None = None

# 获取连接器分类器的分词器
# Get the tokenizer for the connector classifier
def get_connector_classifier_tokenizer() -> AutoTokenizer:
    global _CONNECTOR_CLASSIFIER_TOKENIZER
    if _CONNECTOR_CLASSIFIER_TOKENIZER is None:
        # 分词器详情未上传到HF hub，因为它只是未修改的distilbert分词器。
        # The tokenizer details are not uploaded to the HF hub since it's just the
        # unmodified distilbert tokenizer.
        _CONNECTOR_CLASSIFIER_TOKENIZER = AutoTokenizer.from_pretrained(
            "distilbert-base-uncased"
        )
    return _CONNECTOR_CLASSIFIER_TOKENIZER

# 获取本地连接器分类器模型
# Get the local connector classifier model
def get_local_connector_classifier(
    model_name_or_path: str = CONNECTOR_CLASSIFIER_MODEL_REPO,
    tag: str = CONNECTOR_CLASSIFIER_MODEL_TAG,
) -> ConnectorClassifier:
    global _CONNECTOR_CLASSIFIER_MODEL
    if _CONNECTOR_CLASSIFIER_MODEL is None:
        try:
            # 计算缓存位置，然后从本地加载（如果可用）
            # Calculate where the cache should be, then load from local if available
            local_path = snapshot_download(
                repo_id=model_name_or_path, revision=tag, local_files_only=True
            )
            _CONNECTOR_CLASSIFIER_MODEL = ConnectorClassifier.from_pretrained(
                local_path
            )
        except Exception as e:
            logger.warning(f"Failed to load model directly: {e}")
            try:
                # 尝试下载模型快照
                # Attempt to download the model snapshot
                logger.info(f"Downloading model snapshot for {model_name_or_path}")
                local_path = snapshot_download(repo_id=model_name_or_path, revision=tag)
                _CONNECTOR_CLASSIFIER_MODEL = ConnectorClassifier.from_pretrained(
                    local_path
                )
            except Exception as e:
                logger.error(
                    f"Failed to load model even after attempted snapshot download: {e}"
                )
                raise
    return _CONNECTOR_CLASSIFIER_MODEL

# 获取意图模型的分词器
# Get the tokenizer for the intent model
def get_intent_model_tokenizer() -> AutoTokenizer:
    global _INTENT_TOKENIZER
    if _INTENT_TOKENIZER is None:
        # 分词器详情未上传到HF hub，因为它只是未修改的distilbert分词器。
        # The tokenizer details are not uploaded to the HF hub since it's just the
        # unmodified distilbert tokenizer.
        _INTENT_TOKENIZER = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    return _INTENT_TOKENIZER

# 获取本地意图模型
# Get the local intent model
def get_local_intent_model(
    model_name_or_path: str = INTENT_MODEL_VERSION,
    tag: str = INTENT_MODEL_TAG,
) -> HybridClassifier:
    global _INTENT_MODEL
    if _INTENT_MODEL is None:
        try:
            # 计算缓存位置，然后从本地加载（如果可用）
            # Calculate where the cache should be, then load from local if available
            logger.notice(f"Loading model from local cache: {model_name_or_path}")
            local_path = snapshot_download(
                repo_id=model_name_or_path, revision=tag, local_files_only=True
            )
            _INTENT_MODEL = HybridClassifier.from_pretrained(local_path)
            logger.notice(f"Loaded model from local cache: {local_path}")
        except Exception as e:
            logger.warning(f"Failed to load model directly: {e}")
            try:
                # 尝试下载模型快照
                # Attempt to download the model snapshot
                logger.notice(f"Downloading model snapshot for {model_name_or_path}")
                local_path = snapshot_download(repo_id=model_name_or_path, revision=tag)
                _INTENT_MODEL = HybridClassifier.from_pretrained(local_path)
            except Exception as e:
                logger.error(
                    f"Failed to load model even after attempted snapshot download: {e}"
                )
                raise
    return _INTENT_MODEL

# 对连接器分类查询进行分词
# Tokenize the connectors & user query into one prompt for the forward pass of ConnectorClassifier models
def tokenize_connector_classification_query(
    connectors: list[str],
    query: str,
    tokenizer: PreTrainedTokenizer,
    connector_token_end_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    将连接器和用户查询分词为一个提示，用于ConnectorClassifier模型的前向传递

    注意力掩码全为1。提示是CLS + 每个连接器名称后缀连接器结束标记，然后是用户查询。
    The attention mask is just all 1s. The prompt is CLS + each connector name suffixed with the connector end
    token and then the user query.
    """

    input_ids = torch.tensor([tokenizer.cls_token_id], dtype=torch.long)

    for connector in connectors:
        connector_token_ids = tokenizer(
            connector,
            add_special_tokens=False,
            return_tensors="pt",
        )

        input_ids = torch.cat(
            (
                input_ids,
                connector_token_ids["input_ids"].squeeze(dim=0),
                torch.tensor([connector_token_end_id], dtype=torch.long),
            ),
            dim=-1,
        )
    query_token_ids = tokenizer(
        query,
        add_special_tokens=False,
        return_tensors="pt",
    )

    input_ids = torch.cat(
        (
            input_ids,
            query_token_ids["input_ids"].squeeze(dim=0),
            torch.tensor([tokenizer.sep_token_id], dtype=torch.long),
        ),
        dim=-1,
    )
    attention_mask = torch.ones(input_ids.numel(), dtype=torch.long)

    return input_ids.unsqueeze(0), attention_mask.unsqueeze(0)

# 预热连接器分类器模型
# Warm up the connector classifier model
def warm_up_connector_classifier_model() -> None:
    logger.info(
        f"Warming up connector_classifier model {CONNECTOR_CLASSIFIER_MODEL_TAG}"
    )
    connector_classifier_tokenizer = get_connector_classifier_tokenizer()
    connector_classifier = get_local_connector_classifier()

    input_ids, attention_mask = tokenize_connector_classification_query(
        ["GitHub"],
        "onyx classifier query google doc",
        connector_classifier_tokenizer,
        connector_classifier.connector_end_token_id,
    )
    input_ids = input_ids.to(connector_classifier.device)
    attention_mask = attention_mask.to(connector_classifier.device)

    connector_classifier(input_ids, attention_mask)

# 预热意图模型
# Warm up the intent model
def warm_up_intent_model() -> None:
    logger.notice(f"Warming up Intent Model: {INTENT_MODEL_VERSION}")
    intent_tokenizer = get_intent_model_tokenizer()
    tokens = intent_tokenizer(
        MODEL_WARM_UP_STRING, return_tensors="pt", truncation=True, padding=True
    )

    intent_model = get_local_intent_model()
    device = intent_model.device
    intent_model(
        query_ids=tokens["input_ids"].to(device),
        query_mask=tokens["attention_mask"].to(device),
    )

# 运行推理
# Run inference
@simple_log_function_time()
def run_inference(tokens: BatchEncoding) -> tuple[list[float], list[float]]:
    intent_model = get_local_intent_model()
    device = intent_model.device

    outputs = intent_model(
        query_ids=tokens["input_ids"].to(device),
        query_mask=tokens["attention_mask"].to(device),
    )

    token_logits = outputs["token_logits"]
    intent_logits = outputs["intent_logits"]

    # 在应用softmax并转换为numpy之前，将张量移动到CPU
    # Move tensors to CPU before applying softmax and converting to numpy
    intent_probabilities = F.softmax(intent_logits.cpu(), dim=-1).numpy()[0]
    token_probabilities = F.softmax(token_logits.cpu(), dim=-1).numpy()[0]

    # 提取每个token的正类（索引1）概率
    # Extract the probabilities for the positive class (index 1) for each token
    token_positive_probs = token_probabilities[:, 1].tolist()

    return intent_probabilities.tolist(), token_positive_probs

# 映射关键词
# Map keywords
def map_keywords(
    input_ids: torch.Tensor, tokenizer: AutoTokenizer, is_keyword: list[bool]
) -> list[str]:
    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    if not len(tokens) == len(is_keyword):
        raise ValueError("Length of tokens and keyword predictions must match")

    if input_ids[0] == tokenizer.cls_token_id:
        tokens = tokens[1:]
        is_keyword = is_keyword[1:]

    if input_ids[-1] == tokenizer.sep_token_id:
        tokens = tokens[:-1]
        is_keyword = is_keyword[:-1]

    unk_token = tokenizer.unk_token
    if unk_token in tokens:
        raise ValueError("Unknown token detected in the input")

    keywords = []
    current_keyword = ""

    for ind, token in enumerate(tokens):
        if is_keyword[ind]:
            if token.startswith("##"):
                current_keyword += token[2:]
            else:
                if current_keyword:
                    keywords.append(current_keyword)
                current_keyword = token
        else:
            # 如果错误预测了关键词的后续token，将其添加到当前关键词以完成它
            # If mispredicted a later token of a keyword, add it to the current keyword
            # to complete it
            if current_keyword:
                if len(current_keyword) > 2 and current_keyword.startswith("##"):
                    current_keyword = current_keyword[2:]

                else:
                    keywords.append(current_keyword)
                    current_keyword = ""

    if current_keyword:
        keywords.append(current_keyword)

    return keywords

# 清理关键词
# Clean keywords
def clean_keywords(keywords: list[str]) -> list[str]:
    cleaned_words = []
    for word in keywords:
        word = word[:-2] if word.endswith("'s") else word
        word = word.replace("/", " ")
        word = word.replace("'", "").replace('"', "")
        cleaned_words.extend([w for w in word.strip().split() if w and not w.isspace()])
    return cleaned_words

# 运行连接器分类
# Run connector classification
def run_connector_classification(req: ConnectorClassificationRequest) -> list[str]:
    tokenizer = get_connector_classifier_tokenizer()
    model = get_local_connector_classifier()

    connector_names = req.available_connectors

    input_ids, attention_mask = tokenize_connector_classification_query(
        connector_names,
        req.query,
        tokenizer,
        model.connector_end_token_id,
    )
    input_ids = input_ids.to(model.device)
    attention_mask = attention_mask.to(model.device)

    global_confidence, classifier_confidence = model(input_ids, attention_mask)

    if global_confidence.item() < 0.5:
        return []

    passed_connectors = []

    for i, connector_name in enumerate(connector_names):
        if classifier_confidence.view(-1)[i].item() > 0.5:
            passed_connectors.append(connector_name)

    return passed_connectors

# 运行分析
# Run analysis
def run_analysis(intent_req: IntentRequest) -> tuple[bool, list[str]]:
    tokenizer = get_intent_model_tokenizer()
    model_input = tokenizer(
        intent_req.query, return_tensors="pt", truncation=False, padding=False
    )

    if len(model_input.input_ids[0]) > 512:
        # 如果用户文本太长，假设它是语义的并保留所有单词
        # If the user text is too long, assume it is semantic and keep all words
        return True, intent_req.query.split()

    intent_probs, token_probs = run_inference(model_input)

    is_keyword_sequence = intent_probs[0] >= intent_req.keyword_percent_threshold

    keyword_preds = [
        token_prob >= intent_req.keyword_percent_threshold for token_prob in token_probs
    ]

    try:
        keywords = map_keywords(model_input.input_ids[0], tokenizer, keyword_preds)
    except Exception as e:
        logger.error(
            f"Failed to extract keywords for query: {intent_req.query} due to {e}"
        )
        # 回退到保留所有单词
        # Fallback to keeping all words
        keywords = intent_req.query.split()

    cleaned_keywords = clean_keywords(keywords)

    return is_keyword_sequence, cleaned_keywords

# 处理连接器分类请求
# Process connector classification request
@router.post("/connector-classification")
async def process_connector_classification_request(
    classification_request: ConnectorClassificationRequest,
) -> ConnectorClassificationResponse:
    if INDEXING_ONLY:
        raise RuntimeError(
            "Indexing model server should not call connector classification endpoint"
        )

    if len(classification_request.available_connectors) == 0:
        return ConnectorClassificationResponse(connectors=[])

    connectors = run_connector_classification(classification_request)
    return ConnectorClassificationResponse(connectors=connectors)

# 处理查询分析请求
# Process query analysis request
@router.post("/query-analysis")
async def process_analysis_request(
    intent_request: IntentRequest,
) -> IntentResponse:
    if INDEXING_ONLY:
        raise RuntimeError("Indexing model server should not call intent endpoint")

    is_keyword, keywords = run_analysis(intent_request)
    return IntentResponse(is_keyword=is_keyword, keywords=keywords)
