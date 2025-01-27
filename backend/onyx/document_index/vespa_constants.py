"""
此文件定义了与Vespa搜索引擎相关的常量和配置。
包含了文档索引、搜索端点、字段名称等核心常量的定义。
这些常量被用于在Vespa中进行文档的索引、查询和管理。
"""

from onyx.configs.app_configs import VESPA_CLOUD_URL
from onyx.configs.app_configs import VESPA_CONFIG_SERVER_HOST
from onyx.configs.app_configs import VESPA_HOST
from onyx.configs.app_configs import VESPA_PORT
from onyx.configs.app_configs import VESPA_TENANT_PORT
from onyx.configs.constants import SOURCE_TYPE

# Vespa模式替换模式字符串常量
VESPA_DIM_REPLACEMENT_PAT = "VARIABLE_DIM"  # 维度替换模式
DANSWER_CHUNK_REPLACEMENT_PAT = "DANSWER_CHUNK_NAME"  # 文档块名称替换模式
DOCUMENT_REPLACEMENT_PAT = "DOCUMENT_REPLACEMENT"  # 文档替换模式
SEARCH_THREAD_NUMBER_PAT = "SEARCH_THREAD_NUMBER"  # 搜索线程数替换模式
DATE_REPLACEMENT = "DATE_REPLACEMENT"  # 日期替换模式
TENANT_ID_PAT = "TENANT_ID_REPLACEMENT"  # 租户ID替换模式

# Vespa租户ID字段定义
TENANT_ID_REPLACEMENT = """field tenant_id type string {
            indexing: summary | attribute
            rank: filter
            attribute: fast-search
        }"""

# Vespa配置服务器URL配置
VESPA_CONFIG_SERVER_URL = (
    VESPA_CLOUD_URL or f"http://{VESPA_CONFIG_SERVER_HOST}:{VESPA_TENANT_PORT}"
)
VESPA_APPLICATION_ENDPOINT = f"{VESPA_CONFIG_SERVER_URL}/application/v2"

# Vespa主搜索应用容器URL
VESPA_APP_CONTAINER_URL = VESPA_CLOUD_URL or f"http://{VESPA_HOST}:{VESPA_PORT}"

# 文档ID端点定义
DOCUMENT_ID_ENDPOINT = (
    f"{VESPA_APP_CONTAINER_URL}/document/v1/default/{{index_name}}/docid"
)

# 搜索端点定义
SEARCH_ENDPOINT = f"{VESPA_APP_CONTAINER_URL}/search/"

# 性能和限制相关常量
NUM_THREADS = 32  # 由于Vespa不支持批量插入/更新，使用线程进行处理
MAX_ID_SEARCH_QUERY_SIZE = 400  # 最大ID搜索查询大小
MAX_OR_CONDITIONS = 10  # 最大OR条件数量
# 超时时间从500ms增加到3s，因为观察到较多超时情况
# 长期来看，我们计划提升Vespa的性能以便将超时时间恢复到默认值
VESPA_TIMEOUT = "3s"  # Vespa操作超时时间
BATCH_SIZE = 128  # Vespa批处理大小

# Vespa文档字段定义
TENANT_ID = "tenant_id"  # 租户ID字段
DOCUMENT_ID = "document_id"  # 文档ID字段
CHUNK_ID = "chunk_id"  # 文档块ID字段
BLURB = "blurb"  # 文档简介字段
CONTENT = "content"  # 文档内容字段
SOURCE_LINKS = "source_links"  # 源链接字段
SEMANTIC_IDENTIFIER = "semantic_identifier"  # 语义标识符字段
TITLE = "title"  # 标题字段
SKIP_TITLE_EMBEDDING = "skip_title"  # 跳过标题嵌入标志
SECTION_CONTINUATION = "section_continuation"  # 章节连续标志
EMBEDDINGS = "embeddings"  # 嵌入向量字段
TITLE_EMBEDDING = "title_embedding"  # 标题嵌入向量字段
ACCESS_CONTROL_LIST = "access_control_list"  # 访问控制列表
DOCUMENT_SETS = "document_sets"  # 文档集合
LARGE_CHUNK_REFERENCE_IDS = "large_chunk_reference_ids"  # 大块引用ID
METADATA = "metadata"  # 元数据
METADATA_LIST = "metadata_list"  # 元数据列表
METADATA_SUFFIX = "metadata_suffix"  # 元数据后缀
BOOST = "boost"  # 提升因子
DOC_UPDATED_AT = "doc_updated_at"  # 文档更新时间（以epoch秒为单位）
PRIMARY_OWNERS = "primary_owners"  # 主要所有者
SECONDARY_OWNERS = "secondary_owners"  # 次要所有者
RECENCY_BIAS = "recency_bias"  # 最近偏好
HIDDEN = "hidden"  # 隐藏标志

# Vespa特定字段，用于关键词匹配和章节高亮
CONTENT_SUMMARY = "content_summary"  # 内容摘要字段

# Vespa基础YQL查询语句
YQL_BASE = (
    f"select "
    f"documentid, "
    f"{DOCUMENT_ID}, "
    f"{CHUNK_ID}, "
    f"{BLURB}, "
    f"{CONTENT}, "
    f"{SOURCE_TYPE}, "
    f"{SOURCE_LINKS}, "
    f"{SEMANTIC_IDENTIFIER}, "
    f"{TITLE}, "
    f"{SECTION_CONTINUATION}, "
    f"{BOOST}, "
    f"{HIDDEN}, "
    f"{DOC_UPDATED_AT}, "
    f"{PRIMARY_OWNERS}, "
    f"{SECONDARY_OWNERS}, "
    f"{LARGE_CHUNK_REFERENCE_IDS}, "
    f"{METADATA}, "
    f"{METADATA_SUFFIX}, "
    f"{CONTENT_SUMMARY} "
    f"from {{index_name}} where "
)
