"""
此模块用于解析和处理OpenAPI规范文档。
主要功能包括：
- 解析OpenAPI文档中的路径和方法定义
- 验证OpenAPI规范的有效性
- 构建API请求URL
- 转换OpenAPI规范为工具定义
"""

from typing import Any
from typing import cast

from pydantic import BaseModel

REQUEST_BODY = "requestBody"


class PathSpec(BaseModel):
    """
    表示OpenAPI规范中的路径规格。
    包含路径字符串和与该路径相关的HTTP方法信息。
    """
    path: str  # API路径
    methods: dict[str, Any]  # 该路径支持的HTTP方法及其详细信息


class MethodSpec(BaseModel):
    """
    表示OpenAPI规范中的方法规格。
    包含API方法的完整定义，包括名称、描述、路径、HTTP方法和详细规范。
    """
    name: str  # 操作ID
    summary: str  # 方法描述
    path: str  # API路径
    method: str  # HTTP方法
    spec: dict[str, Any]  # 方法的完整规范

    def get_request_body_schema(self) -> dict[str, Any]:
        """
        获取请求体的JSON Schema定义。
        
        返回值:
            dict: 请求体的schema定义，如果没有请求体则返回空字典
        
        异常:
            ValueError: 当内容类型不是application/json时抛出
        """
        content = self.spec.get("requestBody", {}).get("content", {})
        if "application/json" in content:
            return content["application/json"].get("schema")

        if content:
            raise ValueError(
                f"不支持的内容类型: '{list(content.keys())[0]}'. "
                f"仅支持 'application/json'。"
            )

        return {}

    def get_query_param_schemas(self) -> list[dict[str, Any]]:
        """
        获取查询参数的schema列表。
        
        返回值:
            list: 包含所有查询参数schema的列表
        """
        return [
            param
            for param in self.spec.get("parameters", [])
            if "schema" in param and "in" in param and param["in"] == "query"
        ]

    def get_path_param_schemas(self) -> list[dict[str, Any]]:
        """
        获取路径参数的schema列表。
        
        返回值:
            list: 包含所有路径参数schema的列表
        """
        return [
            param
            for param in self.spec.get("parameters", [])
            if "schema" in param and "in" in param and param["in"] == "path"
        ]

    def build_url(
        self, base_url: str, path_params: dict[str, str], query_params: dict[str, str]
    ) -> str:
        """
        构建完整的API URL。
        
        参数:
            base_url: 基础URL
            path_params: 路径参数字典
            query_params: 查询参数字典
        
        返回值:
            str: 构建好的完整URL
            
        异常:
            ValueError: 当缺少必要的路径参数时抛出
        """
        url = f"{base_url}{self.path}"
        try:
            url = url.format(**path_params)
        except KeyError as e:
            raise ValueError(f"Missing path parameter: {e}")
        if query_params:
            url += "?"
            for param, value in query_params.items():
                url += f"{param}={value}&"
            url = url[:-1]
        return url

    def to_tool_definition(self) -> dict[str, Any]:
        """
        将方法规格转换为工具定义格式。
        
        返回值:
            dict: 符合工具定义格式的字典
        """
        tool_definition: Any = {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.summary,
                "parameters": {"type": "object", "properties": {}},
            },
        }

        request_body_schema = self.get_request_body_schema()
        if request_body_schema:
            tool_definition["function"]["parameters"]["properties"][
                REQUEST_BODY
            ] = request_body_schema

        query_param_schemas = self.get_query_param_schemas()
        if query_param_schemas:
            tool_definition["function"]["parameters"]["properties"].update(
                {param["name"]: param["schema"] for param in query_param_schemas}
            )

        path_param_schemas = self.get_path_param_schemas()
        if path_param_schemas:
            tool_definition["function"]["parameters"]["properties"].update(
                {param["name"]: param["schema"] for param in path_param_schemas}
            )
        return tool_definition

    def validate_spec(self) -> None:
        """
        验证方法规格的有效性。
        检查URL构建、请求体格式和HTTP方法的有效性。
        
        异常:
            ValueError: 当验证失败时抛出
        """
        # 验证URL构造
        # Validate url construction
        path_param_schemas = self.get_path_param_schemas()
        dummy_path_dict = {param["name"]: "value" for param in path_param_schemas}
        query_param_schemas = self.get_query_param_schemas()
        dummy_query_dict = {param["name"]: "value" for param in query_param_schemas}
        self.build_url("", dummy_path_dict, dummy_query_dict)

        # 确保请求体不会抛出异常
        # Make sure request body doesn't throw an exception
        self.get_request_body_schema()

        # 确保HTTP方法是有效的
        # Ensure the method is valid
        if not self.method:
            raise ValueError("未指定HTTP方法。")  # HTTP method is not specified.
        if self.method.upper() not in ["GET", "POST", "PUT", "DELETE", "PATCH"]:
            raise ValueError(f"不支持的HTTP方法 '{self.method}'。")  # HTTP method is not supported.


"""路径级工具函数"""
"""Path-level utils 路径级工具函数"""


def openapi_to_path_specs(openapi_spec: dict[str, Any]) -> list[PathSpec]:
    """
    将OpenAPI规范转换为路径规格列表。
    
    参数:
        openapi_spec: OpenAPI规范文档
        
    返回值:
        list: PathSpec对象列表
    """
    path_specs = []

    for path, methods in openapi_spec.get("paths", {}).items():
        path_specs.append(PathSpec(path=path, methods=methods))

    return path_specs


"""方法级工具函数"""
"""Method-level utils 方法级工具函数"""


def openapi_to_method_specs(openapi_spec: dict[str, Any]) -> list[MethodSpec]:
    """
    将OpenAPI规范转换为方法规格列表。
    
    参数:
        openapi_spec: OpenAPI规范文档
        
    返回值:
        list: MethodSpec对象列表
        
    异常:
        ValueError: 当缺少必要信息或未找到方法时抛出
    """
    path_specs = openapi_to_path_specs(openapi_spec)

    method_specs = []
    for path_spec in path_specs:
        for method_name, method in path_spec.methods.items():
            name = method.get("operationId")
            if not name:
                raise ValueError(
                    f"Operation ID is not specified for {method_name.upper()} {path_spec.path}"
                )

            summary = method.get("summary") or method.get("description")
            if not summary:
                raise ValueError(
                    f"Summary is not specified for {method_name.upper()} {path_spec.path}"
                )

            method_specs.append(
                MethodSpec(
                    name=name,
                    summary=summary,
                    path=path_spec.path,
                    method=method_name,
                    spec=method,
                )
            )

    if not method_specs:
        raise ValueError("No methods found in OpenAPI schema")

    return method_specs


def openapi_to_url(openapi_schema: dict[str, dict | str]) -> str:
    """
    从OpenAPI schema的servers部分提取URL。
    Extract URLs from the servers section of an OpenAPI schema.
    
    参数:
        openapi_schema: OpenAPI规范文档
        
    返回值:
        str: 基础URL
        
    异常:
        ValueError: 当未找到恰好一个URL时抛出
    """
    urls: list[str] = []

    servers = cast(list[dict[str, Any]], openapi_schema.get("servers", []))
    for server in servers:
        url = server.get("url")
        if url:
            urls.append(url)

    if len(urls) != 1:
        raise ValueError(
            f"Expected exactly one URL in OpenAPI schema, but found {urls}"
        )

    return urls[0]


def validate_openapi_schema(schema: dict[str, Any]) -> None:
    """
    验证给定的JSON schema是否为有效的OpenAPI schema。
    Validate the given JSON schema as an OpenAPI schema.
    
    参数:
        schema: 要验证的JSON schema
        
    异常:
        ValueError: 当schema验证失败时抛出
    """

    # 检查基本结构
    # check basic structure
    if "info" not in schema:
        raise ValueError("OpenAPI schema中必须包含`info`部分")

    info = schema["info"]
    if "title" not in info:
        raise ValueError("OpenAPI schema的`info`部分必须包含`title`")
    if "description" not in info:
        raise ValueError("OpenAPI schema的`info`部分必须包含`description`")

    if "openapi" not in schema:
        raise ValueError("必须指定OpenAPI schema版本的`openapi`字段")
    openapi_version = schema["openapi"]
    if not openapi_version.startswith("3."):
        raise ValueError(f"不支持OpenAPI版本 '{openapi_version}'")

    if "paths" not in schema:
        raise ValueError("OpenAPI schema中必须包含`paths`部分")

    url = openapi_to_url(schema)
    if not url:
        raise ValueError("OpenAPI schema的`servers`部分中未包含有效的URL")

    method_specs = openapi_to_method_specs(schema)
    for method_spec in method_specs:
        method_spec.validate_spec()
