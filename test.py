import httpx

SEARCH_ENDPOINT = "http://localhost:8081/search/"
params = {
    "yql": "select * from sources * where userQuery();",
    "query": "apple"
}

try:
    response = httpx.post(SEARCH_ENDPOINT, json=params)
    response.raise_for_status()
    print(f"Status Code: {response.status_code}")
    print(f"Response Body: {response.json()}")
except httpx.HTTPStatusError as e:
    print(f"HTTP Status Error: {e.response.status_code} {e.response.text}")
    # 查看详细的错误信息
    print(f"Detailed Error: {e}")
except Exception as e:
    print(f"An error occurred: {e}")