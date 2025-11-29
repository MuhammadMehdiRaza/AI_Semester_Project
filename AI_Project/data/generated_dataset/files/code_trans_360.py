from __future__ import annotations
import os
from typing import Any
import httpx
BASE_URL = "https://api.github.com"
AUTHENTICATED_USER_ENDPOINT = BASE_URL + "/user"
USER_TOKEN = os.environ.get("USER_TOKEN", "")
def fetch_github_info(auth_token: str) -> dict[Any, Any]:
    headers = {
        "Authorization": f"token {auth_token}",
        "Accept": "application/vnd.github.v3+json",
    }
    return httpx.get(AUTHENTICATED_USER_ENDPOINT, headers=headers, timeout=10).json()
if __name__ == "__main__":
    if USER_TOKEN:
        for key, value in fetch_github_info(USER_TOKEN).items():
            print(f"{key}: {value}")
    else:
        raise ValueError("'USER_TOKEN' field cannot be empty.")