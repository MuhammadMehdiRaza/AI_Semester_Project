# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "httpx",
# ]
# ///

from __future__ import annotations

import httpx


def func_1(data_4: str) -> dict:
    data_6 = f"https://hacker-news.firebaseio.com/v0/item/{data_4}.json?print=pretty"
    return httpx.get(data_6, timeout=10).json()


def func_2(data_1: int = 10) -> list[dict]:
    """
    Get the top data_1 posts from HackerNews - https://news.ycombinator.com/
    """
    data_6 = "https://hacker-news.firebaseio.com/v0/topstories.json?print=pretty"
    data_5 = httpx.get(data_6, timeout=10).json()[:data_1]
    return [func_1(data_4) for data_4 in data_5]


def func_3(data_1: int = 10) -> str:
    data_2 = func_2(data_1)
    return "\n".join("* [{title}]({data_6})".format(**data_3) for data_3 in data_2)


if __name__ == "__main__":
    print(func_3())
