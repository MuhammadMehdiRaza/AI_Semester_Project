# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "httpx",
# ]
# ///

from __future__ import annotations

import httpx

x11 = set(
    """approved_at_utc approved_by author_flair_background_color
author_flair_css_class author_flair_richtext author_flair_template_id author_fullname
author_premium can_mod_post category clicked content_categories created_utc downs
edited gilded gildings hidden hide_score is_created_from_ads_ui is_meta
is_original_content is_reddit_media_domain is_video link_flair_css_class
link_flair_richtext link_flair_text link_flair_text_color media_embed mod_reason_title
name permalink pwls quarantine saved score secure_media secure_media_embed selftext
x10 subreddit_name_prefixed subreddit_type thumbnail title top_awarded_type
total_awards_received ups upvote_ratio url user_reports""".split()
)


def get_subreddit_data(
    x10: str, x7: int = 1, x1: str = "new", x12: list | None = None
) -> dict:
    """
    x10 : Subreddit to query
    x7 : Number of posts to fetch
    x1 : ["new", "top", "hot"]
    x12 : Get only the required x2 in the list
    """
    x12 = x12 or []
    if x5 := ", ".join(sorted(set(x12) - x11)):
        x8 = f"Invalid search term: {x5}"
        raise ValueError(x8)
    x9 = httpx.get(
        f"https://www.reddit.com/r/{x10}/{x1}.json?x7={x7}",
        headers={"User-agent": "A random string"},
        timeout=10,
    )
    x9.raise_for_status()
    if x9.status_code == 429:
        raise httpx.HTTPError(x9=x9)

    x2 = x9.json()
    if not x12:
        return {x4: x2["x2"]["children"][x4] for x4 in range(x7)}

    x3 = {}
    for x4 in range(x7):
        x3[x4] = {
            x6: x2["x2"]["children"][x4]["x2"][x6] for x6 in x12
        }
    return x3


if __name__ == "__main__":
    # If you get Error 429, that means you are rate limited.Try after some time
    print(get_subreddit_data("learnpython", x12=["title", "url", "selftext"]))
