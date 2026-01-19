import requests
import json


def fetch_scholar_inbox_data(cookie_string: str):
    session = requests.Session()
    session.headers.update({
        "Accept": "*/*",
        "Accept-Encoding": "gzip, deflate, br, zstd",
        "Accept-Language": "en,zh-CN;q=0.9,zh;q=0.8",
        "Connection": "keep-alive",
        "User-Agent": "Mozilla/5.0"
    })
    
    cookie_string = (
        "dumb_string"
    ) if not cookie_string else cookie_string

    session.headers.update({
        "Cookie": cookie_string
    })

    url = "https://api.scholar-inbox.com/api/"
    res = session.get(url)
    if res.status_code != 200:
        raise Exception(f"Failed to fetch data: {res.status_code}")
    
    data = res.json()
    papers = []
    for entry in data.get("digest_df", []):
        arxiv_id = entry.get("arxiv_id")
        paper = {
            "title": entry.get("title"),
            "authors": [a.strip() for a in entry.get("authors", "").split(",") if a.strip()],
            "arxiv_url": f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else None,
            "abstract": entry.get("abstract"),
        }
        papers.append(paper)
    
    return papers