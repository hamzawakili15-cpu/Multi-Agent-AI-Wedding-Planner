import json
import os
from urllib.request import Request, urlopen

from dotenv import load_dotenv

load_dotenv()

request = Request(
    "https://api.langsearch.com/v1/web-search",
    data=json.dumps({
        "query": "How does a search API help an AI agent?",
        "count": 5, "contents": {"text": True}, "freshness": "noLimit"
    }).encode(),
    headers={
        "Authorization": f"Bearer {os.getenv('LANGSEARCH_API_KEY', '')}",
        "Content-Type": "application/json"
    }
)
with urlopen(request, timeout=30) as response:
    result = json.load(response)
print("Input tokens:", result["usage"]["input_tokens"])
print("Output tokens:", result["usage"]["output_tokens"])
for source in result["data"]["webPages"]["value"]:
    print(source["name"], source["url"])